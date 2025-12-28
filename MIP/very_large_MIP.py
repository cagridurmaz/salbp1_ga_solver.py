#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SALBP Auto-Solver (Gurobi) — Literature Metrics Focus
"""
import os, re, csv, json, math, time, statistics
from typing import List, Tuple, Dict, Optional
from pathlib import Path

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None
    GRB = None

try:
    import pandas as pd
except ImportError:
    pd = None


# ---------------------------
# Parsing (.alb or simple Excel)
# ---------------------------

def parse_alb(path: Path) -> Dict:
    text = path.read_text (encoding='utf-8', errors='ignore')
    lines = [ln.strip () for ln in text.splitlines ()]

    def find(tag: str) -> int:
        for i, ln in enumerate (lines):
            if ln.lower () == tag.lower ():
                return i
        return -1

    idx_n = find ("<number of tasks>")
    idx_c = find ("<cycle time>")
    idx_t = find ("<task times>")
    idx_p = find ("<precedence relations>")
    idx_end = find ("<end>")
    if idx_n < 0 or idx_c < 0 or idx_t < 0 or idx_end < 0:
        raise ValueError (f"Unexpected .alb format: {path.name}")

    n = int (re.sub (r"[^\d]", "", lines[idx_n + 1]))
    C_raw = lines[idx_c + 1]
    try:
        C = int (C_raw)
    except Exception:
        C = int (float (C_raw.replace (",", ".")))

    times: Dict[int, int] = {}
    i = idx_t + 1
    while i < len (lines) and lines[i] != "" and not lines[i].startswith ("<"):
        p = lines[i].split ()
        if len (p) >= 2 and p[0].isdigit ():
            times[int (p[0])] = int (p[1])
        i += 1

    edges: List[Tuple[int, int]] = []
    if idx_p != -1:
        j = idx_p + 1
        while j < len (lines) and lines[j] != "" and not lines[j].startswith ("<"):
            if re.match (r"^\d+\s*,\s*\d+$", lines[j]):
                a, b = [int (x) for x in lines[j].split (",")]
                edges.append ((a, b))
            j += 1
    else:
        for ln in lines[i:]:
            if re.match (r"^\d+\s*,\s*\d+$", ln):
                a, b = [int (x) for x in ln.split (",")]
                edges.append ((a, b))

    if len (times) != n:
        raise ValueError (f"{path.name}: parsed {len (times)} task times but expected {n}.")

    return {"name": path.stem, "n": n, "C": C, "times": times, "edges": edges, "source": str (path)}


def parse_excel_simple(path: Path) -> Optional[Dict]:
    if pd is None:
        return None
    try:
        xls = pd.ExcelFile (path)
    except Exception:
        return None
    if "tasks" not in xls.sheet_names or "precedence" not in xls.sheet_names:
        return None
    tasks_df = pd.read_excel (path, sheet_name="tasks")
    pred_df = pd.read_excel (path, sheet_name="precedence")
    C = None
    if "params" in xls.sheet_names:
        params_df = pd.read_excel (path, sheet_name="params", header=None)
        try:
            C_cand = params_df.iloc[0, 0]
            if pd.notna (C_cand):
                C = int (float (C_cand))
        except Exception:
            C = None
    times = {int (r["task"]): int (r["time"]) for _, r in tasks_df.iterrows ()}
    edges = [(int (r["i"]), int (r["j"])) for _, r in pred_df.iterrows ()]
    n = len (times)
    return {"name": path.stem, "n": n, "C": C if C is not None else sum (times.values ()),
            "times": times, "edges": edges, "source": str (path)}


# ---------------------------
# Metrics helpers
# ---------------------------

def compute_si(station_loads: dict) -> float:
    """Smoothness Index: sqrt( sum_k (tt_max - tt_k)^2 / m ) — Baykasoglu (2006)."""
    if not station_loads:
        return 0.0
    loads = list (station_loads.values ())
    m = len (loads)
    tt_max = max (loads) if loads else 0
    s = sum ((tt_max - v) * (tt_max - v) for v in loads)
    return math.sqrt (s / m) if m > 0 else 0.0


def compute_eff(times_sum: int, m_used: int, C: float) -> float:
    """Line efficiency: sum(t_i)/(m * C)."""
    denom = (m_used * C)
    return (times_sum / denom) if denom > 0 else 0.0


# ---------------------------
# Multi-run solver
# ---------------------------

def solve_instance_runs(data: Dict, mode: str, stations_for_salbp2: Optional[int], runs: int, time_limit: Optional[int],
                        verbose: bool):
    if gp is None:
        raise RuntimeError ("gurobipy is not available. Install Gurobi.")

    Tsum = sum (data["times"].values ())
    results = []

    for r in range (1, runs + 1):
        seed = 1000 + r
        if mode.upper () == "SALBP1":
            n = data["n"];
            C = int (data["C"]);
            times = data["times"];
            edges = data["edges"];
            name = data["name"]
            Smax = n;
            Sset = range (1, Smax + 1);
            Iset = range (1, n + 1)

            m = gp.Model (f"SALBP1_{name}_r{r}")
            if time_limit: m.Params.TimeLimit = time_limit
            m.Params.OutputFlag = 1 if verbose else 0
            m.Params.Seed = seed

            # === YENİ PERFORMANS PARAMETRELERİ ===
            # 1: Hızlıca olurlu (feasible) çözüm bulmaya odaklan
            m.Params.MIPFocus = 1
            # Zamanın %50'sini çözüm bulma sezgisellerine ayır
            m.Params.Heuristics = 0.5
            # ======================================

            x = m.addVars (Iset, Sset, vtype=GRB.BINARY, name="x")
            y = m.addVars (Sset, vtype=GRB.BINARY, name="y")

            m.addConstrs ((gp.quicksum (x[i, s] for s in Sset) == 1 for i in Iset), name="assign")
            m.addConstrs ((gp.quicksum (times[i] * x[i, s] for i in Iset) <= C * y[s] for s in Sset), name="capacity")

            # === VERİMLİ ÖNCELİK KISITLARI ===
            for (i, j) in edges:
                m.addConstr (
                    gp.quicksum (s * x[i, s] for s in Sset) <= gp.quicksum (s * x[j, s] for s in Sset),
                    name=f"prec_{i}_{j}"
                )
            # ====================================

            # === SİMETRİ KIRMA KISITLARI ===
            m.addConstrs ((y[s] >= y[s + 1] for s in Sset if s < Smax), name="symmetry")

            m.setObjective (gp.quicksum (y[s] for s in Sset), GRB.MINIMIZE)

            t0 = time.time ();
            m.optimize ();
            t1 = time.time ()

            # === YENİ HATA KONTROLÜ VE RAPORLAMA ===
            station_loads = {}
            assign = {}

            # Alt sınırı (Best Bound) her zaman al
            try:
                obj_bound = m.ObjBoundC
            except AttributeError:
                obj_bound = m.ObjBound

            # Çözüm bulunamazsa kullanılacak varsayılan (en kötü) değerler
            m_used = float ('inf')
            si = float ('inf')
            eff = 0.0
            C_val = C  # SALBP1 için C sabittir

            if m.SolCount > 0:
                # Çözüm bulundu, değerleri güvenle oku
                # m.ObjVal objektif fonk. değeridir

                for s in Sset:
                    if y[s].X > 0.5:
                        load = sum (times[i] for i in Iset if x[i, s].X > 0.5)
                        station_loads[s] = load
                for i in Iset:
                    for s in Sset:
                        if x[i, s].X > 0.5: assign[i] = s; break

                m_used = len (station_loads)  # Gerçek istasyon sayısı
                si = compute_si (station_loads)
                eff = compute_eff (Tsum, m_used, C)
            else:
                # Çözüm bulunamadı (TimeLimit vb.)
                if verbose:
                    print (f"[{name}] Run {r}: Zaman asimi, cozum bulunamadi (SolCount = 0). BestBound={obj_bound:.2f}")

            results.append ({
                "run": r, "seed": seed, "status": m.Status,
                "m": m_used, "obj_bound": obj_bound, "sol_count": m.SolCount,
                "SI": si, "Eff": eff, "runtime_sec": (t1 - t0),
                "assign": assign, "station_loads": station_loads, "C": C_val
            })
            # === HATA KONTROLÜ SONU ===

        else:
            # --- SALBP2 Modu ---
            if stations_for_salbp2 is None:
                raise ValueError ("stations_for_salbp2 must be provided for SALBP2 mode.")
            n = data["n"];
            times = data["times"];
            edges = data["edges"];
            name = data["name"]
            S = stations_for_salbp2;
            Sset = range (1, S + 1);
            Iset = range (1, n + 1)

            m = gp.Model (f"SALBP2_{name}_r{r}")
            if time_limit: m.Params.TimeLimit = time_limit
            m.Params.OutputFlag = 1 if verbose else 0
            m.Params.Seed = seed

            # SALBP2 için de performans parametreleri
            m.Params.MIPFocus = 1
            m.Params.Heuristics = 0.5

            x = m.addVars (Iset, Sset, vtype=GRB.BINARY, name="x")

            # Cvar için iyi bir alt sınır (lb) belirlemek çözüme yardımcı olur
            min_c = max (times.values ()) if times else 0.0
            Cvar = m.addVar (vtype=GRB.CONTINUOUS, lb=min_c, name="C")

            m.addConstrs ((gp.quicksum (x[i, s] for s in Sset) == 1 for i in Iset), name="assign")
            m.addConstrs ((gp.quicksum (times[i] * x[i, s] for i in Iset) <= Cvar for s in Sset), name="capacity")

            # Verimli Öncelik Kısıtları
            for (i, j) in edges:
                m.addConstr (
                    gp.quicksum (s * x[i, s] for s in Sset) <= gp.quicksum (s * x[j, s] for s in Sset),
                    name=f"prec_{i}_{j}"
                )

            m.setObjective (Cvar, GRB.MINIMIZE)
            t0 = time.time ();
            m.optimize ();
            t1 = time.time ()

            # === YENİ HATA KONTROLÜ VE RAPORLAMA (SALBP2) ===
            station_loads = {}
            assign = {}

            try:
                obj_bound = m.ObjBoundC
            except AttributeError:
                obj_bound = m.ObjBound

            Cval = float ('inf')
            si = float ('inf')
            eff = 0.0

            if m.SolCount > 0:
                Cval = Cvar.X
                for s in Sset:
                    load = sum (times[i] for i in Iset if x[i, s].X > 0.5)
                    station_loads[s] = load
                for i in Iset:
                    for s in Sset:
                        if x[i, s].X > 0.5: assign[i] = s; break
                si = compute_si (station_loads)
                eff = compute_eff (Tsum, S, Cval if Cval else 0.0)
            else:
                if verbose:
                    print (f"[{name}] Run {r}: Zaman asimi, cozum bulunamadi (SolCount = 0). BestBound={obj_bound:.2f}")

            results.append ({
                "run": r, "seed": seed, "status": m.Status,
                "m": S, "obj_bound": obj_bound, "sol_count": m.SolCount,
                "SI": si, "Eff": eff, "runtime_sec": (t1 - t0),
                "assign": assign, "station_loads": station_loads, "C": Cval
            })

    # Aggregates per literature
    # Sadece çözüm bulunan ('inf' olmayan) çalıştırmaların istatistiklerini al
    ms_sol = [r["m"] for r in results if r["sol_count"] > 0]
    sis_sol = [r["SI"] for r in results if r["sol_count"] > 0]
    effs_sol = [r["Eff"] for r in results if r["sol_count"] > 0]
    rtimes = [r["runtime_sec"] for r in results]  # Runtime her zaman geçerlidir
    bounds = [r["obj_bound"] for r in results]  # Bound her zaman geçerlidir

    smallest_m = min (ms_sol) if ms_sol else None
    SI_mean = statistics.mean (sis_sol) if sis_sol else None
    Eff_mean = statistics.mean (effs_sol) if effs_sol else None
    mu_m = statistics.mean (ms_sol) if ms_sol else None
    sigma_m = statistics.pstdev (ms_sol) if len (ms_sol) > 1 else 0.0 if ms_sol else None
    mu_rt = statistics.mean (rtimes) if rtimes else None
    sigma_rt = statistics.pstdev (rtimes) if len (rtimes) > 1 else 0.0 if rtimes else None

    # Yeni Metrikler
    # Tüm çalıştırmalar arasındaki en iyi (en yüksek) alt sınırı bul
    best_lower_bound = max (bounds) if bounds else None
    solutions_found_runs = len (ms_sol)

    # choose best (min m, then min SI)
    best = None
    for r in results:
        # (inf, inf) karşılaştırması güvenle çalışır
        if best is None or (r["m"], r["SI"]) < (best["m"], best["SI"]):
            best = r

    aggregates = {"Smallest_m": smallest_m, "SI_mean": SI_mean, "Eff_mean": Eff_mean,
                  "mu_m": mu_m, "sigma_m": sigma_m, "mu_runtime": mu_rt, "sigma_runtime": sigma_rt,
                  "best_run": best["run"] if best and best["sol_count"] > 0 else None,
                  "Best_Lower_Bound": best_lower_bound,
                  "Solutions_Found_Runs": solutions_found_runs
                  }
    return results, aggregates


# ---------------------------
# Orchestrator
# ---------------------------

def solve_folder(input_dir: str,
                 mode: str = "SALBP1",
                 time_limit: Optional[int] = None,
                 stations_for_salbp2: Optional[int] = None,
                 out_dir: Optional[str] = None,
                 verbose: bool = True,
                 runs: int = 5) -> Dict[str, Dict]:
    p = Path (input_dir);
    assert p.exists () and p.is_dir (), f"Input dir not found: {input_dir}"
    outp = Path (out_dir) if out_dir else p / "solutions";
    outp.mkdir (parents=True, exist_ok=True)

    files = list (p.glob ("*.alb")) + list (p.glob ("*.xlsx"))
    if not files:
        print (f"No .alb or .xlsx files found in {input_dir}")
        return {}

    summary_path = outp / "metrics_summary.csv"
    # ÖZET CSV DOSYASINA YENİ SÜTUNLAR EKLENDİ
    summary_header = ["instance", "Smallest_m", "SI_mean", "Eff_mean", "mu_m", "sigma_m",
                      "mu_runtime", "sigma_runtime", "best_run",
                      "Best_Lower_Bound", "Solutions_Found_Runs"]

    with open (summary_path, "w", newline="") as sf:
        sw = csv.writer (sf)
        sw.writerow (summary_header)

        out_json = {}
        for f in sorted (files):  # Dosyaları sıralı işlemek tutarlılık sağlar
            if f.name.startswith ("~"): continue  # Geçici excel dosyalarını atla
            print (f"--- Processing {f.name} (Mode: {mode}) ---")
            try:
                if f.suffix.lower () == ".alb":
                    data = parse_alb (f)
                elif f.suffix.lower () == ".xlsx":
                    data = parse_excel_simple (f)
                    if data is None:
                        print (f"Skipping {f.name}: Not a valid simple excel format.")
                        continue
                else:
                    continue

                run_results, agg = solve_instance_runs (data, mode, stations_for_salbp2, runs, time_limit, verbose)

                # per-run CSV
                # ÇALIŞTIRMA BAZLI CSV'YE YENİ SÜTUNLAR EKLENDİ
                per_path = outp / f"{data['name']}_runs.csv"
                run_header = ["run", "seed", "status", "m", "obj_bound", "sol_count", "SI", "Eff", "runtime_sec"]
                with open (per_path, "w", newline="") as pf:
                    pw = csv.writer (pf)
                    pw.writerow (run_header)
                    for r in run_results:
                        pw.writerow ([r["run"], r["seed"], r["status"],
                                      r["m"], r["obj_bound"], r["sol_count"],
                                      r["SI"], r["Eff"], r["runtime_sec"]])

                # best assignment & loads CSV
                best = next ((r for r in run_results if r["run"] == agg["best_run"]), None)

                # Sadece 'best' bir çözümse (None değilse) dosyaları yaz
                if best:
                    with open (outp / f"{data['name']}_assignment.csv", "w", newline="") as cf:
                        w = csv.writer (cf);
                        w.writerow (["task", "station"])
                        for i in sorted (best["assign"].keys ()):
                            w.writerow ([i, best["assign"][i]])
                    with open (outp / f"{data['name']}_station_loads.csv", "w", newline="") as lf:
                        w = csv.writer (lf);
                        w.writerow (["station", "load", "cycle_time"])
                        for s in sorted (best["station_loads"].keys ()):
                            w.writerow ([s, best["station_loads"][s], best["C"]])

                # summary line
                # ÖZET SATIRINA YENİ DEĞERLER EKLENDİ
                sw.writerow ([data["name"], agg["Smallest_m"], agg["SI_mean"], agg["Eff_mean"],
                              agg["mu_m"], agg["sigma_m"], agg["mu_runtime"], agg["sigma_runtime"],
                              agg["best_run"], agg["Best_Lower_Bound"], agg["Solutions_Found_Runs"]])

                sf.flush ()  # Her dosyadan sonra diske yazmayı garantile

                out_json[data["name"]] = {"aggregate": agg, "best_run_data": best}
            except Exception as e:
                print (f"!!! ERROR on file {f.name}: {e}")
                import traceback
                traceback.print_exc ()
                out_json[f.stem] = {"error": str (e), "file": str (f)}

    with open (outp / "summary.json", "w", encoding="utf-8") as jf:
        json.dump (out_json, jf, indent=2, ensure_ascii=False)

    print (f"--- All instances processed. Summary saved to: {summary_path} ---")
    return out_json


def main():
    import argparse
    parser = argparse.ArgumentParser (description="SALBP solver with literature metrics.")
    parser.add_argument ("--input_dir", type=str, default=".", help="Folder containing .alb/.xlsx")
    parser.add_argument ("--mode", type=str, default="SALBP1", choices=["SALBP1", "SALBP2"])
    parser.add_argument ("--time_limit", type=int, default=100, help="Time limit per run (sec)")
    parser.add_argument ("--stations", type=int, default=None, help="#stations for SALBP2")
    # === AŞAĞIDAKİ SATIR DÜZELTİLDİ ===
    parser.add_argument ("--out_dir", type=str, default=None, help="Output folder (default: input_dir/solutions)")
    parser.add_argument ("--quiet", action="store_true")
    parser.add_argument ("--runs", type=int, default=1, help="number of runs for metrics")
    args = parser.parse_args ()

    if gp is None:
        print ("ERROR: gurobipy library not found. Please install Gurobi.")
        return
    if pd is None:
        print ("WARNING: pandas library not found. .xlsx parsing will be disabled.")

    res = solve_folder (args.input_dir, mode=args.mode, time_limit=args.time_limit, stations_for_salbp2=args.stations,
                        out_dir=args.out_dir, verbose=not args.quiet, runs=args.runs)

    print ("\n--- FINAL SUMMARY ---")
    for name, obj in res.items ():
        if "error" in obj:
            print (f"[{name}] ERROR: {obj['error']}")
        else:
            agg = obj["aggregate"]
            if agg["Solutions_Found_Runs"] > 0:
                print (
                    f"[{name}] OK ({agg['Solutions_Found_Runs']}/{args.runs} runs solved). Smallest_m={agg['Smallest_m']} | Best_Bound={agg['Best_Lower_Bound']:.2f} | mu_runtime={agg['mu_runtime']:.3f}s")
            else:
                print (
                    f"[{name}] NO_SOLUTION ({agg['Solutions_Found_Runs']}/{args.runs} runs solved). Best_Bound={agg['Best_Lower_Bound']:.2f} | mu_runtime={agg['mu_runtime']:.3f}s")


if __name__ == "__main__":
    main ()