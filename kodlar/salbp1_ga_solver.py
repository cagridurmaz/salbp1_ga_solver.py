
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# See docstring inside for description.

import os, re, math, csv, time, random, statistics
from typing import List, Tuple, Dict, Optional
from pathlib import Path

try:
    import pandas as pd
except Exception:
    pd = None

def parse_alb(path: Path) -> Dict:
    text = path.read_text(encoding='utf-8', errors='ignore')
    lines = [ln.strip() for ln in text.splitlines()]

    def find(tag: str) -> int:
        for i, ln in enumerate(lines):
            if ln.lower() == tag.lower():
                return i
        return -1

    idx_n = find("<number of tasks>")
    idx_c = find("<cycle time>")
    idx_t = find("<task times>")
    idx_p = find("<precedence relations>")
    idx_end = find("<end>")
    if idx_n < 0 or idx_c < 0 or idx_t < 0 or idx_end < 0:
        raise ValueError(f"Unexpected .alb format: {path.name}")

    n = int(re.sub(r"[^\d]", "", lines[idx_n+1]))
    C_raw = lines[idx_c+1]
    try:
        C = int(C_raw)
    except Exception:
        C = int(float(C_raw.replace(",", ".")))

    times: Dict[int,int] = {}
    i = idx_t + 1
    while i < len(lines) and lines[i] != "" and not lines[i].startswith("<"):
        p = lines[i].split()
        if len(p)>=2 and p[0].isdigit():
            times[int(p[0])] = int(p[1])
        i += 1

    edges: List[Tuple[int,int]] = []
    if idx_p != -1:
        j = idx_p + 1
        while j < len(lines) and lines[j] != "" and not lines[j].startswith("<"):
            if re.match(r"^\d+\s*,\s*\d+$", lines[j]):
                a,b = [int(x) for x in lines[j].split(",")]
                edges.append((a,b))
            j += 1
    else:
        for ln in lines[i:]:
            if re.match(r"^\d+\s*,\s*\d+$", ln):
                a,b = [int(x) for x in ln.split(",")]
                edges.append((a,b))

    if len(times) != n:
        raise ValueError(f"{path.name}: parsed {len(times)} task times but expected {n}.")

    return {"name": path.stem, "n": n, "C": C, "times": times, "edges": edges, "source": str(path)}

def parse_excel_simple(path: Path) -> Optional[Dict]:
    if pd is None:
        return None
    try:
        xls = pd.ExcelFile(path)
    except Exception:
        return None
    if "tasks" not in xls.sheet_names or "precedence" not in xls.sheet_names:
        return None
    tasks_df = pd.read_excel(path, sheet_name="tasks")
    pred_df = pd.read_excel(path, sheet_name="precedence")
    C = None
    if "params" in xls.sheet_names:
        params_df = pd.read_excel(path, sheet_name="params", header=None)
        try:
            C_cand = params_df.iloc[0,0]
            if pd.notna(C_cand):
                C = int(float(C_cand))
        except Exception:
            C = None
    times = {int(r["task"]): int(r["time"]) for _, r in tasks_df.iterrows()}
    edges = [(int(r["i"]), int(r["j"])) for _, r in pred_df.iterrows()]
    n = len(times)
    return {"name": path.stem, "n": n, "C": C if C is not None else sum(times.values()),
            "times": times, "edges": edges, "source": str(path)}

def topo_levels(n: int, edges: List[Tuple[int,int]]) -> List[List[int]]:
    indeg = [0]*(n+1)
    adj = [[] for _ in range(n+1)]
    for i,j in edges:
        indeg[j]+=1; adj[i].append(j)
    level = []
    frontier = [u for u in range(1, n+1) if indeg[u]==0]
    used = [False]*(n+1)
    while frontier:
        level.append(sorted(frontier))
        nxt = []
        for u in frontier:
            used[u]=True
            for v in adj[u]:
                indeg[v]-=1
                if indeg[v]==0:
                    nxt.append(v)
        frontier = nxt
    for u in range(1, n+1):
        if not used[u]:
            level.append([u])
    return level

def random_topo_sort(n: int, edges: List[Tuple[int,int]], rng: random.Random) -> List[int]:
    indeg = [0]*(n+1)
    adj = [[] for _ in range(n+1)]
    for i,j in edges:
        indeg[j]+=1; adj[i].append(j)
    S = [u for u in range(1, n+1) if indeg[u]==0]
    seq = []
    while S:
        rng.shuffle(S)
        u = S.pop()
        seq.append(u)
        for v in adj[u]:
            indeg[v]-=1
            if indeg[v]==0:
                S.append(v)
    if len(seq)!=n:
        raise ValueError("Graph is not a DAG.")
    return seq

def is_topological(seq, edges):
    pos = {node:i for i,node in enumerate(seq)}
    for i,j in edges:
        if pos[i] > pos[j]:
            return False
    return True

def repair_topological(seq: List[int], edges: List[Tuple[int,int]]) -> List[int]:
    pos = {node:i for i,node in enumerate(seq)}
    changed = True
    while changed:
        changed = False
        for (i,j) in edges:
            if pos[i] > pos[j]:
                idx_i, idx_j = pos[i], pos[j]
                node_i = seq[idx_i]
                del seq[idx_i]
                seq.insert(idx_j, node_i)
                pos = {node:i for i,node in enumerate(seq)}
                changed = True
    return seq

def decode_sequence_to_stations(seq: List[int], times: Dict[int,int], C: int):
    stations = []
    current = []
    load = 0
    for i in seq:
        ti = times[i]
        if load + ti <= C:
            current.append(i); load += ti
        else:
            stations.append(current)
            current = [i]; load = ti
    if current: stations.append(current)
    return stations

def station_loads(stations, times):
    return [sum(times[i] for i in st) for st in stations]

def smoothness_index(stations, times):
    loads = station_loads(stations, times)
    if not loads: return 0.0
    m = len(loads)
    tt_max = max(loads)
    s = sum((tt_max - v)*(tt_max - v) for v in loads)
    return math.sqrt(s / m)

def line_efficiency(times_sum, m_used, C):
    denom = m_used * C
    return (times_sum / denom) if denom>0 else 0.0

def fitness_tuple(stations, times, C):
    m = len(stations)
    si = smoothness_index(stations, times)
    eff = line_efficiency(sum(times.values()), m, C)
    return (-m, -si, eff)

def tournament_select(pop, k, rng: random.Random):
    cand = rng.sample(pop, k)
    return max(cand, key=lambda x: x[2])

def pox_crossover(seqA, seqB, levels, edges, rng: random.Random):
    chosen = set()
    for lvl in levels:
        if rng.random() < 0.5:
            chosen.update(lvl)
    A_first = [i for i in seqA if i in chosen]
    A_second = [i for i in seqB if i not in chosen]
    child1 = A_first + A_second

    B_first = [i for i in seqB if i in chosen]
    B_second = [i for i in seqA if i not in chosen]
    child2 = B_first + B_second

    if not is_topological(child1, edges):
        child1 = repair_topological(child1, edges)
    if not is_topological(child2, edges):
        child2 = repair_topological(child2, edges)
    return child1, child2

def precedence_aware_swap(seq, edges, rng: random.Random):
    if len(seq) < 2: return seq[:]
    i, j = rng.sample(range(len(seq)), 2)
    if i > j: i, j = j, i
    child = seq[:]
    child[i], child[j] = child[j], child[i]
    if not is_topological(child, edges):
        child = repair_topological(child, edges)
    return child

def run_ga_once(data: Dict,
                pop_size=100, generations=200,
                pc=0.8, pm=0.05, tour_k=3, elite=2,
                seed=1234, verbose=False,
                max_time_sec=0.5):   # <<< YENİ PARAMETRE
    rng = random.Random(seed)
    n = data["n"]; C = int(data["C"]); times = data["times"]; edges = data["edges"]
    times_sum = sum(times.values())
    levels = topo_levels(n, edges)

    population = []
    for _ in range(pop_size):
        seq = random_topo_sort(n, edges, rng)
        stations = decode_sequence_to_stations(seq, times, C)
        fit = fitness_tuple(stations, times, C)
        population.append((seq, stations, fit))

    best = max (population, key=lambda x: x[2])
    start = time.time ()

    for g in range (generations):

        # --- TIME LIMIT KONTROLÜ ---
        if max_time_sec is not None and (time.time () - start) >= max_time_sec:
            if verbose:
                print (f"Time limit reached for this GA run at generation {g}")
            break
        newpop = []
        elites = sorted(population, key=lambda x: x[2], reverse=True)[:elite]
        newpop.extend(elites)

        while len(newpop) < pop_size:
            p1 = tournament_select(population, tour_k, rng)
            p2 = tournament_select(population, tour_k, rng)

            if rng.random() < pc:
                c1_seq, c2_seq = pox_crossover(p1[0], p2[0], topo_levels(n, edges), edges, rng)
            else:
                c1_seq, c2_seq = p1[0][:], p2[0][:]

            if rng.random() < pm:
                c1_seq = precedence_aware_swap(c1_seq, edges, rng)
            if len(newpop)+1 < pop_size and rng.random() < pm:
                c2_seq = precedence_aware_swap(c2_seq, edges, rng)

            c1_st = decode_sequence_to_stations(c1_seq, times, C)
            c1_fit = fitness_tuple(c1_st, times, C)
            newpop.append((c1_seq, c1_st, c1_fit))

            if len(newpop) < pop_size:
                c2_st = decode_sequence_to_stations(c2_seq, times, C)
                c2_fit = fitness_tuple(c2_st, times, C)
                newpop.append((c2_seq, c2_st, c2_fit))

        population = newpop
        cur_best = max(population, key=lambda x: x[2])
        if cur_best[2] > best[2]:
            best = cur_best

    end = time.time()
    best_seq, best_st, _ = best
    m_used = len(best_st)
    si = smoothness_index(best_st, times)
    eff = line_efficiency(times_sum, m_used, C)
    runtime = end - start

    return {
        "best_seq": best_seq,
        "best_stations": best_st,
        "m": m_used,
        "SI": si,
        "Eff": eff,
        "runtime_sec": runtime,
        "params": {
            "pop_size": pop_size, "generations": generations,
            "pc": pc, "pm": pm, "tour_k": tour_k, "elite": elite, "seed": seed
        }
    }

def solve_folder_ga(input_dir: str, runs: int = 10, out_dir: Optional[str] = None,
                    pop_size=100, generations=200, pc=0.8, pm=0.05, tour_k=3, elite=2,
                    base_seed=1000, verbose=False):
    p = Path(input_dir); assert p.exists() and p.is_dir(), f"Input dir not found: {input_dir}"
    outp = Path(out_dir) if out_dir else p / "solutions_ga"; outp.mkdir(parents=True, exist_ok=True)

    files = list(p.glob("*.alb")) + list(p.glob("*.xlsx"))
    if not files:
        print(f"No .alb or .xlsx files found in {input_dir}")
        return {}

    summary_path = outp / "GA_metrics_summary.csv"
    with open(summary_path, "w", newline="") as sf:
        sw = csv.writer(sf)
        sw.writerow(["instance","Smallest_m","SI_mean","Eff_mean","mu_m","sigma_m","mu_runtime","sigma_runtime","best_run"])

        out_json = {}
        for f in files:
            if f.suffix.lower()==".alb":
                data = parse_alb(f)
            else:
                data = parse_excel_simple(f)
                if data is None: continue

            per_results = []
            for r in range(1, runs+1):
                seed = base_seed + r
                res = run_ga_once(data, pop_size=pop_size, generations=generations, pc=pc, pm=pm,
                                  tour_k=tour_k, elite=elite, seed=seed, verbose=verbose)
                per_results.append(res)

            ms = [r["m"] for r in per_results]
            sis = [r["SI"] for r in per_results]
            effs = [r["Eff"] for r in per_results]
            rtimes = [r["runtime_sec"] for r in per_results]
            smallest_m = min(ms) if ms else None
            SI_mean = statistics.mean(sis) if sis else None
            Eff_mean = statistics.mean(effs) if effs else None
            mu_m = statistics.mean(ms) if ms else None
            sigma_m = statistics.pstdev(ms) if len(ms)>1 else 0.0 if ms else None
            mu_rt = statistics.mean(rtimes) if rtimes else None
            sigma_rt = statistics.pstdev(rtimes) if len(rtimes)>1 else 0.0 if rtimes else None

            best_idx = min(range(len(per_results)), key=lambda i: (per_results[i]["m"], per_results[i]["SI"]))
            best = per_results[best_idx]

            per_path = outp / f"{data['name']}_GA_runs.csv"
            with open(per_path, "w", newline="") as pf:
                pw = csv.writer(pf)
                pw.writerow(["run","m","SI","Eff","runtime_sec","best_seq_prefix","pop_size","generations","pc","pm","tour_k","elite","seed"])
                for i, r in enumerate(per_results, start=1):
                    pw.writerow([i, r["m"], r["SI"], r["Eff"], r["runtime_sec"],
                                 " ".join(map(str, r["best_seq"][:10])),
                                 r["params"]["pop_size"], r["params"]["generations"],
                                 r["params"]["pc"], r["params"]["pm"], r["params"]["tour_k"],
                                 r["params"]["elite"], r["params"]["seed"]])

            best_assign = {}
            for s_idx, st in enumerate(best["best_stations"], start=1):
                for i in st:
                    best_assign[i] = s_idx
            with open(outp / f"{data['name']}_GA_assignment.csv", "w", newline="") as cf:
                w = csv.writer(cf); w.writerow(["task","station"])
                for i in sorted(best_assign.keys()):
                    w.writerow([i, best_assign[i]])

            loads = [sum(data["times"][i] for i in st) for st in best["best_stations"]]
            with open(outp / f"{data['name']}_GA_station_loads.csv", "w", newline="") as lf:
                w = csv.writer(lf); w.writerow(["station","load","cycle_time"])
                for s_idx, ld in enumerate(loads, start=1):
                    w.writerow([s_idx, ld, data["C"]])

            sw.writerow([data["name"], smallest_m, SI_mean, Eff_mean, mu_m, sigma_m, mu_rt, sigma_rt, best_idx+1])

            out_json[data["name"]] = {"aggregate": {
                "Smallest_m": smallest_m, "SI_mean": SI_mean, "Eff_mean": Eff_mean,
                "mu_m": mu_m, "sigma_m": sigma_m, "mu_runtime": mu_rt, "sigma_runtime": sigma_rt,
                "best_run": best_idx+1
            }}
    return out_json

def main():
    import argparse
    parser = argparse.ArgumentParser(description="GA solver for SALBP-1 with precedence-feasible chromosomes.")
    parser.add_argument("--input_dir", type=str, default=".", help="Folder containing .alb/.xlsx")
    parser.add_argument("--out_dir", type=str, default=None, help="Output folder (default: input_dir/solutions_ga)")
    parser.add_argument("--runs", type=int, default=10, help="Number of independent GA runs per instance")
    parser.add_argument("--pop_size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--pc", type=float, default=0.8)
    parser.add_argument("--pm", type=float, default=0.05)
    parser.add_argument("--tour_k", type=int, default=3)
    parser.add_argument("--elite", type=int, default=2)
    parser.add_argument("--base_seed", type=int, default=1000)
    args = parser.parse_args()

    res = solve_folder_ga(args.input_dir, runs=args.runs, out_dir=args.out_dir,
                          pop_size=args.pop_size, generations=args.generations,
                          pc=args.pc, pm=args.pm, tour_k=args.tour_k, elite=args.elite,
                          base_seed=args.base_seed)
    for name, obj in res.items():
        agg = obj["aggregate"]
        print(f"[{name}] Smallest_m={agg['Smallest_m']} SI_mean={agg['SI_mean']:.6f} Eff_mean={agg['Eff_mean']:.6f} mu_m={agg['mu_m']:.3f} sigma_m={agg['sigma_m']:.3f} mu_runtime={agg['mu_runtime']:.3f}s (best_run={agg['best_run']})")

if __name__ == "__main__":
    main()
