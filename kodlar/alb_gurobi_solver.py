
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assembly Line Balancing (SALBP) Auto Solver for .alb Benchmarks (and simple Excel)
- Supports SALBP-1 (min #stations with given cycle time)
- Optional SALBP-2 (min cycle time with given #stations)
- Parses .alb files in the folder. Also supports a very simple Excel format (see notes below).
- Saves assignment and station loads to CSV for each instance.
- NEW: Exports practical performance metrics per instance (efficiency, idle, balance loss, etc.).
Author: ChatGPT
"""
import os
import re
import csv
import math
import json
import statistics
from typing import List, Tuple, Dict, Optional
from pathlib import Path

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:
    gp = None
    GRB = None

try:
    import pandas as pd
except Exception:
    pd = None

# ---------------------------
# Parsing utilities
# ---------------------------

def parse_alb(path: Path) -> Dict:
    """
    Parse a .alb file with sections like:
    <number of tasks>
    20
    
    <cycle time>
    1000
    
    <order strength>
    0,3   # optional, ignored
    
    <task times>
    1 190
    2 159
    ...
    
    <precedence relations>
    1,12
    2,12
    ...
    
    <end>
    """
    text = path.read_text(encoding='utf-8', errors='ignore')
    lines = [ln.strip() for ln in text.splitlines()]
    # helper to find a section
    def find_index(tag: str) -> int:
        for i, ln in enumerate(lines):
            if ln.lower() == tag.lower():
                return i
        return -1

    idx_n = find_index("<number of tasks>")
    idx_c = find_index("<cycle time>")
    idx_t = find_index("<task times>")
    idx_p = find_index("<precedence relations>")
    idx_end = find_index("<end>")

    if idx_n == -1 or idx_c == -1 or idx_t == -1 or idx_end == -1:
        raise ValueError(f"Unexpected .alb format in {path.name}")

    try:
        n_tasks = int(lines[idx_n+1])
    except Exception:
        # try to strip commas etc.
        n_tasks = int(re.sub(r"[^\d]", "", lines[idx_n+1]))

    # cycle time might be integer
    cycle_raw = lines[idx_c+1]
    try:
        C = int(cycle_raw)
    except Exception:
        C = int(float(cycle_raw.replace(",", ".")))

    # task times
    times: Dict[int, int] = {}
    i = idx_t + 1
    while i < len(lines) and lines[i] != "" and not lines[i].startswith("<"):
        parts = lines[i].split()
        if len(parts) >= 2 and parts[0].isdigit():
            tid = int(parts[0])
            val = int(parts[1])
            times[tid] = val
        i += 1

    # precedence relations
    preds: List[Tuple[int, int]] = []
    if idx_p != -1:
        j = idx_p + 1
        while j < len(lines) and lines[j] != "" and not lines[j].startswith("<"):
            if re.match(r"^\d+\s*,\s*\d+$", lines[j]):
                a, b = [int(x) for x in lines[j].split(",")]
                preds.append((a, b))
            j += 1
    else:
        # try to find any lines of "i,j" after task times
        for ln in lines[i:]:
            if re.match(r"^\d+\s*,\s*\d+$", ln):
                a, b = [int(x) for x in ln.split(",")]
                preds.append((a, b))

    # basic checks
    if len(times) != n_tasks:
        raise ValueError(f"{path.name}: parsed {len(times)} task times but expected {n_tasks}.")

    return {
        "name": path.stem,
        "n": n_tasks,
        "C": C,
        "times": times,
        "edges": preds,
        "source": str(path)
    }


def parse_excel_simple(path: Path) -> Optional[Dict]:
    """
    Very simple Excel support (optional):
    - If there's a sheet named 'tasks' with columns: task, time
    - And a sheet named 'precedence' with columns: i, j  (edge i -> j)
    - And an optional cell 'C' in a sheet 'params' (single value)
    Returns None if pandas not available or required sheets are missing.
    """
    if pd is None:
        return None
    try:
        xls = pd.ExcelFile(path)
    except Exception:
        return None

    req1 = "tasks" in xls.sheet_names
    req2 = "precedence" in xls.sheet_names
    if not (req1 and req2):
        return None

    tasks_df = pd.read_excel(path, sheet_name="tasks")
    pred_df = pd.read_excel(path, sheet_name="precedence")
    C = None
    if "params" in xls.sheet_names:
        params_df = pd.read_excel(path, sheet_name="params", header=None)
        # try to find a numeric in the first cell
        try:
            C_candidate = params_df.iloc[0,0]
            if pd.notna(C_candidate):
                C = int(float(C_candidate))
        except Exception:
            C = None

    times = {int(row["task"]): int(row["time"]) for _, row in tasks_df.iterrows()}
    edges = [(int(row["i"]), int(row["j"])) for _, row in pred_df.iterrows()]
    n = len(times)
    return {
        "name": path.stem,
        "n": n,
        "C": C if C is not None else sum(times.values()),  # fallback
        "times": times,
        "edges": edges,
        "source": str(path)
    }

# ---------------------------
# SALBP models
# ---------------------------

def build_and_solve_salbp1(data: Dict, time_limit: Optional[int] = None, verbose: bool = True):
    """
    SALBP-1: minimize number of stations for given cycle time C.
    data keys: n, C, times(dict 1..n), edges(list of (i,j)), name
    """
    if gp is None:
        raise RuntimeError("gurobipy is not available in this environment. Install Gurobi to solve.")

    n = data["n"]
    C = int(data["C"])
    times = data["times"]
    edges = data["edges"]
    name = data.get("name", "instance")

    Smax = n  # trivial upper bound
    Sset = range(1, Smax+1)
    Iset = range(1, n+1)

    m = gp.Model(f"SALBP1_{name}")
    if time_limit:
        m.Params.TimeLimit = time_limit
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.NonConvex = 2

    x = m.addVars(Iset, Sset, vtype=GRB.BINARY, name="x")
    y = m.addVars(Sset, vtype=GRB.BINARY, name="y")

    # Each task assigned exactly once
    m.addConstrs((gp.quicksum(x[i, s] for s in Sset) == 1 for i in Iset), name="assign")

    # Station capacity (link y via capacity bound)
    m.addConstrs((gp.quicksum(times[i] * x[i, s] for i in Iset) <= C * y[s] for s in Sset), name="capacity")

    # Precedence: j cannot be before i
    for (i, j) in edges:
        # cumulative form: for every k, sum_{s<=k} x[j,s] - sum_{s<=k} x[i,s] >= 0
        for k in Sset:
            m.addConstr(gp.quicksum(x[j, s] for s in Sset if s <= k) -
                        gp.quicksum(x[i, s] for s in Sset if s <= k) >= 0,
                        name=f"prec_{i}_{j}_upto_{k}")

    # Objective: minimize number of used stations
    m.setObjective(gp.quicksum(y[s] for s in Sset), GRB.MINIMIZE)

    m.optimize()

    sol = {
        "status": m.Status,
        "obj": m.objVal if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) else None,
        "assign": {},
        "station_loads": {},
        "C": C,
        "name": name,
        "is_optimal": 1 if m.Status == GRB.OPTIMAL else 0
    }

    if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        # extract solution
        for s in Sset:
            if y[s].X > 0.5:
                load = sum(times[i] for i in Iset if x[i, s].X > 0.5)
                sol["station_loads"][s] = load
        for i in Iset:
            for s in Sset:
                if x[i, s].X > 0.5:
                    sol["assign"][i] = s
                    break
    return sol


def build_and_solve_salbp2(data: Dict, stations: int, time_limit: Optional[int] = None, verbose: bool = True):
    """
    SALBP-2: minimize cycle time with given number of stations.
    """
    if gp is None:
        raise RuntimeError("gurobipy is not available in this environment. Install Gurobi to solve.")

    n = data["n"]
    times = data["times"]
    edges = data["edges"]
    name = data.get("name", "instance")

    S = stations
    Sset = range(1, S+1)
    Iset = range(1, n+1)

    m = gp.Model(f"SALBP2_{name}")
    if time_limit:
        m.Params.TimeLimit = time_limit
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.NonConvex = 2

    x = m.addVars(Iset, Sset, vtype=GRB.BINARY, name="x")
    C = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="C")

    # Assignment
    m.addConstrs((gp.quicksum(x[i, s] for s in Sset) == 1 for i in Iset), name="assign")

    # Capacity per station
    m.addConstrs((gp.quicksum(times[i] * x[i, s] for i in Iset) <= C for s in Sset), name="capacity")

    # Precedence cumulative
    for (i, j) in edges:
        for k in Sset:
            m.addConstr(gp.quicksum(x[j, s] for s in Sset if s <= k) -
                        gp.quicksum(x[i, s] for s in Sset if s <= k) >= 0,
                        name=f"prec_{i}_{j}_upto_{k}")

    m.setObjective(C, GRB.MINIMIZE)
    m.optimize()

    sol = {
        "status": m.Status,
        "obj": m.objVal if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) else None,
        "assign": {},
        "station_loads": {},
        "C": None,
        "name": name,
        "is_optimal": 1 if m.Status == GRB.OPTIMAL else 0
    }
    if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        Cval = C.X
        sol["C"] = Cval
        for s in Sset:
            load = sum(times[i] for i in Iset if x[i, s].X > 0.5)
            sol["station_loads"][s] = load
        for i in Iset:
            for s in Sset:
                if x[i, s].X > 0.5:
                    sol["assign"][i] = s
                    break
    return sol

# ---------------------------
# Metrics
# ---------------------------

def compute_metrics(data: Dict, sol: Dict) -> Dict:
    """
    Practical performance & workload metrics derivable from the instance and solution.
    Returns a dict of metrics.
    """
    times = data["times"]
    C = sol.get("C", data["C"])
    S_used = len(sol["station_loads"])
    Tsum = sum(times.values())
    loads = list(sol["station_loads"].values()) if sol["station_loads"] else []
    max_load = max(loads) if loads else 0
    min_load = min(loads) if loads else 0
    stdev_load = statistics.pstdev(loads) if loads else 0.0
    total_capacity = S_used * C
    total_idle = total_capacity - Tsum if total_capacity >= Tsum else 0
    efficiency = (Tsum / total_capacity) if total_capacity > 0 else 0.0
    balance_loss = 1.0 - efficiency if efficiency > 0 else 0.0

    # Lower bound (continuous): ceil(Tsum/C) for SALBP-1
    LB_stations = math.ceil(Tsum / C) if C else None

    Tmin = min(times.values())
    Tmax = max(times.values())
    stdev_t = statistics.pstdev(list(times.values()))

    return {
        "instance": data.get("name"),
        "n_tasks": data["n"],
        "cycle_time": C,
        "stations_used": S_used,
        "is_optimal_flag": sol.get("is_optimal", 0),  # 1 if Gurobi status OPTIMAL
        "LB_stations_continuous": LB_stations,
        "efficiency": round(efficiency, 6),
        "balance_loss": round(balance_loss, 6),
        "total_idle": total_idle,
        "avg_idle_per_station": (total_idle / S_used) if S_used else None,
        "max_station_load": max_load,
        "min_station_load": min_load,
        "stdev_station_load": round(stdev_load, 6),
        "Tsum": Tsum,
        "Tsum_over_C": round(Tsum / C, 6) if C else None,
        "Tmin_over_C": round(Tmin / C, 6) if C else None,
        "Tmax_over_C": round(Tmax / C, 6) if C else None,
        "stdev_t_over_C": round(stdev_t / C, 6) if C else None,
    }


# ---------------------------
# Structural (instance) metrics
# ---------------------------

from collections import defaultdict, deque

def _topo_layers(n, edges):
    """Return list of layers (each layer is a list of nodes) using Kahn's algorithm (level by level)."""
    indeg = [0]*(n+1)
    adj = [[] for _ in range(n+1)]
    for i,j in edges:
        adj[i].append(j)
        indeg[j]+=1
    q = deque([u for u in range(1,n+1) if indeg[u]==0])
    layers = []
    while q:
        layer = list(q)
        layers.append(layer)
        for _ in range(len(q)):
            u = q.popleft()
            for v in adj[u]:
                indeg[v]-=1
                if indeg[v]==0:
                    q.append(v)
    return layers

def _reachability_count(n, edges):
    """Count number of ordered comparable pairs (i<->j with path either way). DAG expected."""
    adj = [[] for _ in range(n+1)]
    for i,j in edges:
        adj[i].append(j)
    # DFS from each node (could be O(n*(n+e))). For n<=few hundreds fine.
    reachable_pairs = 0
    for s in range(1,n+1):
        seen = [False]*(n+1)
        stack = [s]
        seen[s]=True
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v]=True
                    stack.append(v)
        # don't count self
        reachable_pairs += sum(1 for v in range(1,n+1) if v!=s and seen[v])
    # We counted ordered pairs (i->j). For OS we need unordered comparable pairs; but in a DAG, if i reaches j, j cannot reach i.
    # So unordered comparable pairs = ordered comparable pairs.
    return reachable_pairs  # equals number of comparable unordered pairs

def _chains(n, edges, indeg, outdeg):
    """Decompose into maximal chains where internal nodes have indeg=1 and outdeg=1; return list of chain lengths >=2."""
    adj = defaultdict(list)
    for i,j in edges:
        adj[i].append(j)
    starts = [u for u in range(1,n+1) if outdeg[u]>0 and indeg[u]!=1]  # candidate starts (sources and forks)
    seen = set()
    chains = []
    for s in starts:
        for v in adj[s]:
            if (indeg[v]==1):
                # grow chain
                length = 2  # s -> v
                cur = v
                while outdeg.get(cur,0)==1:
                    nxt = adj[cur][0]
                    if indeg.get(nxt,0)!=1:
                        break
                    length += 1
                    cur = nxt
                if length>=2:
                    chains.append(length)
    return chains

def compute_structural_metrics(data: Dict) -> Dict:
    n = data["n"]
    edges = data["edges"]
    times = data["times"]

    # indegree / outdegree
    indeg = {i:0 for i in range(1,n+1)}
    outdeg = {i:0 for i in range(1,n+1)}
    for i,j in edges:
        outdeg[i]+=1
        indeg[j]+=1

    total_deg = {i: indeg[i]+outdeg[i] for i in range(1,n+1)}
    max_task_degree = max(total_deg.values()) if total_deg else 0

    # Share of tasks without predecessors
    share_no_preds = sum(1 for i in range(1,n+1) if indeg[i]==0)/n if n else 0.0

    # AIP: Average Immediate Predecessors (avg indegree)
    AIP = sum(indeg.values())/n if n else 0.0

    # Divergence / Convergence (raw avg degrees and normalized by n-1)
    avg_out = sum(outdeg.values())/n if n else 0.0
    avg_in  = sum(indeg.values())/n if n else 0.0
    norm_out = avg_out/(n-1) if n>1 else 0.0
    norm_in  = avg_in /(n-1) if n>1 else 0.0

    # Chain tasks: exactly one pred and one succ
    chain_mask = [1 for i in range(1,n+1) if indeg[i]==1 and outdeg[i]==1]
    share_chain_tasks = sum(chain_mask)/n if n else 0.0

    # Chain lengths
    ch_lengths = _chains(n, edges, indeg, outdeg)
    avg_chain_length = (sum(ch_lengths)/len(ch_lengths)) if ch_lengths else 0.0

    # Bottlenecks: define as nodes achieving max total degree
    if total_deg:
        max_deg = max(total_deg.values())
        bottlenecks = [i for i,d in total_deg.items() if d==max_deg]
        share_bottleneck = len(bottlenecks)/n
        avg_deg_bottlenecks = (sum(total_deg[i] for i in bottlenecks)/len(bottlenecks)) if bottlenecks else 0.0
    else:
        share_bottleneck = 0.0
        avg_deg_bottlenecks = 0.0
        max_deg = 0

    # Stages (levels) and avg tasks per stage
    layers = _topo_layers(n, edges)
    num_stages = len(layers) if layers else 0
    avg_tasks_per_stage = (n/num_stages) if num_stages>0 else 0.0

    # Order Strength (OS): comparable pairs / total pairs
    reachable_pairs = _reachability_count(n, edges)
    total_pairs = n*(n-1)//2
    order_strength = (reachable_pairs/total_pairs) if total_pairs>0 else 0.0

    return {
        "order_strength": round(order_strength,6),
        "AIP": round(AIP,6),
        "max_task_degree": max_task_degree,
        "degree_of_divergence_raw": round(avg_out,6),
        "degree_of_divergence_norm": round(norm_out,6),
        "degree_of_convergence_raw": round(avg_in,6),
        "degree_of_convergence_norm": round(norm_in,6),
        "share_of_chain_tasks": round(share_chain_tasks,6),
        "avg_chain_length": round(avg_chain_length,6),
        "share_of_bottleneck_tasks": round(share_bottleneck,6),
        "avg_degree_of_bottlenecks": round(avg_deg_bottlenecks,6),
        "avg_no_tasks_per_stage": round(avg_tasks_per_stage,6),
        "share_tasks_without_predecessors": round(share_no_preds,6),
    }

# ---------------------------
# Orchestrator
# ---------------------------

def solve_folder(input_dir: str,
                 mode: str = "SALBP1",
                 time_limit: Optional[int] = None,
                 stations_for_salbp2: Optional[int] = None,
                 out_dir: Optional[str] = None,
                 verbose: bool = True) -> Dict[str, Dict]:
    """
    Scan a folder for .alb and simple Excel files and solve.
    Returns a dict of instance_name -> solution dict (or error).
    """
    p = Path(input_dir)
    assert p.exists() and p.is_dir(), f"Input dir not found: {input_dir}"
    results = {}
    outp = Path(out_dir) if out_dir else p / "solutions"
    outp.mkdir(parents=True, exist_ok=True)

    files = list(p.glob("*.alb")) + list(p.glob("*.xlsx"))
    if not files:
        print(f"No .alb or .xlsx files found in {input_dir}")
        return results

    # open metrics CSV
    metrics_path = outp / "metrics.csv"
    with open(metrics_path, "w", newline="") as mf:
        w = csv.writer(mf)
        w.writerow([
            "instance","n_tasks","cycle_time","stations_used","is_optimal_flag",
            "LB_stations_continuous","efficiency","balance_loss","total_idle",
            "avg_idle_per_station","max_station_load","min_station_load","stdev_station_load",
            "Tsum","Tsum_over_C","Tmin_over_C","Tmax_over_C","stdev_t_over_C",
            "order_strength","AIP","max_task_degree","degree_of_divergence_raw","degree_of_divergence_norm",
            "degree_of_convergence_raw","degree_of_convergence_norm","share_of_chain_tasks","avg_chain_length",
            "share_of_bottleneck_tasks","avg_degree_of_bottlenecks","avg_no_tasks_per_stage","share_tasks_without_predecessors"
        ])

        for f in files:
            try:
                data = None
                if f.suffix.lower() == ".alb":
                    data = parse_alb(f)
                elif f.suffix.lower() == ".xlsx":
                    data = parse_excel_simple(f)  # only simple excel is supported
                    if data is None:
                        # skip non-simple excel
                        continue

                if data is None:
                    continue

                if mode.upper() == "SALBP1":
                    sol = build_and_solve_salbp1(data, time_limit=time_limit, verbose=verbose)
                elif mode.upper() == "SALBP2":
                    if stations_for_salbp2 is None:
                        raise ValueError("stations_for_salbp2 must be provided for SALBP2 mode.")
                    sol = build_and_solve_salbp2(data, stations=stations_for_salbp2, time_limit=time_limit, verbose=verbose)
                else:
                    raise ValueError("mode must be SALBP1 or SALBP2")

                results[data["name"]] = sol

                # Save CSVs
                assign_csv = outp / f"{data['name']}_assignment.csv"
                loads_csv = outp / f"{data['name']}_station_loads.csv"
                with open(assign_csv, "w", newline="") as cf:
                    w_assign = csv.writer(cf)
                    w_assign.writerow(["task", "station"])
                    for i in sorted(sol["assign"].keys()):
                        w_assign.writerow([i, sol["assign"][i]])
                with open(loads_csv, "w", newline="") as lf:
                    w_loads = csv.writer(lf)
                    w_loads.writerow(["station", "load", "cycle_time"])
                    for s in sorted(sol["station_loads"].keys()):
                        w_loads.writerow([s, sol["station_loads"][s], sol.get("C")])

                # Write metrics line
                met = compute_metrics(data, sol)
                smet = compute_structural_metrics(data)
                w.writerow([
                    met["instance"], met["n_tasks"], met["cycle_time"], met["stations_used"], met["is_optimal_flag"],
                    met["LB_stations_continuous"], met["efficiency"], met["balance_loss"], met["total_idle"],
                    met["avg_idle_per_station"], met["max_station_load"], met["min_station_load"], met["stdev_station_load"],
                    met["Tsum"], met["Tsum_over_C"], met["Tmin_over_C"], met["Tmax_over_C"], met["stdev_t_over_C"],
                    smet["order_strength"], smet["AIP"], smet["max_task_degree"], smet["degree_of_divergence_raw"], smet["degree_of_divergence_norm"],
                    smet["degree_of_convergence_raw"], smet["degree_of_convergence_norm"], smet["share_of_chain_tasks"], smet["avg_chain_length"],
                    smet["share_of_bottleneck_tasks"], smet["avg_degree_of_bottlenecks"], smet["avg_no_tasks_per_stage"], smet["share_tasks_without_predecessors"]
                ])

            except Exception as e:
                results[f.stem] = {"error": str(e), "file": str(f)}

    # Also save a summary JSON
    with open(outp / "summary.json", "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2, ensure_ascii=False)
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Solve SALBP instances in a folder with Gurobi.")
    parser.add_argument("--input_dir", type=str, default=".", help="Folder containing .alb (and simple Excel) files.")
    parser.add_argument("--mode", type=str, default="SALBP1", choices=["SALBP1", "SALBP2"], help="Problem variant.")
    parser.add_argument("--time_limit", type=int, default=None, help="Time limit in seconds.")
    parser.add_argument("--stations", type=int, default=None, help="#stations (required for SALBP2).")
    parser.add_argument("--out_dir", type=str, default=None, help="Output folder (default: input_dir/solutions).")
    parser.add_argument("--quiet", action="store_true", help="Suppress Gurobi output.")
    args = parser.parse_args()

    results = solve_folder(
        input_dir=args.input_dir,
        mode=args.mode,
        time_limit=args.time_limit,
        stations_for_salbp2=args.stations,
        out_dir=args.out_dir,
        verbose=not args.quiet
    )
    # Print a short console summary
    for name, sol in results.items():
        if "error" in sol:
            print(f"[{name}] ERROR: {sol['error']}")
        else:
            print(f"[{name}] status={sol['status']} obj={sol['obj']} C={sol.get('C')} assigned={len(sol['assign'])}")

if __name__ == "__main__":
    main()
