PROJECT: GA solver for SALBP-1 (precedence-feasible chromosomes)
SCRIPT: ga_salbp1.py  (rename if different)
AUTHOR: <Your Name>
DATE: <YYYY-MM-DD>

============================================================
1) Overview
============================================================
This project solves SALBP-1 (Simple Assembly Line Balancing Problem Type-1)
instances using a Genetic Algorithm (GA) with precedence-feasible chromosomes.
It reads problem instances from a folder containing:
- .alb files (standard SALBP benchmark-like text format)
- .xlsx files (custom Excel format with sheets: tasks, precedence, optional params)

For each instance, the script runs the GA multiple times (independent runs),
writes per-run metrics and best assignment outputs to CSV files, and generates
an aggregate summary file.

============================================================
2) Requirements
============================================================
- Python 3.9+ recommended (tested with Python 3.x)
- Optional: pandas (only needed if you want to read .xlsx files)

Install dependencies:
- If you will only use .alb files: pandas is NOT required.
- If you will also use .xlsx files: pandas IS required.

============================================================
3) Repository structure (recommended)
============================================================
your-repo/
  src/
    ga_salbp1.py
  data/
    sample/
      example.alb
      example.xlsx   (optional)
  outputs/           (created automatically if you choose)
  requirements.txt
  run_instructions.txt
  README.md

============================================================
4) Setup (Windows / macOS / Linux)
============================================================
A) Create a virtual environment (recommended)
1. In terminal, go to repo root folder
2. Create venv:
   python -m venv .venv

3. Activate:
   Windows: .venv\Scripts\activate
   macOS/Linux: source .venv/bin/activate

B) Install requirements
If you will use Excel inputs:
   pip install pandas openpyxl
If you will only use .alb inputs:
   (no extra packages required)

Tip: You can also keep a requirements.txt such as:
   pandas
   openpyxl

============================================================
5) Input data formats
============================================================
A) .alb format
The parser expects the following tags in the file:
  <number of tasks>
  <cycle time>
  <task times>
  <precedence relations>   (optional; can appear elsewhere too)
  <end>

Task times section example (space separated):
  task_id  time

Precedence lines example (comma separated):
  i, j

B) .xlsx format
Excel files must contain these sheets:
- "tasks" sheet with columns:
    task   time
- "precedence" sheet with columns:
    i      j
Optional:
- "params" sheet: cycle time C read from cell (0,0) if provided.

Notes:
- If cycle time C is missing in Excel, the script uses sum(times) as a fallback.

============================================================
6) How to run
============================================================
The script processes ALL .alb and .xlsx files in the input folder.

Basic command (run from repo root):
   python src/ga_salbp1.py --input_dir data/sample

If you want to specify output folder:
   python src/ga_salbp1.py --input_dir data/sample --out_dir outputs

Control number of GA runs per instance:
   python src/ga_salbp1.py --input_dir data/sample --runs 10

Control GA parameters:
   python src/ga_salbp1.py --input_dir data/sample --runs 10 --pop_size 100 --generations 200 --pc 0.8 --pm 0.05 --tour_k 3 --elite 2 --base_seed 1000

Argument explanation:
--input_dir     Folder containing .alb/.xlsx files
--out_dir       Output folder (default: <input_dir>/solutions_ga)
--runs          Independent GA runs per instance
--pop_size      Population size
--generations   Maximum generations
--pc            Crossover probability
--pm            Mutation probability
--tour_k        Tournament size (selection pressure)
--elite         Number of elite individuals preserved each generation
--base_seed     Seed base; each run uses (base_seed + run_index)

============================================================
7) Outputs (what files are created)
============================================================
In the output directory (default: input_dir/solutions_ga), the script creates:

1) GA_metrics_summary.csv
   Columns:
   instance, Smallest_m, SI_mean, Eff_mean, mu_m, sigma_m, mu_runtime, sigma_runtime, best_run

2) For each instance named <name>:
   - <name>_GA_runs.csv
       Per-run metrics including m, smoothness index (SI), efficiency (Eff), runtime, and parameters
   - <name>_GA_assignment.csv
       Best run’s task-to-station assignment (task, station)
   - <name>_GA_station_loads.csv
       Best run’s station loads and cycle time (station, load, cycle_time)

Also prints a one-line summary to console for each instance:
   [instance] Smallest_m=... SI_mean=... Eff_mean=... mu_m=... sigma_m=... mu_runtime=... (best_run=...)

============================================================
8) Notes about time limits
============================================================
The function run_ga_once(...) contains a parameter:
   max_time_sec=0.5
to stop evolution after a time threshold.

IMPORTANT: In the current script version, solve_folder_ga(...) does not pass
max_time_sec to run_ga_once(...), so runs use the default behavior defined
inside run_ga_once unless you modify solve_folder_ga to forward max_time_sec
and add a CLI argument. If you need strict run-level time limits from CLI,
update the code accordingly (small change).

============================================================
9) Troubleshooting
============================================================
- "No .alb or .xlsx files found":
  Check --input_dir path and ensure files exist.

- Excel file is ignored:
  The Excel must have sheets named exactly "tasks" and "precedence".
  Install pandas + openpyxl.

- "Graph is not a DAG.":
  The precedence graph contains cycles; the GA requires a DAG.

- Encoding issues reading .alb:
  Files are read with utf-8 and errors ignored; still, ensure tags exist exactly.

============================================================
10) Reproducibility
============================================================
- Each run uses seed = base_seed + r (r = 1..runs).
- For reproducible results, keep the same input files and parameters.

============================================================
11) GitHub link
============================================================
GitHub repository:
<https://github.com/cagridurmaz/salbp1_ga_solver.py>

In the report, mention that:
- all source code is in this repository,
- run instructions are provided in run_instructions.txt,
- sample input data is provided under data/sample/ (or linked if confidential).
