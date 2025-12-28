# pip install ortools
from ortools.sat.python import cp_model

def flexible_jobshop_example():
    model = cp_model.CpModel()

    # -----------------------------
    # 1) Veri: işler, operasyonlar, alternatif makineler ve süreleri
    # -----------------------------
    # 3 iş (J0, J1, J2); her işte 3 operasyon (O0->O1->O2)
    # Her operasyon bazı makinelerde yapılabilir ve süreleri farklıdır (dakika).
    # Ayrıca bazı operasyonlar 1 işçi, bazıları 2 işçi gerektirir.
    jobs = [
        # Job 0
        {
            "ops": [
                {"name":"J0O0", "machines": {"M1": 6, "M2": 9},  "workers": 1},
                {"name":"J0O1", "machines": {"M1": 7, "M3": 5},  "workers": 2},
                {"name":"J0O2", "machines": {"M2": 4, "M3": 6},  "workers": 1},
            ],
            "due": 18,  # termin zamanı
            "tardiness_weight": 3
        },
        # Job 1
        {
            "ops": [
                {"name":"J1O0", "machines": {"M1": 5, "M2": 7},  "workers": 1},
                {"name":"J1O1", "machines": {"M2": 6, "M3": 6},  "workers": 1},
                {"name":"J1O2", "machines": {"M1": 8, "M3": 5},  "workers": 2},
            ],
            "due": 22,
            "tardiness_weight": 2
        },
        # Job 2
        {
            "ops": [
                {"name":"J2O0", "machines": {"M2": 6, "M3": 4},  "workers": 1},
                {"name":"J2O1", "machines": {"M1": 6, "M2": 5},  "workers": 2},
                {"name":"J2O2", "machines": {"M1": 3, "M3": 5},  "workers": 1},
            ],
            "due": 16,
            "tardiness_weight": 4
        },
    ]

    machines = ["M1", "M2", "M3"]
    worker_capacity = 2  # aynı anda en fazla 2 işçi mevcut

    # Global horizon kaba bir üst sınır: tüm maksimum sürelerin toplamı (basit yaklaşım)
    horizon = 0
    for job in jobs:
        for op in job["ops"]:
            horizon += max(op["machines"].values())

    # -----------------------------
    # 2) Değişkenler: Her op için alternatif makinelerde opsiyonel interval
    # -----------------------------
    # op_vars[(j, o)] = {
    #   "start": start var,
    #   "end": end var,
    #   "chosen_m": {m: literal},    # o makine seçildiyse True
    #   "intervals": {m: optional interval}
    # }
    op_vars = {}
    machine_to_intervals = {m: [] for m in machines}
    intervals_for_workers = []
    job_end_vars = []

    for j, job in enumerate(jobs):
        for o, op in enumerate(job["ops"]):
            # Ortak start/end vars (seçilen makineye göre bağlanacak)
            start = model.NewIntVar(0, horizon, f"start_{j}_{o}")
            end   = model.NewIntVar(0, horizon, f"end_{j}_{o}")
            present = {}
            intervals = {}

            # "Exactly one machine" mantığı için boole literal'lar
            chosen_literals = []
            for m, dur in op["machines"].items():
                lit = model.NewBoolVar(f"assign_{j}_{o}_{m}")
                interval = model.NewOptionalIntervalVar(
                    start, dur, end, lit, f"int_{j}_{o}_{m}"
                )
                present[m] = lit
                intervals[m] = interval
                chosen_literals.append(lit)

            # Tam olarak bir makine seçilsin
            model.AddExactlyOne(chosen_literals)

            # Makine takvimlerine ekle
            for m in op["machines"].keys():
                machine_to_intervals[m].append(intervals[m])

            # İşgücü (cumulative) kaynağı: demand = op["workers"]
            # Tek bir interval üzerinden demand bağlamak için "alternatif interval" yapısında
            # hepsini cumulative'e eklemek gerekir (opsiyoneller zaten lit ile kontrol ediliyor).
            for m, dur in op["machines"].items():
                # Cumulative'e eklenecek interval ile demand
                intervals_for_workers.append((intervals[m], op["workers"]))

            op_vars[(j, o)] = {
                "start": start,
                "end": end,
                "chosen_m": present,
                "intervals": intervals,
            }

        # Her işin bitiş zamanı: son operasyonun "end" değişkeni
        job_end_vars.append(op_vars[(j, len(job["ops"]) - 1)]["end"])

    # -----------------------------
    # 3) Kısıtlar
    # -----------------------------

    # 3a) Öncelik (precedence): O0 -> O1 -> O2
    for j, job in enumerate(jobs):
        for o in range(len(job["ops"]) - 1):
            model.Add(op_vars[(j, o)]["end"] <= op_vars[(j, o + 1)]["start"])

    # 3b) Makine NoOverlap: aynı makinede interval çakışmasın
    for m in machines:
        model.AddNoOverlap([iv for iv in machine_to_intervals[m]])

    # 3c) İşgücü cumulative: aynı anda en fazla worker_capacity
    # intervals_for_workers = [(interval, demand), ...]
    model.AddCumulative(
        [iv for iv, d in intervals_for_workers],
        [d  for iv, d in intervals_for_workers],
        worker_capacity
    )

    # 3d) Termin gecikmesi ve gecikme cezası
    tard_vars = []
    for j, job in enumerate(jobs):
        tard = model.NewIntVar(0, horizon, f"tard_{j}")
        # tardiness = max(0, end - due)
        model.Add(tard >= job_end_vars[j] - job["due"])
        model.Add(tard >= 0)
        tard_vars.append((tard, job["tardiness_weight"]))

    # 3e) Makespan
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, job_end_vars)

    # -----------------------------
    # 4) Amaç: alpha * makespan + sum(w_j * tard_j)
    # -----------------------------
    alpha = 2  # makespan ağırlığı
    obj_terms = [alpha * makespan] + [w * t for (t, w) in tard_vars]
    model.Minimize(sum(obj_terms))

    # -----------------------------
    # 5) Çözüm
    # -----------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 8  # paralel çözüm (destek varsa)
    status = solver.Solve(model)

    # -----------------------------
    # 6) Çıktı
    # -----------------------------
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"Status: {solver.StatusName(status)}")
        print(f"Makespan: {solver.Value(makespan)}")
        total_tard = 0
        for j, (t, w) in enumerate(tard_vars):
            tv = solver.Value(t)
            total_tard += w * tv
            print(f"Job {j} end = {solver.Value(job_end_vars[j])}  | due={jobs[j]['due']}  tard={tv} (w={w})")
        print(f"Weighted tardiness: {total_tard}")
        print("Objective:", solver.ObjectiveValue())

        print("\n--- Detailed schedule ---")
        for j, job in enumerate(jobs):
            for o, op in enumerate(job["ops"]):
                st = solver.Value(op_vars[(j, o)]["start"])
                en = solver.Value(op_vars[(j, o)]["end"])
                # Hangi makine seçilmiş?
                chosen = None
                for m, lit in op_vars[(j, o)]["chosen_m"].items():
                    if solver.Value(lit) == 1:
                        chosen = m
                        break
                print(f"{op['name']:>4s} | Job {j} Op {o} | {chosen} | start={st:2d} end={en:2d} | workers={op['workers']}")
    else:
        print("No solution found.")

if __name__ == "__main__":
    flexible_jobshop_example()