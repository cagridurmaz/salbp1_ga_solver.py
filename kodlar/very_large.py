#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, csv, time, random, statistics, heapq, copy
from pathlib import Path
from typing import List, Dict, Optional


# ---------------------------------------------------------
# 1. VERİ OKUMA (AYNI)
# ---------------------------------------------------------
def parse_alb_data(path: Path) -> Dict:
    text = path.read_text (encoding='utf-8', errors='ignore')
    lines = [ln.strip () for ln in text.splitlines ()]

    def find(tag):
        for i, ln in enumerate (lines):
            if ln.lower ().startswith (tag.lower ()): return i
        return -1

    n_idx = find ("<number of tasks>")
    c_idx = find ("<cycle time>")
    t_idx = find ("<task times>")
    p_idx = find ("<precedence relations>")

    if n_idx < 0: raise ValueError (f"Format Hatası: {path.name}")

    n = int (lines[n_idx + 1].split ()[0])
    c_line = lines[c_idx + 1]
    C = int (float (c_line.replace (",", ".")))

    times = {}
    row = t_idx + 1
    while row < len (lines) and not lines[row].startswith ("<"):
        parts = lines[row].split ()
        if len (parts) >= 2: times[int (parts[0])] = int (parts[1])
        row += 1

    edges = []
    if p_idx != -1:
        row = p_idx + 1
        while row < len (lines) and not lines[row].startswith ("<"):
            if "," in lines[row]:
                a, b = map (int, lines[row].split (","))
                edges.append ((a, b))
            row += 1

    # Ters grafik (Reverse Graph) için kenarlar
    rev_edges = [(v, u) for u, v in edges]

    return {"name": path.stem, "n": n, "C": C, "times": times, "edges": edges, "rev_edges": rev_edges}


# ---------------------------------------------------------
# 2. OPTİMİZE EDİLMİŞ GRAF VE DECODER YAPISI
# ---------------------------------------------------------

class FastGraph:
    """Grafik yapısını ve öncülleri hızlı yönetmek için sınıf"""

    def __init__(self, n, edges, times):
        self.n = n
        self.adj = [[] for _ in range (n + 1)]
        self.indeg_template = [0] * (n + 1)
        for u, v in edges:
            self.adj[u].append (v)
            self.indeg_template[v] += 1
        self.times = times


def priority_based_decode(priority_list: List[int], fg: FastGraph, C: int):
    """
    ÖNEMLİ DEĞİŞİKLİK:
    Sıralı liste (priority_list) bir 'yapılacaklar sırası' değil, 'öncelik skoru'dur.

    Algoritma:
    1. Yapılabilir (öncülleri bitmiş) görevleri bul.
    2. Mevcut istasyonda kalan süreye sığan görevler içinden
       priority_list'te en önde olanı seç.
    3. Eğer hiçbiri sığmıyorsa yeni istasyon aç.
    """
    # Öncelikleri map'le (Task ID -> Index)
    # priority_list[0] en yüksek öncelik.
    prio_map = {task: i for i, task in enumerate (priority_list)}

    indeg = list (fg.indeg_template)
    # Heap'te (Priority Score, Task ID) tutuyoruz. Score ne kadar düşükse öncelik o kadar yüksek.
    # İlk başta yapılabilir olanlar (Indegree 0)
    ready_heap = []
    for u in range (1, fg.n + 1):
        if indeg[u] == 0:
            heapq.heappush (ready_heap, (prio_map[u], u))

    stations = []
    current_station = []
    time_left = C

    # Henüz atanmamış ama şu an available olan görevleri heap'ten çıkarıp
    # "sığmayanlar" listesine (buffer) atacağız, istasyon yenilenince geri alacağız.
    buffer = []

    tasks_assigned = 0
    while tasks_assigned < fg.n:
        # 1. Mevcut istasyona sığan en yüksek öncelikli işi bul
        chosen_task = None

        # Heap'ten adayları çekip kontrol et
        temp_popped = []

        while ready_heap:
            score, u = heapq.heappop (ready_heap)
            t_u = fg.times[u]

            if t_u <= time_left:
                # Bulduk! En yüksek öncelikli ve sığıyor.
                chosen_task = u
                # Çektiklerimizi (bu hariç) ve sığmayanları geri koymaya gerek yok,
                # çünkü bu döngü içinde heap yapısını koruyarak ilerleyeceğiz.
                # Ancak, daha önce "sığmadığı için" elediklerimizi (temp_popped) geri koymalıyız.
                break
            else:
                # Sığmadı, kenara ayır
                temp_popped.append ((score, u))

        # Kenara ayırdıklarımızı geri heap'e at
        for item in temp_popped:
            heapq.heappush (ready_heap, item)

        if chosen_task:
            # İstasyona ekle
            current_station.append (chosen_task)
            time_left -= fg.times[chosen_task]
            tasks_assigned += 1

            # Yeni açılan görevleri (successors) heap'e ekle
            for v in fg.adj[chosen_task]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    heapq.heappush (ready_heap, (prio_map[v], v))
        else:
            # Mevcut istasyona hiçbir "ready" görev sığmıyor. İstasyonu kapat.
            stations.append (current_station)
            current_station = []
            time_left = C
            # Buffer mantığına gerek kalmadı çünkü heap'teki herkes
            # yeni istasyonda (Time Left = C) tekrar aday olacak.

    if current_station:
        stations.append (current_station)

    return stations


# ---------------------------------------------------------
# 3. GENETİK OPERATÖRLER
# ---------------------------------------------------------

def evaluate(seq: List[int], fg: FastGraph, rev_fg: FastGraph, C: int):
    """
    Çift Yönlü Değerlendirme (Forward & Backward)
    """
    # 1. Düz Çözüm
    st_fwd = priority_based_decode (seq, fg, C)
    m_fwd = len (st_fwd)

    # 2. Ters Çözüm (Reverse Decoding)
    # Diziyi ters çevirip, ters grafikte çözüyoruz
    seq_rev = seq[::-1]
    st_bwd_raw = priority_based_decode (seq_rev, rev_fg, C)

    # Ters çözümü düzelt (Görevleri ve istasyonları geriye çevir)
    # Ancak m sayısı değişmez, o yüzden sadece m'yi karşılaştırmak yeterli
    m_bwd = len (st_bwd_raw)

    # Hangisi iyiyse onu döndür
    if m_fwd <= m_bwd:
        stations = st_fwd
        m = m_fwd
    else:
        # Ters çözüm daha iyi ise istasyonları fiziksel sıraya sokmamız lazım
        # st_bwd_raw: [[Son İstasyon], [Sondan bir önceki]...]
        # İçerik de tersten doldu.
        stations = []
        for st in reversed (st_bwd_raw):
            stations.append (list (reversed (st)))  # Sadece görsel düzeltme
        m = m_bwd

    # Metrics
    loads = [sum (fg.times[t] for t in st) for st in stations]
    eff = sum (loads) / (m * C)
    max_load = max (loads)
    si = math.sqrt (sum ((max_load - l) ** 2 for l in loads) / m)

    return stations, (-m, -si, eff)  # Fitness tuple


def crossover_ox(p1, p2, n, rng):
    # Order Crossover (OX) - Permütasyon korur
    # Sequence based representation kullanıyoruz ama priority rule olarak decode ediliyor.
    # Bu yüzden topological constraint şart değil (Decoder hallediyor).
    # Ancak Topolojik sıralı genler daha hızlı converge olur.

    # Basitlik ve çeşitlilik için OX:
    cut1, cut2 = sorted (rng.sample (range (n), 2))
    child = [-1] * n
    child[cut1:cut2] = p1[cut1:cut2]

    ptr = 0
    for gene in p2:
        if gene not in p1[cut1:cut2]:
            while ptr < n and child[ptr] != -1:
                ptr += 1
            if ptr < n:
                child[ptr] = gene
    return child


# ---------------------------------------------------------
# 4. MAIN GA LOOP
# ---------------------------------------------------------

def run_solver(data, runs=1, time_limit=60):
    n = data["n"];
    C = data["C"];
    times = data["times"]
    fg = FastGraph (n, data["edges"], times)
    rev_fg = FastGraph (n, data["rev_edges"], times)  # Ters grafik

    # Parametreler
    pop_size = 60  # Large scale için biraz küçük tutuyoruz ki çok iterasyon dönsün
    generations = 999999
    tournament_k = 4

    best_global = None

    for r in range (runs):
        rng = random.Random (1000 + r)
        start_time = time.time ()

        # Initial Pop: Topolojik sıralı rastgele diziler
        # Kahn algoritmasını randomize ederek üretmek en iyisidir
        population = []
        base_seq = list (range (1, n + 1))

        for _ in range (pop_size):
            # Tamamen rastgele shuffle (Decoder bunu düzeltecek zaten)
            # Amaç çeşitlilik
            rng.shuffle (base_seq)
            chrom = list (base_seq)
            st, fit = evaluate (chrom, fg, rev_fg, C)
            population.append ({"chrom": chrom, "fit": fit, "st": st})

        current_best = max (population, key=lambda x: x["fit"])

        gen = 0
        while (time.time () - start_time) < time_limit:
            gen += 1

            # Elitizm (En iyi 2)
            pop_sorted = sorted (population, key=lambda x: x["fit"], reverse=True)
            new_pop = pop_sorted[:2]

            while len (new_pop) < pop_size:
                # Turnuva
                parents = rng.sample (population, tournament_k * 2)
                p1 = max (parents[:tournament_k], key=lambda x: x["fit"])["chrom"]
                p2 = max (parents[tournament_k:], key=lambda x: x["fit"])["chrom"]

                # Crossover
                if rng.random () < 0.9:
                    child_chrom = crossover_ox (p1, p2, n, rng)
                else:
                    child_chrom = list (p1)

                # Mutasyon (Swap)
                if rng.random () < 0.2:  # Mutasyon oranını yüksek tuttum (Local optima'dan kaçış)
                    i, j = rng.sample (range (n), 2)
                    child_chrom[i], child_chrom[j] = child_chrom[j], child_chrom[i]

                # Değerlendirme
                st, fit = evaluate (child_chrom, fg, rev_fg, C)
                new_pop.append ({"chrom": child_chrom, "fit": fit, "st": st})

            population = new_pop
            gen_best = max (population, key=lambda x: x["fit"])
            if gen_best["fit"] > current_best["fit"]:
                current_best = gen_best
                # print(f"Run {r+1} Update: m={-current_best['fit'][0]} (Gen {gen})")

        runtime = time.time () - start_time
        if best_global is None or current_best["fit"] > best_global["fit"]:
            best_global = current_best
            best_global["runtime"] = runtime

        print (
            f"Run {r + 1} Finished: m={-current_best['fit'][0]}, SI={-current_best['fit'][1]:.2f}, Time={runtime:.1f}s")

    return best_global, times, C


# ---------------------------------------------------------
# 5. DOSYA KAYIT
# ---------------------------------------------------------
def save_results(res, times, C, out_path: Path, name):
    m = -res["fit"][0]
    si = -res["fit"][1]
    eff = res["fit"][2]

    # Assignment File
    with open (out_path / f"{name}_solution.csv", "w", newline="") as f:
        w = csv.writer (f)
        w.writerow (["Task", "Station", "Time"])
        for s_idx, station in enumerate (res["st"], 1):
            for t in station:
                w.writerow ([t, s_idx, times[t]])

    # Station Summary
    with open (out_path / f"{name}_stations.csv", "w", newline="") as f:
        w = csv.writer (f)
        w.writerow (["Station", "Load", "Capacity", "Idle", "Tasks"])
        for s_idx, station in enumerate (res["st"], 1):
            load = sum (times[t] for t in station)
            w.writerow ([s_idx, load, C, C - load, str (station)])

    # Metrics
    with open (out_path / "Summary.csv", "a", newline="") as f:
        w = csv.writer (f)
        # Header yoksa ekle kontrolü yapılabilir
        w.writerow ([name, m, f"{si:.4f}", f"{eff:.4f}", f"{res['runtime']:.2f}"])


def main():
    import argparse
    parser = argparse.ArgumentParser ()
    parser.add_argument ("--input", type=str, default=".")
    parser.add_argument ("--time", type=float, default=60.0)
    args = parser.parse_args ()

    in_dir = Path (args.input)
    out_dir = in_dir / "HighPerformance_Solutions"
    out_dir.mkdir (parents=True, exist_ok=True)

    files = list (in_dir.glob ("*.alb"))

    # Summary Header
    with open (out_dir / "Summary.csv", "w", newline="") as f:
        csv.writer (f).writerow (["Instance", "Min_m", "SI", "Eff", "Runtime"])

    for fpath in files:
        print (f"Solving {fpath.name}...")
        try:
            data = parse_alb_data (fpath)
            best, times, C = run_solver (data, runs=3, time_limit=args.time)
            save_results (best, times, C, out_dir, data["name"])
        except Exception as e:
            print (f"Error on {fpath.name}: {e}")


if __name__ == "__main__":
    main ()