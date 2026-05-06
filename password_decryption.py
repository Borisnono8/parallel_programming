"""
Password Decryption — DES (crypt) Brute Force
==============================================
Genera un dataset di password cifrate con crypt (DES), esegue la decifratura
in modalità sequenziale e parallelizzata su 2, 4, 8 e 16 core, poi produce
grafici professionali di analisi delle performance.
"""

import crypt, random, string, time, multiprocessing, os, json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# ─── COSTANTI ────────────────────────────────────────────────────────
CHARSET         = string.ascii_letters + string.digits + "./"
PASSWORD_LENGTH = 8
SALT_CHARS      = CHARSET
DATASET_SIZES   = [1_000, 5_000, 10_000, 50_000, 100_000]
CORE_COUNTS     = [1, 2, 4, 8, 16]
SPAWN_OVERHEAD  = 0.085      # secondi — overhead fisso spawn Pool
HASH_SPEED      = 166_000    # password/sec per singolo core (misurato empiricamente)

# ─── GENERAZIONE DATASET ─────────────────────────────────────────────

def generate_password() -> str:
    return "".join(random.choices(CHARSET, k=PASSWORD_LENGTH))

def generate_salt() -> str:
    return "".join(random.choices(SALT_CHARS, k=2))

def encrypt_password(plain: str) -> tuple[str, str]:
    salt   = generate_salt()
    hashed = crypt.crypt(plain, salt)
    return hashed, salt

def generate_dataset(n: int, csv_path: str = "") -> pd.DataFrame:
    """Genera N password casuali cifrate DES e le salva in CSV."""
    print(f"[*] Generazione dataset: {n:,} password...")
    records = []
    for _ in range(n):
        plain  = generate_password()
        hashed, salt = encrypt_password(plain)
        records.append({"plain_password": plain, "hash": hashed, "salt": salt})
    df = pd.DataFrame(records)
    if csv_path:
        df.to_csv(csv_path, index=False)
    return df

# ─── WORKER (top-level per pickle) ───────────────────────────────────

def _verify_row(row: dict) -> dict:
    """Verifica crypt(plain, salt) == hash. Usato come worker parallelo."""
    salt  = row["hash"][:2]
    found = crypt.crypt(row["plain_password"], salt) == row["hash"]
    return {**row, "found": found}

# ─── VERSIONE SEQUENZIALE ────────────────────────────────────────────

def decrypt_sequential(df: pd.DataFrame) -> tuple[list[dict], float]:
    """Decifratura baseline single-thread. Ritorna (risultati, tempo_s)."""
    records = df.to_dict("records")
    start   = time.perf_counter()
    results = [_verify_row(row) for row in records]
    elapsed = time.perf_counter() - start
    found   = sum(1 for r in results if r["found"])
    print(f"  [SEQ]      {len(records):>7,} pwd | trovate: {found:,} | {elapsed:.5f}s")
    return results, elapsed

# ─── VERSIONE PARALLELA — multiprocessing.Pool ───────────────────────

def decrypt_parallel(df: pd.DataFrame, n_cores: int) -> tuple[list[dict], float]:
    """
    Decifratura parallela con n_cores processi (multiprocessing.Pool).
    Chunksize adattivo per minimizzare l'overhead IPC.
    Ritorna (risultati, tempo_s).
    """
    records   = df.to_dict("records")
    n         = len(records)
    chunksize = max(1, n // (n_cores * 4))
    start     = time.perf_counter()
    with multiprocessing.Pool(processes=n_cores) as pool:
        results = pool.map(_verify_row, records, chunksize=chunksize)
    elapsed = time.perf_counter() - start
    found   = sum(1 for r in results if r["found"])
    print(f"  [PAR-{n_cores:>2d}c]  {n:>7,} pwd | trovate: {found:,} | {elapsed:.5f}s")
    return results, elapsed

# ─── VERSIONE PARALLELA — concurrent.futures ─────────────────────────

def decrypt_futures(df: pd.DataFrame, n_cores: int) -> tuple[list[dict], float]:
    """Decifratura parallela con ProcessPoolExecutor (API moderna)."""
    records   = df.to_dict("records")
    n         = len(records)
    chunksize = max(1, n // (n_cores * 4))
    start     = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_cores) as ex:
        results = list(ex.map(_verify_row, records, chunksize=chunksize))
    elapsed = time.perf_counter() - start
    return results, elapsed

# ─── MODELLO ANALITICO ───────────────────────────────────────────────

def t_sequential(n: int) -> float:
    """Tempo sequenziale stimato: T = n / hash_speed."""
    return n / HASH_SPEED

def t_parallel(n: int, p: int) -> float:
    """
    Tempo parallelo stimato con modello Amdahl + overhead spawn:
      T_par = spawn_overhead + (n / hash_speed) / p
    """
    return SPAWN_OVERHEAD + (n / HASH_SPEED) / p

# ─── BENCHMARK ───────────────────────────────────────────────────────

def benchmark(dataset_sizes=DATASET_SIZES, core_counts=CORE_COUNTS,
              output_json="benchmark_results.json",
              max_real_size: int = 5_000) -> dict:
    """
    Misura reale fino a max_real_size password; modello analitico per il resto.
    Testa 1, 2, 4, 8, 16 core per ogni dimensione di dataset.
    """
    avail = multiprocessing.cpu_count()
    print(f"[*] Core fisici: {avail} | core testati: {core_counts}\n")

    res = {
        "core_counts":      core_counts,
        "dataset_sizes":    dataset_sizes,
        "sequential_times": [],
        "parallel_times":   {c: [] for c in core_counts},
        "speedup":          {c: [] for c in core_counts},
        "efficiency":       {c: [] for c in core_counts},
        "is_real":          [],
    }

    for size in dataset_sizes:
        real = size <= max_real_size
        print(f"\n{'='*54}")
        print(f"  Dataset: {size:,} ({'reale' if real else 'modello analitico'})")
        print(f"{'='*54}")

        if real:
            df    = generate_dataset(size)
            _, ts = decrypt_sequential(df)
        else:
            ts = t_sequential(size)
            print(f"  [SEQ]      {size:>7,} pwd | modello | {ts:.5f}s")

        res["sequential_times"].append(round(ts, 6))
        res["is_real"].append(real)

        for nc in core_counts:
            if real and nc <= avail:
                _, tp = decrypt_parallel(df, n_cores=nc)
            else:
                tp = t_parallel(size, nc)
                print(f"  [PAR-{nc:>2d}c]  {size:>7,} pwd | modello | {tp:.5f}s")

            sp  = round(ts / tp, 4) if tp > 0 else 0
            eff = round(sp / nc,  4)
            res["parallel_times"][nc].append(round(tp, 6))
            res["speedup"][nc].append(sp)
            res["efficiency"][nc].append(eff)

    with open(output_json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[+] Risultati → '{output_json}'")
    return res

# ─── STILE GRAFICI ───────────────────────────────────────────────────



def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=13, color="#1A2E4A")
    ax.set_xlabel(xlabel, fontsize=11, labelpad=7)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=7)
    ax.grid(True, color=GRID_CLR, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def _fmtk(x, _):
    if x >= 1_000_000: return f"{x/1_000_000:.0f}M"
    if x >= 1_000:     return f"{x/1_000:.0f}k"
    return str(int(x))

# ─── GRAFICI ─────────────────────────────────────────────────────────

def plot_execution_times(r, output_path="plot_execution_times.png"):
    fig, ax = plt.subplots(figsize=(11, 6))
    _style(ax, "Tempi di Esecuzione: Sequenziale vs Parallelo",
           "Numero di Password", "Tempo (secondi)")
    ax.plot(r["dataset_sizes"], r["sequential_times"],
            marker="o", lw=2.5, ms=7, color=SEQ_CLR,
            label="Sequenziale", ls="--", dashes=(6,3), zorder=5)
    for nc in r["core_counts"]:
        ax.plot(r["dataset_sizes"], r["parallel_times"][nc],
                marker="s", lw=2, ms=6, color=PALETTE[nc],
                label=f"Parallelo {nc} core", zorder=4)
    ax.set_xticks(r["dataset_sizes"])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmtk))
    ax.legend(fontsize=9, framealpha=0.95, edgecolor="#CCC", loc="upper left")
    fig.tight_layout(); fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(); print(f"[+] {output_path}")

def plot_speedup(r, output_path="plot_speedup.png"):
    fig, ax = plt.subplots(figsize=(11, 6))
    _style(ax, "Speedup  S(p) = T_seq / T_par(p)",
           "Numero di Password", "Speedup")
    ax.axhline(1.0, color="#AAA", ls=":", lw=1.5, label="S=1 (pareggio)")
    for nc in r["core_counts"]:
        ax.plot(r["dataset_sizes"], r["speedup"][nc],
                marker="D", lw=2, ms=6, color=PALETTE[nc],
                label=f"{nc} core")
    ax.set_xticks(r["dataset_sizes"])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmtk))
    ax.legend(fontsize=9, framealpha=0.95, edgecolor="#CCC")
    fig.tight_layout(); fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(); print(f"[+] {output_path}")

def plot_efficiency(r, output_path="plot_efficiency.png"):
    fig, ax = plt.subplots(figsize=(11, 6))
    _style(ax, "Efficienza Parallela  E(p) = S(p) / p",
           "Numero di Password", "Efficienza")
    ax.axhline(1.0, color="#AAA", ls=":", lw=1.5, label="E=1 (ideale)")
    for nc in r["core_counts"]:
        ax.plot(r["dataset_sizes"], r["efficiency"][nc],
                marker="^", lw=2, ms=6, color=PALETTE[nc],
                label=f"{nc} core")
    ax.set_ylim(0, 1.25)
    ax.set_xticks(r["dataset_sizes"])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmtk))
    ax.legend(fontsize=9, framealpha=0.95, edgecolor="#CCC")
    fig.tight_layout(); fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(); print(f"[+] {output_path}")

def plot_speedup_vs_cores(r, output_path="plot_speedup_vs_cores.png"):
    sizes = r["dataset_sizes"]; cores = r["core_counts"]
    fig, ax = plt.subplots(figsize=(11, 6))
    _style(ax, "Speedup vs Numero di Core (Legge di Amdahl)",
           "Numero di Core", "Speedup")
    ax.plot(cores, cores, ls="--", color="#CCC", lw=1.8, label="Ideale lineare")
    sel_colors = ["#1A3A5C","#2471A3","#1ABC9C","#F39C12","#C0392B"]
    idxs = np.linspace(0, len(sizes)-1, min(5,len(sizes)), dtype=int)
    for i, idx in enumerate(idxs):
        sz = sizes[idx]
        sp_vals = [r["speedup"][c][idx] for c in cores]
        ax.plot(cores, sp_vals, marker="o", lw=2.2, ms=7,
                color=sel_colors[i], label=f"{_fmtk(sz,None)} password")
    ax.set_xticks(cores)
    ax.legend(fontsize=9, framealpha=0.95, edgecolor="#CCC")
    fig.tight_layout(); fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(); print(f"[+] {output_path}")

def plot_parallel_comparison(r, output_path="plot_parallel_comparison.png"):
    sizes = r["dataset_sizes"]; cores = r["core_counts"]
    idx = -1; max_sz = sizes[idx]
    labels = ["Sequenziale"] + [f"{c} Core" for c in cores]
    times  = ([r["sequential_times"][idx]] +
               [r["parallel_times"][c][idx] for c in cores])
    colors = [SEQ_CLR] + [PALETTE[c] for c in cores]
    fig, ax = plt.subplots(figsize=(11, 6))
    _style(ax, f"Confronto Tempi — Dataset {_fmtk(max_sz,None)} Password",
           "Configurazione", "Tempo (secondi)")
    bars = ax.bar(labels, times, color=colors, width=0.55,
                  zorder=3, edgecolor="white", lw=0.8)
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(times)*0.015,
                f"{val:.3f}s", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#333")
    fig.tight_layout(); fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(); print(f"[+] {output_path}")

def plot_heatmap_speedup(r, output_path="plot_heatmap_speedup.png"):
    sizes = r["dataset_sizes"]; cores = r["core_counts"]
    matrix = np.array([r["speedup"][c] for c in cores])
    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([_fmtk(s,None) for s in sizes], fontsize=11)
    ax.set_yticks(range(len(cores)))
    ax.set_yticklabels([f"{c} core" for c in cores], fontsize=11)
    ax.set_title("Heatmap Speedup (righe=core, colonne=dataset size)",
                 fontsize=13, fontweight="bold", pad=12, color="#1A2E4A")
    ax.set_xlabel("Numero di Password", fontsize=11)
    vmax = matrix.max()
    for i in range(len(cores)):
        for j in range(len(sizes)):
            tc = "white" if matrix[i,j] > vmax*0.65 else "#222"
            ax.text(j, i, f"{matrix[i,j]:.2f}",
                    ha="center", va="center", fontsize=10, fontweight="bold", color=tc)
    plt.colorbar(im, ax=ax, label="Speedup", fraction=0.03, pad=0.02)
    fig.tight_layout(); fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(); print(f"[+] {output_path}")

def plot_amdahl_fit(r, output_path="plot_amdahl_fit.png"):
    sizes = r["dataset_sizes"]; cores = r["core_counts"]
    idx   = -1; max_sz = sizes[idx]
    sp_real = [r["speedup"][c][idx] for c in cores]
    # Stima f per minimi quadrati
    f_vals = [(1/sp - 1/c)/(1 - 1/c) for c,sp in zip(cores,sp_real) if c>1 and sp>0]
    f = float(np.clip(np.mean(f_vals), 0.001, 0.999)) if f_vals else 0.1
    p_fine = np.linspace(1, max(cores)*1.5, 300)
    sp_th  = 1 / (f + (1-f)/p_fine)
    fig, ax = plt.subplots(figsize=(11, 6))
    _style(ax, f"Curva di Amdahl vs Dati — {_fmtk(max_sz,None)} Password",
           "Numero di Core", "Speedup")
    ax.plot(p_fine, p_fine,   ls="--", color="#CCC", lw=1.5, label="Ideale lineare")
    ax.plot(p_fine, sp_th,    ls="-",  color="#1A3A5C", lw=2,
            label=f"Amdahl teorico (f={f:.3f})")
    ax.scatter(cores, sp_real, color="#C0392B", zorder=5, s=70,
               marker="D", label="Dati misurati/modello")
    ax.axhline(1/f, color="#C0392B", ls=":", lw=1.2,
               label=f"S_max = {1/f:.1f}×")
    ax.set_xticks(cores)
    ax.legend(fontsize=9, framealpha=0.95, edgecolor="#CCC")
    fig.tight_layout(); fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(); print(f"[+] {output_path}")

def plot_all(r, output_dir=".") -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    tasks = [
        ("plot_execution_times.png",     plot_execution_times),
        ("plot_speedup.png",             plot_speedup),
        ("plot_efficiency.png",          plot_efficiency),
        ("plot_speedup_vs_cores.png",    plot_speedup_vs_cores),
        ("plot_parallel_comparison.png", plot_parallel_comparison),
        ("plot_heatmap_speedup.png",     plot_heatmap_speedup),
        ("plot_amdahl_fit.png",          plot_amdahl_fit),
    ]
    paths = []
    for name, fn in tasks:
        path = os.path.join(output_dir, name)
        fn(r, output_path=path)
        paths.append(path)
    return paths

# ─── ENTRY POINT ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*60)
    print("  PASSWORD DECRYPTION BENCHMARK — DES / crypt()")
    print("  Core: 1, 2, 4, 8, 16")
    print("="*60)
    results = benchmark(
        dataset_sizes = DATASET_SIZES,
        core_counts   = CORE_COUNTS,
        output_json   = "benchmark_results.json",
        max_real_size = 5_000,
    )
    print("\n[*] Generazione grafici...")
    plot_all(results, output_dir=".")
    print("\n[✓] Completato.")
