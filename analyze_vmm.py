"""
analyze_vmm.py
----------------
Reads runtime_basic.csv, runtime_vectorized.csv, runtime_blas.csv and
runtime_omp_1/4/16/64.csv, computes MFLOP/s (and % memory bandwidth) for
each run, and produces the charts required by the report:

  1. chart1_mflops.pdf            -> CBLAS vs Basic VMM vs Vectorized VMM
  2. chart2_speedup.pdf           -> OpenMP speedup (4 datasets: 1/4/16/64 threads)
  3. chart3_best_omp_vs_blas.pdf  -> best OpenMP config vs serial CBLAS

Usage:
    python3 analyze_vmm.py
Run it in the same directory as the CSV files (e.g. your build/ dir),
or edit the paths in main() below.
"""

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# FLOP / byte counting
# ---------------------------------------------------------------------------
# y := y + A*x for n x n matrix A.
# Each of the n^2 elements of A: 1 multiply + 1 add -> 2*n^2 FLOPs.
def flops_for_n(n: int) -> float:
    return 2.0 * (n ** 2)


# Bytes moved (double = 8 bytes): read A (n^2) + read x (n) + read y (n) + write y (n)
# TODO: change if the assignment/lecture defines it differently.
def bytes_for_n(n: int) -> float:
    return 8.0 * (n ** 2 + 3 * n)


# TODO: peak memory bandwidth of the machine in GB/s (204.8 is only a placeholder).
PEAK_BW_GBS = 204.8


def add_mflops_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mflops"] = flops_for_n(df["problem_size"]) / df["elapsed_time_sec"] / 1e6
    return df


def add_bw_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    gbs = bytes_for_n(df["problem_size"]) / df["elapsed_time_sec"] / 1e9
    df["bw_pct"] = 100.0 * gbs / PEAK_BW_GBS
    return df


# ---------------------------------------------------------------------------
# Warm-up row removal
# ---------------------------------------------------------------------------
# test_sizes = {1024, 1024, 2048, ...} -> N=1024 is intentionally run twice.
# Drop the FIRST row (the warm-up run) before computing/plotting anything.
def drop_warmup(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[1:].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_data(basic_csv="runtime_basic.csv",
              vec_csv="runtime_vectorized.csv",
              blas_csv="runtime_blas.csv",
              omp_csvs=("runtime_omp_1.csv", "runtime_omp_4.csv",
                        "runtime_omp_16.csv", "runtime_omp_64.csv")):

    basic = add_bw_column(add_mflops_column(drop_warmup(pd.read_csv(basic_csv))))
    vec = add_bw_column(add_mflops_column(drop_warmup(pd.read_csv(vec_csv))))
    blas = add_bw_column(add_mflops_column(drop_warmup(pd.read_csv(blas_csv))))

    # OpenMP runs: list of (label, dataframe), one entry per thread count
    omps = []
    for threads, csv_file in zip([1, 4, 16, 64], omp_csvs):
        df = add_bw_column(add_mflops_column(drop_warmup(pd.read_csv(csv_file))))
        omps.append((f"omp-{threads}", df))

    # sanity check: flag any run that failed the correctness check
    for name, df in [("basic", basic), ("vectorized", vec), ("blas", blas)] + omps:
        bad = df[df["correct"] == 0]
        if len(bad):
            print(f"WARNING: {name} has {len(bad)} row(s) that failed correctness check:")
            print(bad)

    return basic, vec, blas, omps


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def plot_mflops(basic, vec, blas, outfile="chart1_mflops.pdf"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(basic["problem_size"], basic["mflops"], marker="o", label="Basic VMM")
    ax.plot(vec["problem_size"], vec["mflops"], marker="^", label="Vectorized VMM")
    ax.plot(blas["problem_size"], blas["mflops"], marker="s", label="CBLAS")
    ax.set_xlabel("Problem size (N)")
    ax.set_ylabel("MFLOP/s")
    ax.set_title("Basic vs Vectorized VMM vs CBLAS")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile)
    print(f"wrote {outfile}")


def plot_speedup(omps, outfile="chart2_speedup.pdf"):
    # speedup = time with 1 thread / time with P threads (same problem size)
    t1 = omps[0][1]["elapsed_time_sec"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, df in omps:
        ax.plot(df["problem_size"], t1 / df["elapsed_time_sec"], marker="o", label=name)
    ax.set_xlabel("Problem size (N)")
    ax.set_ylabel("Speedup (vs omp-1)")
    ax.set_title("OpenMP VMM speedup (static scheduling)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile)
    print(f"wrote {outfile}")


def plot_best_omp_vs_blas(omps, blas, outfile="chart3_best_omp_vs_blas.pdf"):
    # best config = highest average MFLOP/s over all problem sizes
    # (to pick by hand instead, replace with e.g.  best_name, best = omps[2])
    best_name, best = max(omps, key=lambda item: item[1]["mflops"].mean())
    print(f"Best OpenMP configuration: {best_name}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(best["problem_size"], best["mflops"], marker="o", label=best_name)
    ax.plot(blas["problem_size"], blas["mflops"], marker="s", linestyle="--",
            color="black", label="CBLAS (serial)")
    ax.set_xlabel("Problem size (N)")
    ax.set_ylabel("MFLOP/s")
    ax.set_title("Best OpenMP VMM vs CBLAS")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile)
    print(f"wrote {outfile}")


# ---------------------------------------------------------------------------
# Summary tables (handy for writing the discussion paragraphs)
# ---------------------------------------------------------------------------
def print_summary(basic, vec, blas, omps):
    runs = [("CBLAS", blas), ("basic", basic), ("vectorized", vec)] + omps

    print("\n=== MFLOP/s ===")
    mflops = pd.DataFrame({"problem_size": basic["problem_size"]})
    for name, df in runs:
        mflops[name] = df["mflops"]
    print(mflops.round(1).to_string(index=False))

    print("\n=== OpenMP speedup (vs omp-1) ===")
    speedup = pd.DataFrame({"problem_size": basic["problem_size"]})
    for name, df in omps:
        speedup[name] = omps[0][1]["elapsed_time_sec"] / df["elapsed_time_sec"]
    print(speedup.round(2).to_string(index=False))

    print(f"\n=== % of peak memory bandwidth (peak = {PEAK_BW_GBS} GB/s) ===")
    bw = pd.DataFrame({"problem_size": basic["problem_size"]})
    for name, df in runs:
        bw[name] = df["bw_pct"]
    print(bw.round(2).to_string(index=False))
    bw.round(2).to_csv("bandwidth_table.csv", index=False)


# ---------------------------------------------------------------------------
def main():
    basic, vec, blas, omps = load_data()
    print_summary(basic, vec, blas, omps)
    plot_mflops(basic, vec, blas)
    plot_speedup(omps)
    plot_best_omp_vs_blas(omps, blas)


if __name__ == "__main__":
    main()