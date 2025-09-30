# roofline.py — makes three graphs: one per compiler opt (O0, O2, O3),
# each showing all layout/unroll configs (contig/rows × u1/u4/u8).
#
# Build from matvec.cpp:
#   g++ -O0 matvec.cpp -o matvec_O0
#   g++ -O2 matvec.cpp -o matvec_O2
#   g++ -O3 matvec.cpp -o matvec_O3
#
# Run:
#   python3 roofline.py

import subprocess, shutil, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PEAK_BW_GBPS = 160.0
PEAK_TFLOPS  = 1.0

# Sweep sizes (avoid huge allocs/time)
SIZES = [64, 256, 1024, 4096, 10000]
MAX_ELEMS = 120_000_000   # skip if N*M exceeds this (~0.96 GB for A)
R = "1"                   # repetitions per run

# Executables by optimization
EXES = {
    "O0": "./matvec_O0",
    "O2": "./matvec_O2",
    "O3": "./matvec_O3",
}

# Configs per optimization: layout × unroll
LAYOUTS = ["contig", "rows"]
UNROLLS = ["1", "4", "8"]

# Parser for Matvec output
# Output line: layout,N,M,R,unroll,seconds,GFLOPs,AI,ok
def parse_output(s):
    p = s.strip().split(",")
    if len(p) != 9:
        return None
    layout = p[0].strip()
    N = int(p[1]); M = int(p[2]); Rrun = int(p[3])
    unroll = int(p[4])
    seconds = float(p[5])
    gflops = float(p[6])      # already GFLOP/s (not TFLOP/s)
    ai = float(p[7])
    ok = int(p[8]) == 1
    return layout, N, M, Rrun, unroll, seconds, gflops, ai, ok

# Roofline helpers
AI_MIN, AI_MAX = 1e-2, 1.0
xs = np.logspace(np.log10(AI_MIN), np.log10(AI_MAX), 200)
mem_roof_gflops = ((PEAK_BW_GBPS * xs) / 1000.0) * 1000.0   # -> GFLOP/s
comp_roof_gflops = np.full_like(xs, PEAK_TFLOPS * 1000.0)   # -> GFLOP/s
AI_MATVEC = 0.25
GF_AT_AI = ((PEAK_BW_GBPS * AI_MATVEC) / 1000.0) * 1000.0   # predicted ceiling at AI≈0.25

def nice_ceiling(x):
    import math
    if x <= 0: return 5.0
    m = 10 ** math.floor(math.log10(x))
    for k in (1, 2, 5, 10):
        if k*m >= x: return k*m
    return 10*m

# For each optimization level: collect data, then make one figure
for opt, exe in EXES.items():
    if not shutil.which(exe):
        print(f"[skip] {opt}: {exe} not found on PATH", file=sys.stderr)
        continue

    # Gather series for all (layout, unroll) under this optimization
    # Each series: dict(label, layout, unroll, AI[], GF[])
    series = []
    for layout in LAYOUTS:
        for unroll in UNROLLS:
            AIs, GFs = [], []
            for N in SIZES:
                for M in SIZES:
                    if N * M > MAX_ELEMS:
                        continue
                    cmd = [exe, str(N), str(M), layout, R, unroll]
                    try:
                        out = subprocess.check_output(cmd, text=True).strip()
                    except subprocess.CalledProcessError as e:
                        print(f"[fail] {opt}_{layout}_u{unroll}: {e}", file=sys.stderr)
                        continue
                    parsed = parse_output(out)
                    if not parsed:
                        print(f"[warn] {opt}_{layout}_u{unroll}: unexpected output: {out}", file=sys.stderr)
                        continue
                    _, Np, Mp, _, _, _, gflops, ai, ok = parsed
                    if not ok:
                        print(f"[warn] {opt}_{layout}_u{unroll}: validation failed (N={Np}, M={Mp})", file=sys.stderr)
                        continue
                    AIs.append(ai); GFs.append(gflops)
            if AIs:
                series.append({
                    "label": f"{opt}_{layout}_u{unroll}",
                    "layout": layout,
                    "unroll": unroll,
                    "AI": np.array(AIs),
                    "GF": np.array(GFs)
                })

    if not series:
        print(f"[info] No data for {opt}. Skipping figure.", file=sys.stderr)
        continue

    # Plot (one figure per opt)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    ax.set_xscale("log")
    ax.set_xlim(AI_MIN, AI_MAX)
    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)")
    ax.set_ylabel("Performance (GFLOP/s)")
    ax.set_title(f"Roofline — MatVec ({opt}): layouts {{contig,rows}} × unroll {{1,4,8}}")

    # Dynamic y-limit: fit measured vs predicted matvec ceiling
    max_meas = max(s["GF"].max() for s in series)
    y_target = max(max_meas, GF_AT_AI) * 1.12
    ax.set_ylim(0, nice_ceiling(y_target))

    # Roofs
    ax.plot(xs, mem_roof_gflops, "--", color="tab:blue",  lw=2, label=f"Memory roof ~ {PEAK_BW_GBPS:.0f} GB/s")
    if comp_roof_gflops[0] <= ax.get_ylim()[1]:
        ax.plot(xs, comp_roof_gflops, "--", color="tab:orange", lw=2, label=f"Compute roof ~ {PEAK_TFLOPS:.2f} TFLOP/s")

    # Style: marker = layout, color = unroll (so you can see both at a glance)
    marker_for_layout = {"contig": "o", "rows": "^"}
    color_for_unroll = {"1": "tab:green", "4": "tab:red", "8": "tab:purple"}

    for s in series:
        m = marker_for_layout.get(s["layout"], "s")
        c = color_for_unroll.get(s["unroll"], "tab:gray")
        ax.scatter(s["AI"], s["GF"], s=42, marker=m, color=c,
                   edgecolor="k", linewidths=0.3, label=s["label"])

    # AI≈0.25 guide & predicted ceiling
    ax.axvline(AI_MATVEC, color="gray", ls=":", lw=1)
    ax.scatter([AI_MATVEC], [GF_AT_AI], color="k", marker="D", s=56)
    ax.text(AI_MATVEC*1.02, min(GF_AT_AI*1.05, ax.get_ylim()[1]*0.95),
            f"AI≈0.25 → {GF_AT_AI:.0f} GFLOP/s", fontsize=9)

    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.85)
    
    # Put legend outside so it never covers points
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              framealpha=0.95, borderaxespad=0.)

    fname = f"roofline_{opt}_contig_rows_u1u4u8.png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")
