"""
plot_ripple_regions.py
=====================
Reads motor current data starting from cell K7 in 002_DX1H_SLP_WithPWM.xlsx
and creates detailed plots for each time region:

  Region layout (based on signal description):
    0   – 2500 ms  : Zero                → 1 plot
    2500 – 7500 ms : Fluctuating (ripples) → 10 plots (500 ms each)
    7500 – 8000 ms : Zero                → 1 plot
    8000 – 14000 ms: Fluctuating (ripples) → 12 plots (500 ms each)
    14000 ms +     : Zero                → 1 plot  (if data exists)

  Total: up to 25 subplots across a multi-page PDF.

Usage:
    python plot_ripple_regions.py
    python plot_ripple_regions.py "002_DX1H_SLP_WithPWM.xlsx" --sheet "12_PWL_Antipinch" --fs 4000
    python plot_ripple_regions.py --output my_plots.pdf
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ── Data Loading ─────────────────────────────────────────────────────
def load_k7_data(filepath, sheet_name='12_PWL_Antipinch', fs=4000):
    """Load data starting from cell K7 (column K, row 7 in Excel).

    K  = 11th column (0-indexed: 10)
    Row 7 in Excel = 0-indexed row 6  (skip first 6 rows)
    """
    print(f"Loading data from '{filepath}' (Sheet: {sheet_name}) ...")
    df = pd.read_excel(filepath, sheet_name=sheet_name, header=None)

    # Cell K7 → iloc[6:, 10]
    raw = df.iloc[6:, 10]
    data = pd.to_numeric(raw, errors='coerce').dropna().values.astype(float)

    time_ms = np.arange(len(data)) / fs * 1000.0  # milliseconds
    print(f"  Loaded {len(data)} samples  |  Duration: {time_ms[-1]:.1f} ms")
    return time_ms, data


# ── Region Builder ───────────────────────────────────────────────────
def build_regions(time_ms):
    """Return a list of plot-region dicts based on the known signal structure.

    Fluctuating regions are split into 500 ms sub-windows so every ripple
    is clearly visible.  Zero (quiet) regions get a single plot each.
    """
    max_time = time_ms[-1] if len(time_ms) > 0 else 14500

    regions = []

    # ── Region 1: Zero (0 – 2500 ms) ─────────────────────────────
    regions.append({
        'title': 'Zero Region  (0 – 2500 ms)',
        'start': 0,
        'end':   2500,
        'fluctuating': False,
    })

    # ── Region 2: Fluctuating (2500 – 7500 ms) ── 10 × 500 ms ───
    for t in range(2500, 7500, 500):
        regions.append({
            'title': f'Fluctuating Region 1  ({t} – {t + 500} ms)',
            'start': t,
            'end':   t + 500,
            'fluctuating': True,
        })

    # ── Region 3: Zero (7500 – 8000 ms) ──────────────────────────
    regions.append({
        'title': 'Zero Region  (7500 – 8000 ms)',
        'start': 7500,
        'end':   8000,
        'fluctuating': False,
    })

    # ── Region 4: Fluctuating (8000 – 14000 ms) ── 12 × 500 ms ──
    for t in range(8000, 14000, 500):
        regions.append({
            'title': f'Fluctuating Region 2  ({t} – {t + 500} ms)',
            'start': t,
            'end':   t + 500,
            'fluctuating': True,
        })

    # ── Region 5: Zero (14000 ms +) ──────────────────────────────
    if max_time > 14000:
        regions.append({
            'title': f'Zero Region  (14000 – {max_time:.0f} ms)',
            'start': 14000,
            'end':   max_time,
            'fluctuating': False,
        })

    return regions


# ── Plotting ─────────────────────────────────────────────────────────
def plot_regions(time_ms, data, output_pdf='ripple_region_plots.pdf',
                 plots_per_page=5):
    """Create a multi-page PDF with one subplot per region."""
    regions = build_regions(time_ms)

    n_pages = (len(regions) + plots_per_page - 1) // plots_per_page

    with PdfPages(output_pdf) as pdf:
        for page in range(n_pages):
            idx_start = page * plots_per_page
            idx_end   = min(idx_start + plots_per_page, len(regions))
            n_plots   = idx_end - idx_start

            fig, axes = plt.subplots(n_plots, 1, figsize=(16, 3.5 * n_plots))
            if n_plots == 1:
                axes = [axes]

            for i, region in enumerate(regions[idx_start:idx_end]):
                ax   = axes[i]
                mask = (time_ms >= region['start']) & (time_ms < region['end'])
                t    = time_ms[mask]
                d    = data[mask]

                # Use blue for ripple regions, grey for zero regions
                color = '#1976D2' if region['fluctuating'] else '#757575'
                lw    = 0.6 if region['fluctuating'] else 0.8

                ax.plot(t, d, color=color, linewidth=lw)
                ax.set_title(region['title'], fontsize=11, fontweight='bold')
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('Current')
                ax.set_xlim(region['start'], region['end'])
                ax.grid(True, alpha=0.3)
                ax.axhline(0, color='black', linewidth=0.5, linestyle='--')

            fig.suptitle(
                'Ripple Region Analysis — 002_DX1H_SLP_WithPWM',
                fontsize=14, fontweight='bold', y=1.02
            )
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    # Summary
    n_zero  = sum(1 for r in regions if not r['fluctuating'])
    n_fluct = sum(1 for r in regions if r['fluctuating'])

    print(f"\n{'=' * 55}")
    print(f"  Saved {len(regions)} plots across {n_pages} pages")
    print(f"  → Zero regions      : {n_zero} plots")
    print(f"  → Fluctuating regions: {n_fluct} plots  (500 ms intervals)")
    print(f"  Output: '{output_pdf}'")
    print(f"{'=' * 55}")


# ── CLI ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Plot ripple regions from motor current data (cell K7)")
    parser.add_argument("filepath", type=str, nargs="?",
                        default="002_DX1H_SLP_WithPWM.xlsx",
                        help="Path to the Excel file")
    parser.add_argument("--sheet", type=str, default="12_PWL_Antipinch",
                        help="Sheet name (default: 12_PWL_Antipinch)")
    parser.add_argument("--fs", type=int, default=4000,
                        help="Sampling frequency in Hz (default: 4000)")
    parser.add_argument("--output", type=str,
                        default="ripple_region_plots.pdf",
                        help="Output PDF filename (default: ripple_region_plots.pdf)")
    parser.add_argument("--plots-per-page", type=int, default=5,
                        help="Number of subplots per PDF page (default: 5)")

    args = parser.parse_args()

    try:
        sheet = int(args.sheet)
    except ValueError:
        sheet = args.sheet

    time_ms, data = load_k7_data(args.filepath, sheet_name=sheet, fs=args.fs)
    plot_regions(time_ms, data, output_pdf=args.output,
                 plots_per_page=args.plots_per_page)


if __name__ == "__main__":
    main()
