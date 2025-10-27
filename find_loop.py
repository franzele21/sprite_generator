import sys
import os
import argparse
import pandas as pd

"""
find_loop.py

Read a CSV (default: clean_srpites_latent.csv), sort by pkmn_nb and idx,
and try to detect loops in the latent sequences per pkmn_nb.

Usage:
    python find_loop.py [path/to/clean_srpites_latent.csv] [--decimals N]

Outputs:
    - prints a short summary to stdout
    - writes loops_summary.csv next to the input file
"""

def minimal_period(seq):
    """Return smallest p >= 1 such that for all i, seq[i] == seq[i % p]."""
    n = len(seq)
    if n == 0:
        return None
    for p in range(1, n + 1):
        ok = True
        for i in range(n):
            if seq[i] != seq[i % p]:
                ok = False
                break
        if ok:
            return p
    return None

def max_prefix_suffix_overlap(seq):
    """Return maximum k (0..n-1) such that suffix of length k equals prefix of length k."""
    n = len(seq)
    best = 0
    for k in range(1, n):
        if seq[:k] == seq[-k:]:
            best = k
    return best

def main():
    parser = argparse.ArgumentParser(description="Find loops in latent CSV per pkmn_nb")
    parser.add_argument("csv", nargs="?", default="clean_sprites_latents.csv",
                        help="CSV file (default: clean_sprites_latents.csv)")
    parser.add_argument("--decimals", type=int, default=6,
                        help="round float latent columns to this many decimals before comparing (default 6)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: file not found: {args.csv}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(args.csv)
    if 'pkmn_nb' not in df.columns or 'idx' not in df.columns:
        print("ERROR: CSV must contain 'pkmn_nb' and 'idx' columns", file=sys.stderr)
        sys.exit(2)

    # sort by pkmn_nb then idx
    df = df.sort_values(['pkmn_nb', 'idx']).reset_index(drop=True)

    # latent columns are all except pkmn_nb and idx
    latent_cols = [c for c in df.columns if c not in ('pkmn_nb', 'idx')]
    if not latent_cols:
        print("ERROR: No latent columns found (only 'pkmn_nb' and 'idx' present)", file=sys.stderr)
        sys.exit(2)

    # Round floats to avoid tiny FP differences
    df[latent_cols] = df[latent_cols].apply(lambda s: pd.to_numeric(s, errors='coerce'))
    df[latent_cols] = df[latent_cols].round(args.decimals).fillna(0)

    rows = []
    min = 5000
    max = 0
    for pkmn_nb, g in df.groupby('pkmn_nb', sort=True):
        g_sorted = g.sort_values('idx')
        seq = [tuple(row) for row in g_sorted[latent_cols].to_numpy()]
        n = len(seq)
        period = minimal_period(seq)
        overlap = max_prefix_suffix_overlap(seq)
        looped = (period is not None and period < n) or (overlap > 0)
        rows.append({
            'pkmn_nb': pkmn_nb,
            'n_frames': n,
            'period': period if period is not None else '',
            'overlap_prefix_suffix': overlap,
            'is_loop_detected': bool(looped)
        })
        if n < min:
            min = n
        if n > max:
            max = n

    out_df = pd.DataFrame(rows).sort_values('pkmn_nb')
    out_csv = os.path.join(os.path.dirname(os.path.abspath(args.csv)), 'loops_summary.csv')
    out_df.to_csv(out_csv, index=False)

    # Print short summary
    total = len(out_df)
    loops = out_df['is_loop_detected'].sum()
    print(f"Processed {total} pkmn_nb groups. Loops detected in {loops}. Summary written to: {out_csv}")

    # Optionally print detailed lines (one per pokemon)
    for _, r in out_df.iterrows():
        print(f"pkmn_nb={r['pkmn_nb']}: frames={r['n_frames']}, period={r['period']}, overlap={r['overlap_prefix_suffix']}, loop={r['is_loop_detected']}")

    print(f"Overall frames range: min={min}, max={max}")
    print(f"median frames: {out_df['n_frames'].median()} \nfirst quartile: {out_df['n_frames'].quantile(0.25)} \nthird quartile: {out_df['n_frames'].quantile(0.75)}")

if __name__ == "__main__":
    main()