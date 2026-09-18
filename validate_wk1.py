"""
validate_w1.py

Compares a pipeline output CSV against week 1 actuals.

Usage:
    python3 validate_w1.py sim_input_projections_2026_w1.csv last_week_results.csv
"""
import sys
import pandas as pd
import numpy as np


def norm_name(n):
    if pd.isna(n):
        return ''
    return (str(n).replace('.', '').replace("'", '').replace('-', ' ')
            .replace(' Jr', '').replace(' Sr', '').replace(' III', '')
            .lower().strip())


def main(proj_path, results_path):
    print(f"[Validate] Loading projections from {proj_path}")
    proj = pd.read_csv(proj_path)
    print(f"  {len(proj)} players, columns: {proj.columns.tolist()}")

    print(f"\n[Validate] Loading week 1 actuals from {results_path}")
    actuals = pd.read_csv(results_path)
    print(f"  {len(actuals)} players")

    # Normalize names
    proj['_key'] = proj['Player'].map(norm_name)
    actuals['_key'] = actuals['player_name'].map(norm_name)

    merged = pd.merge(proj, actuals, on='_key', how='inner', suffixes=('_proj', '_act'))
    print(f"\n[Validate] Matched {len(merged)}/{len(proj)} projected players")

    if merged.empty:
        print("[Validate] No matches — check name normalization.")
        return

    merged['error'] = merged['Projection'] - merged['actual_dk']
    merged['abs_error'] = merged['error'].abs()

    print("\n" + "=" * 80)
    print("ACCURACY BY POSITION")
    print("=" * 80)
    print(f"{'Pos':<4} {'N':>4} {'MAE':>7} {'Bias':>7} {'Corr':>7} {'AvgProj':>9} {'AvgAct':>9}")
    print("-" * 60)

    for pos in ['QB', 'RB', 'WR', 'TE', 'DST']:
        sub = merged[merged['Position'] == pos]
        if len(sub) < 3:
            continue
        mae = sub['abs_error'].mean()
        bias = sub['error'].mean()
        corr = sub['Projection'].corr(sub['actual_dk'])
        print(f"{pos:<4} {len(sub):>4} {mae:>7.2f} {bias:>+7.2f} {corr:>7.3f} "
              f"{sub['Projection'].mean():>9.2f} {sub['actual_dk'].mean():>9.2f}")

    overall_mae = merged['abs_error'].mean()
    overall_bias = merged['error'].mean()
    overall_corr = merged['Projection'].corr(merged['actual_dk'])
    print("-" * 60)
    print(f"{'ALL':<4} {len(merged):>4} {overall_mae:>7.2f} {overall_bias:>+7.2f} {overall_corr:>7.3f}")

    print("\n" + "=" * 80)
    print("BIGGEST OVER-PROJECTIONS (model too high)")
    print("=" * 80)
    over = merged.nlargest(12, 'error')[['Player', 'Position', 'Projection', 'actual_dk', 'error', 'Salary']]
    for _, r in over.iterrows():
        print(f"  {r['Player']:<25} {r['Position']:<3} proj={r['Projection']:>6.2f} "
              f"actual={r['actual_dk']:>6.2f} error={r['error']:>+6.2f}")

    print("\n" + "=" * 80)
    print("BIGGEST UNDER-PROJECTIONS (model too low)")
    print("=" * 80)
    under = merged.nsmallest(12, 'error')[['Player', 'Position', 'Projection', 'actual_dk', 'error', 'Salary']]
    for _, r in under.iterrows():
        print(f"  {r['Player']:<25} {r['Position']:<3} proj={r['Projection']:>6.2f} "
              f"actual={r['actual_dk']:>6.2f} error={r['error']:>+6.2f}")

    merged.to_csv('validation_w1_detailed.csv', index=False)
    print(f"\n[Validate] Wrote validation_w1_detailed.csv")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 validate_w1.py <projections_csv> <results_csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])