"""
validate_projections.py

Compares projected DK points against actual DK points for a completed week.

Usage:
    python3 validate_projections.py sim_input_projections_2026_w1.csv 2026 1

Pulls actual DK points from nflreadpy.load_player_stats for the target season/week,
joins against your projection CSV on normalized player name, and reports:
  - MAE (mean absolute error) per position
  - Bias (mean signed error) per position — positive means you're over-projecting
  - Correlation between projection and actual
  - Biggest misses (both over- and under-projections)
"""
import sys
import pandas as pd
import numpy as np
import nflreadpy


def normalize_name(n):
    if pd.isna(n):
        return ''
    return (str(n).replace('.', '').replace("'", '').replace('-', ' ')
            .replace(' Jr', '').replace(' Sr', '').replace(' III', '')
            .lower().strip())


def dk_score_row(r):
    """Recompute DK points from nflreadpy player stats row."""
    pass_pts = r['passing_yards'] * 0.04 + r['passing_tds'] * 4.0 - r.get('passing_interceptions', 0) * 1.0
    rush_pts = r['rushing_yards'] * 0.1 + r['rushing_tds'] * 6.0
    rec_pts  = r['receptions'] * 1.0 + r['receiving_yards'] * 0.1 + r['receiving_tds'] * 6.0
    fumbles = r.get('sack_fumbles_lost', 0) + r.get('rushing_fumbles_lost', 0) + r.get('receiving_fumbles_lost', 0)
    return pass_pts + rush_pts + rec_pts + fumbles * -2.0


def load_actual_week(season, week):
    """Load actual DK points for a completed week."""
    raw = nflreadpy.load_player_stats([season])
    df = raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw)
    df = df[df['position'].isin(['QB', 'RB', 'WR', 'TE'])].copy()
    df = df[df['week'] == week].copy()

    if 'player_display_name' in df.columns:
        df['player_name'] = df['player_display_name']
    elif 'full_name' in df.columns:
        df['player_name'] = df['full_name']

    for c in ['passing_yards', 'passing_tds', 'passing_interceptions',
              'rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards',
              'receiving_tds', 'sack_fumbles_lost', 'rushing_fumbles_lost',
              'receiving_fumbles_lost']:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

    df['actual_dk'] = df.apply(dk_score_row, axis=1)
    df['_key'] = df['player_name'].map(normalize_name)
    return df[['_key', 'player_name', 'position', 'actual_dk']]


def main(projection_csv, season, week):
    print(f"[Validate] Loading projections from {projection_csv}")
    proj = pd.read_csv(projection_csv)
    proj['_key'] = proj['Player'].map(normalize_name)

    print(f"[Validate] Loading actual DK points for {season} week {week}")
    actual = load_actual_week(season, week)

    merged = pd.merge(proj, actual, on='_key', how='inner', suffixes=('_proj', '_act'))
    print(f"[Validate] Matched {len(merged)}/{len(proj)} projected players to actual results.")

    if merged.empty:
        print("[Validate] No matches. Check name normalization or week.")
        return

    # Rename to clean columns
    merged = merged.rename(columns={'Position': 'position', 'Projection': 'proj_dk',
                                    'Ceiling': 'proj_ceil', 'Salary': 'salary',
                                    'Ownership': 'ownership'})
    if 'position_proj' in merged.columns:
        merged['position'] = merged['position_proj']

    merged['error'] = merged['proj_dk'] - merged['actual_dk']
    merged['abs_error'] = merged['error'].abs()

    print("\n=== Accuracy by Position ===")
    print(f"{'Pos':<4} {'N':>4} {'MAE':>7} {'Bias':>7} {'Corr':>7} {'AvgProj':>8} {'AvgAct':>8}")
    print("-" * 55)
    for pos in ['QB', 'RB', 'WR', 'TE']:
        sub = merged[merged['position'] == pos]
        if len(sub) < 3:
            continue
        mae = sub['abs_error'].mean()
        bias = sub['error'].mean()
        corr = sub['proj_dk'].corr(sub['actual_dk'])
        print(f"{pos:<4} {len(sub):>4} {mae:>7.2f} {bias:>+7.2f} {corr:>7.3f} "
              f"{sub['proj_dk'].mean():>8.2f} {sub['actual_dk'].mean():>8.2f}")

    overall_mae = merged['abs_error'].mean()
    overall_bias = merged['error'].mean()
    overall_corr = merged['proj_dk'].corr(merged['actual_dk'])
    print("-" * 55)
    print(f"{'ALL':<4} {len(merged):>4} {overall_mae:>7.2f} {overall_bias:>+7.2f} {overall_corr:>7.3f}")

    print("\n=== Biggest Over-Projections (model too high) ===")
    top_over = merged.nlargest(10, 'error')[['Player', 'position', 'proj_dk', 'actual_dk', 'error', 'salary']]
    print(top_over.to_string(index=False))

    print("\n=== Biggest Under-Projections (model too low) ===")
    top_under = merged.nsmallest(10, 'error')[['Player', 'position', 'proj_dk', 'actual_dk', 'error', 'salary']]
    print(top_under.to_string(index=False))

    # Also report ceiling accuracy
    merged['ceil_error'] = merged['proj_ceil'] - merged['actual_dk']
    print("\n=== Ceiling (95th) vs Actual ===")
    print(f"Mean ceiling error: {merged['ceil_error'].mean():+.2f}")
    print(f"Ceiling hit rate (actual >= 0.9 × ceiling): {(merged['actual_dk'] >= 0.9 * merged['proj_ceil']).mean():.3f}")
    print(f"Ceiling exceed rate (actual > ceiling): {(merged['actual_dk'] > merged['proj_ceil']).mean():.3f}")

    out = f"validation_{season}_w{week}.csv"
    merged.to_csv(out, index=False)
    print(f"\n--> Wrote detailed comparison to {out}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 validate_projections.py <projection_csv> <season> <week>")
        sys.exit(1)
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))