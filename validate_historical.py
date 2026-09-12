"""
validate_historical.py

Runs the projection pipeline against a past week and compares to actual results.

Usage:
    # Single week
    python3 validate_historical.py --stats-season 2024 --target-season 2025 --week 6

    # Full season sweep (every week of the target season)
    python3 validate_historical.py --stats-season 2024 --target-season 2025 --season-sweep

The script:
  1. Runs dfs_final_model.DKSimulatorDataPipeline for the given config
  2. Loads actual DK points from nflreadpy for the target season/week
  3. Joins, computes MAE/bias/correlation per position
  4. Writes per-week and aggregate CSVs for tuning
"""
import argparse
import sys
import pandas as pd
import numpy as np
import nflreadpy
from pathlib import Path

from dfs_final_model import DKSimulatorDataPipeline


def normalize_name(n):
    if pd.isna(n):
        return ''
    return (str(n).replace('.', '').replace("'", '').replace('-', ' ')
            .replace(' Jr', '').replace(' Sr', '').replace(' III', '')
            .lower().strip())


def dk_score_row(r):
    pass_pts = r['passing_yards'] * 0.04 + r['passing_tds'] * 4.0 - r['passing_interceptions'] * 1.0
    rush_pts = r['rushing_yards'] * 0.1 + r['rushing_tds'] * 6.0
    rec_pts  = r['receptions'] * 1.0 + r['receiving_yards'] * 0.1 + r['receiving_tds'] * 6.0
    fumbles = r['sack_fumbles_lost'] + r['rushing_fumbles_lost'] + r['receiving_fumbles_lost']
    return pass_pts + rush_pts + rec_pts + fumbles * -2.0


def load_actual_week(season, week):
    """Load actual DK points for a completed NFL week."""
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

    # Dedupe: one row per player (should already be one, but defensive)
    df = df.sort_values('actual_dk', ascending=False).drop_duplicates('_key')
    return df[['_key', 'player_name', 'position', 'actual_dk']]


def run_projection_for_week(stats_season, target_season, week, salary_csv):
    """Run the pipeline and return the sim export dataframe."""
    print(f"\n{'='*60}")
    print(f"[Validate] Projecting {target_season} week {week} using {stats_season} stats")
    print(f"{'='*60}\n")

    model = DKSimulatorDataPipeline(
        stats_season=stats_season,
        target_season=target_season,
        target_week=week,
        dk_salary_csv=salary_csv,
    )
    result = model.run_gpp_optimized_pipeline(injuries={})
    return result


def evaluate_week(projections_df, actual_df, target_season, week):
    """Join projections with actuals and compute error metrics."""
    proj = projections_df.copy()
    proj['_key'] = proj['Player'].map(normalize_name)

    merged = pd.merge(proj, actual_df, on='_key', how='inner', suffixes=('_proj', '_act'))
    if merged.empty:
        print(f"[Validate] No matches for {target_season} week {week}")
        return None

    # Ensure position column is present
    if 'position' not in merged.columns:
        merged['position'] = merged['Position']
    elif 'Position' in merged.columns:
        merged['position'] = merged['position'].fillna(merged['Position'])

    merged['proj_dk'] = merged['Projection']
    merged['error'] = merged['proj_dk'] - merged['actual_dk']
    merged['abs_error'] = merged['error'].abs()
    merged['season'] = target_season
    merged['week'] = week

    return merged


def summarize_week(merged, target_season, week):
    """Print per-position accuracy for one week."""
    if merged is None or merged.empty:
        print(f"[Validate] No data for {target_season} week {week}")
        return None

    rows = []
    print(f"\n=== {target_season} Week {week} ===")
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
        rows.append({
            'season': target_season, 'week': week, 'position': pos,
            'n': len(sub), 'mae': round(mae, 3), 'bias': round(bias, 3),
            'corr': round(corr, 4),
            'avg_proj': round(sub['proj_dk'].mean(), 3),
            'avg_act': round(sub['actual_dk'].mean(), 3),
        })

    overall_mae = merged['abs_error'].mean()
    overall_bias = merged['error'].mean()
    overall_corr = merged['proj_dk'].corr(merged['actual_dk'])
    print("-" * 55)
    print(f"{'ALL':<4} {len(merged):>4} {overall_mae:>7.2f} {overall_bias:>+7.2f} {overall_corr:>7.3f}")
    rows.append({
        'season': target_season, 'week': week, 'position': 'ALL',
        'n': len(merged), 'mae': round(overall_mae, 3), 'bias': round(overall_bias, 3),
        'corr': round(overall_corr, 4),
        'avg_proj': round(merged['proj_dk'].mean(), 3),
        'avg_act': round(merged['actual_dk'].mean(), 3),
    })
    return rows


def print_aggregate(all_rows):
    """Print cross-week aggregate metrics."""
    if not all_rows:
        return
    df = pd.DataFrame(all_rows)
    print(f"\n{'='*60}")
    print("AGGREGATE ACROSS ALL VALIDATED WEEKS")
    print(f"{'='*60}")
    agg = df.groupby('position').agg(
        n=('n', 'sum'),
        mae=('mae', 'mean'),
        bias=('bias', 'mean'),
        corr=('corr', 'mean'),
    ).round(3)
    print(agg.to_string())
    df.to_csv('validation_aggregate.csv', index=False)
    print(f"\n--> Wrote aggregate results to validation_aggregate.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats-season', type=int, required=True)
    parser.add_argument('--target-season', type=int, required=True)
    parser.add_argument('--week', type=int, default=None)
    parser.add_argument('--season-sweep', action='store_true',
                        help='Validate every week of the target season (weeks 1-18)')
    parser.add_argument('--salary-csv', type=str, default=None,
                        help='Optional DK salary CSV. Without it, projections run '
                             'without salary, and the pool is unpruned.')
    parser.add_argument('--weeks', type=str, default=None,
                        help='Comma-separated week list, e.g. 6,7,8')
    args = parser.parse_args()

    if args.season_sweep:
        weeks = list(range(1, 19))
    elif args.weeks:
        weeks = [int(w) for w in args.weeks.split(',')]
    elif args.week is not None:
        weeks = [args.week]
    else:
        print("Specify --week, --weeks, or --season-sweep")
        sys.exit(1)

    all_rows = []
    all_merged = []

    for wk in weeks:
        try:
            proj_df = run_projection_for_week(
                args.stats_season, args.target_season, wk, args.salary_csv
            )
            actual_df = load_actual_week(args.target_season, wk)
            merged = evaluate_week(proj_df, actual_df, args.target_season, wk)
            rows = summarize_week(merged, args.target_season, wk)
            if rows:
                all_rows.extend(rows)
            if merged is not None and not merged.empty:
                all_merged.append(merged)
        except Exception as e:
            print(f"[Validate] Week {wk} failed: {e}")
            continue

    if all_rows:
        print_aggregate(all_rows)

    if all_merged:
        full = pd.concat(all_merged, ignore_index=True)
        out = f"validation_{args.stats_season}_stats_{args.target_season}_actuals.csv"
        full.to_csv(out, index=False)
        print(f"--> Wrote detailed player-level data to {out}")


if __name__ == "__main__":
    main()