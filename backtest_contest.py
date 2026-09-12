"""
backtest_contest.py

Analyzes a DraftKings contest export to see how your projections would have performed.

Assumes the contest CSV has one row per lineup entry with columns:
  EntryId, Contest_Key, Points, Rank, Lineup (comma-separated player names)
Or use DK's standard export format which has one row per (entry, player) pair:
  EntryId, Contest_Key, Points, Rank, Name, Roster Position, Salary

Usage:
    python3 backtest_contest.py <contest_csv> <projection_csv> <payout_structure_json>

If payout_structure_json isn't provided, uses a rough DK GPP payout curve.
"""
import sys
import pandas as pd
import numpy as np
import json


def normalize_name(n):
    if pd.isna(n):
        return ''
    return (str(n).replace('.', '').replace("'", '').replace('-', ' ')
            .replace(' Jr', '').replace(' Sr', '').replace(' III', '')
            .lower().strip())


def payout_for_rank(rank, total_entries, buy_in, payout_curve):
    """Approximate DK GPP payout. payout_curve is list of (percentile, multiplier)."""
    pct = rank / total_entries
    for cutoff, mult in payout_curve:
        if pct <= cutoff:
            return buy_in * mult
    return 0.0


DEFAULT_PAYOUT = [
    (0.001, 100.0),   # top 0.1% → 100x buy-in
    (0.005, 20.0),    # top 0.5% → 20x
    (0.01, 8.0),      # top 1% → 8x
    (0.05, 3.0),      # top 5% → 3x
    (0.15, 1.5),      # top 15% → 1.5x
    (0.25, 0.5),      # top 25% → 0.5x (min cash)
]


def compute_lineup_metrics(contest_df, projection_df):
    """
    contest_df: one row per (entry, player) with columns EntryId, Name, Points, Rank
    Returns per-entry aggregate with lineup composition.
    """
    proj_lookup = dict(zip(projection_df['_key'], projection_df['Projection']))
    proj_own_lookup = dict(zip(projection_df['_key'], projection_df['Ownership']))

    # Group by entry
    entries = []
    for entry_id, group in contest_df.groupby('EntryId'):
        lineup_players = group['Name'].tolist()
        lineup_keys = [normalize_name(n) for n in lineup_players]

        proj_total = sum(proj_lookup.get(k, 0) for k in lineup_keys)
        own_total = sum(proj_own_lookup.get(k, 0) for k in lineup_keys)

        entries.append({
            'EntryId': entry_id,
            'Actual_Points': group['Points'].iloc[0],
            'Rank': group['Rank'].iloc[0],
            'Projected_Points': proj_total,
            'Projected_Ownership_Sum': own_total,
            'Lineup': ', '.join(lineup_players),
        })

    return pd.DataFrame(entries)


def main(contest_csv, projection_csv, total_entries=None, buy_in=20.0):
    print(f"[Backtest] Loading contest export from {contest_csv}")
    contest = pd.read_csv(contest_csv)
    print(f"[Backtest] Contest rows: {len(contest)}")

    print(f"[Backtest] Loading projections from {projection_csv}")
    proj = pd.read_csv(projection_csv)
    proj['_key'] = proj['Player'].map(normalize_name)

    # Detect contest format
    if 'Lineup' in contest.columns:
        # One row per entry, lineup is comma-separated string
        contest['Name'] = contest['Lineup'].str.split(',').str[0]  # dummy for grouping
        raise NotImplementedError("Single-row-per-entry format not yet supported. "
                                  "Use the multi-row (entry, player) export from DK.")
    elif 'EntryId' in contest.columns and 'Name' in contest.columns:
        # Standard DK export format
        pass
    else:
        raise ValueError(f"Unrecognized contest CSV columns: {contest.columns.tolist()}")

    total = total_entries or contest['EntryId'].nunique()
    print(f"[Backtest] Total unique entries: {total}")

    entries = compute_lineup_metrics(contest, proj)

    # Apply payout curve
    entries['Payout'] = entries['Rank'].apply(
        lambda r: payout_for_rank(r, total, buy_in, DEFAULT_PAYOUT)
    )
    entries['Profit'] = entries['Payout'] - buy_in

    # Overall field stats
    print("\n=== Field Summary ===")
    print(f"Entries: {len(entries)}")
    print(f"Buy-in: ${buy_in}")
    print(f"Total field payout: ${entries['Payout'].sum():,.0f}")
    print(f"Mean points in field: {entries['Actual_Points'].mean():.2f}")
    print(f"Median points in field: {entries['Actual_Points'].median():.2f}")
    print(f"Max points in field: {entries['Actual_Points'].max():.2f}")
    print(f"Min cash line (rank at 25%): {entries.nsmallest(int(total * 0.25), 'Rank').iloc[-1]['Actual_Points']:.2f}")

    # Segment entries by how well they aligned with our projections
    entries['proj_percentile'] = entries['Projected_Points'].rank(pct=True)

    print("\n=== Performance by Projection Tier ===")
    print(f"{'Tier':<15} {'N':>6} {'Mean Rank':>10} {'Mean Points':>12} {'Mean Profit':>12} {'ROI':>8}")
    print("-" * 70)

    tiers = [
        ('Top 1% of our proj', entries[entries['proj_percentile'] >= 0.99]),
        ('Top 5%', entries[(entries['proj_percentile'] >= 0.95) & (entries['proj_percentile'] < 0.99)]),
        ('Top 20%', entries[(entries['proj_percentile'] >= 0.80) & (entries['proj_percentile'] < 0.95)]),
        ('Middle 60%', entries[(entries['proj_percentile'] >= 0.20) & (entries['proj_percentile'] < 0.80)]),
        ('Bottom 20%', entries[entries['proj_percentile'] < 0.20]),
    ]
    for label, sub in tiers:
        if sub.empty:
            continue
        roi = sub['Profit'].sum() / (len(sub) * buy_in)
        print(f"{label:<20} {len(sub):>6} {sub['Rank'].mean():>10.0f} "
              f"{sub['Actual_Points'].mean():>12.2f} {sub['Profit'].mean():>+12.2f} {roi:>+8.1%}")

    # Would our top lineup have cashed?
    top_proj_entry = entries.nlargest(1, 'Projected_Points').iloc[0]
    print(f"\n=== Best Projection-Aligned Lineup ===")
    print(f"Projected points: {top_proj_entry['Projected_Points']:.2f}")
    print(f"Actual points: {top_proj_entry['Actual_Points']:.2f}")
    print(f"Rank: {top_proj_entry['Rank']:.0f} / {total}")
    print(f"Payout: ${top_proj_entry['Payout']:.2f}")

    out = contest_csv.replace('.csv', '_backtest.csv')
    entries.to_csv(out, index=False)
    print(f"\n--> Wrote backtest results to {out}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 backtest_contest.py <contest_csv> <projection_csv> [total_entries] [buy_in]")
        sys.exit(1)
    total = int(sys.argv[3]) if len(sys.argv) > 3 else None
    buy_in = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0
    main(sys.argv[1], sys.argv[2], total, buy_in)