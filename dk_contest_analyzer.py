"""
dk_contest_analyzer.py

Parse a DK contest export. Handles the multi-row format where each entry
appears as 9 rows (one per roster slot):
  Rank, EntryId, EntryName, TimeRemaining, Points, Lineup, Unnamed: 6,
  Player, Roster Position, %Drafted, FPTS

Produces:
  1. Field summary (size, cash lines, top scores)
  2. Top lineups (grouped by EntryId)
  3. Your lineups' results
  4. last_week_results.csv: player -> actual DK points for the pipeline EWMA

Usage:
    python3 dk_contest_analyzer.py <contest_csv> [--username NAME]
                                                 [--season YYYY --week N]
"""
import argparse
import sys
from collections import Counter
import pandas as pd
import numpy as np


def normalize_name(n):
    if pd.isna(n):
        return ''
    return (str(n).replace('.', '').replace("'", '').replace('-', ' ')
            .replace(' Jr', '').replace(' Sr', '').replace(' III', '')
            .lower().strip())


def load_contest(path):
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    print(f"[DK] Loaded {len(df):,} rows, columns: {df.columns.tolist()}")

    # Coerce numeric columns
    for c in ['Rank', 'Points', 'FPTS', '%Drafted']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    return df


def field_summary(df):
    print(f"\n=== Field Summary ===")

    # Count unique entries
    if 'EntryId' in df.columns:
        entries = df.drop_duplicates('EntryId')[['EntryId', 'Rank', 'Points']].copy()
    else:
        entries = df[['Rank', 'Points']].drop_duplicates().copy()

    n = len(entries)
    print(f"Field size: {n:,} entries")
    print(f"Max score: {entries['Points'].max():.2f}")
    print(f"Median score: {entries['Points'].median():.2f}")
    print(f"Mean score: {entries['Points'].mean():.2f}")
    print(f"Min score: {entries['Points'].min():.2f}")

    # Cash lines at common GPP cutoffs
    for pct in [0.01, 0.05, 0.10, 0.20, 0.25]:
        cutoff = int(n * pct)
        if cutoff < 1:
            continue
        cash_line = entries.nsmallest(cutoff, 'Rank').iloc[-1]['Points']
        print(f"Score at top {int(pct*100):>2}%: {cash_line:.2f}")

    return entries


def show_top_lineups(df, entries, n=10):
    print(f"\n=== Top {n} Lineups ===")
    top_ids = entries.nsmallest(n, 'Rank')['EntryId'].tolist()
    for entry_id in top_ids:
        lu = df[df['EntryId'] == entry_id]
        if lu.empty:
            continue
        rank = lu['Rank'].iloc[0]
        points = lu['Points'].iloc[0]
        entry_name = lu['EntryName'].iloc[0] if 'EntryName' in lu.columns else 'N/A'
        players = lu.sort_values('Roster Position')['Player'].tolist()
        print(f"\n  Rank {int(rank):,}: {points:.2f} pts ({entry_name})")
        for _, r in lu.iterrows():
            drafted = r.get('%Drafted', 'N/A')
            fpts = r.get('FPTS', 'N/A')
            print(f"    {r['Roster Position']:<4} {r['Player']:<25} "
                  f"owned {drafted}%  scored {fpts}")


def show_your_lineups(df, username):
    print(f"\n=== Your Lineups ({username}) ===")
    if 'EntryName' not in df.columns:
        print("  No EntryName column. Available:", df.columns.tolist())
        return

    mask = df['EntryName'].fillna('').str.lower().str.contains(username.lower())
    your_entries = df[mask]['EntryId'].unique()

    if len(your_entries) == 0:
        print(f"  No entries found matching '{username}'")
        return

    # Field size for percentile calculation
    field_size = df['EntryId'].nunique()

    for entry_id in your_entries:
        lu = df[df['EntryId'] == entry_id]
        rank = int(lu['Rank'].iloc[0])
        points = lu['Points'].iloc[0]
        top_pct = 100 * rank / field_size
        players = lu.sort_values('Roster Position')['Player'].tolist()
        print(f"\n  Rank {rank:,} / {field_size:,}  (top {top_pct:.1f}%)  —  {points:.2f} pts")
        for _, r in lu.iterrows():
            print(f"    {r['Roster Position']:<4} {r['Player']}")


def player_ownership_and_scores(df):
    """
    Aggregate player-level data across all entries.

    Returns a DataFrame with:
      player_name, position, avg_ownership_pct, avg_fpts, appearances
    Because each entry lists every player once, '%Drafted' is the same across
    all rows for the same player. FPTS is the same too. So we just take the
    first row per player.
    """
    if 'Player' not in df.columns:
        return pd.DataFrame()

    players = df.drop_duplicates(subset='Player')[[
        'Player', 'Roster Position', '%Drafted', 'FPTS'
    ]].copy()
    players.columns = ['player_name', 'position', 'ownership_pct', 'actual_dk']
    players = players.sort_values('actual_dk', ascending=False)
    return players


def build_ewma_results(df, out_path='last_week_results.csv'):
    """
    Write a player-level file with actual DK points. The pipeline uses this
    for its EWMA blend.
    """
    players = player_ownership_and_scores(df)
    if players.empty:
        print("[EWMA] No player data — skipping last_week_results.csv")
        return
    players[['player_name', 'actual_dk']].to_csv(out_path, index=False)
    print(f"\n[EWMA] Wrote {out_path} ({len(players)} players)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('contest_csv')
    parser.add_argument('--username', default=None)
    parser.add_argument('--top-lineups', type=int, default=10)
    parser.add_argument('--save-ewma', action='store_true',
                        help='Write last_week_results.csv for the pipeline')
    args = parser.parse_args()

    df = load_contest(args.contest_csv)
    entries = field_summary(df)
    show_top_lineups(df, entries, n=args.top_lineups)

    if args.username:
        show_your_lineups(df, args.username)

    # Player ownership + actual scores
    print(f"\n=== Top Players by Ownership ===")
    players = player_ownership_and_scores(df)
    if not players.empty:
        top_owned = players.nlargest(15, 'ownership_pct')
        print(top_owned.to_string(index=False))

        print(f"\n=== Top Scorers ===")
        top_scores = players.nlargest(15, 'actual_dk')
        print(top_scores.to_string(index=False))

    if args.save_ewma:
        build_ewma_results(df)


if __name__ == '__main__':
    main()