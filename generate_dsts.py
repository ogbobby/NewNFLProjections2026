"""
generate_dsts.py

Add DST projections to a pipeline output CSV. DSTs aren't modeled by the main
pipeline; this script generates them from Vegas totals.

Usage:
    python3 generate_dsts.py <projections_csv> <season> <week>
"""
import sys
import pandas as pd
import numpy as np
import nflreadpy


TEAM_TO_NICK = {
    'ARI': 'Cardinals', 'ATL': 'Falcons', 'BAL': 'Ravens', 'BUF': 'Bills',
    'CAR': 'Panthers', 'CHI': 'Bears', 'CIN': 'Bengals', 'CLE': 'Browns',
    'DAL': 'Cowboys', 'DEN': 'Broncos', 'DET': 'Lions', 'GB': 'Packers',
    'HOU': 'Texans', 'IND': 'Colts', 'JAX': 'Jaguars', 'KC': 'Chiefs',
    'LAC': 'Chargers', 'LAR': 'Rams', 'LV': 'Raiders', 'MIA': 'Dolphins',
    'MIN': 'Vikings', 'NE': 'Patriots', 'NO': 'Saints', 'NYG': 'Giants',
    'NYJ': 'Jets', 'PHI': 'Eagles', 'PIT': 'Steelers', 'SEA': 'Seahawks',
    'SF': '49ers', 'TB': 'Buccaneers', 'TEN': 'Titans', 'WAS': 'Commanders',
}


def build_dsts(season, week):
    sched = nflreadpy.load_schedules([season]).to_pandas()
    type_col = 'game_type' if 'game_type' in sched.columns else 'season_type'
    sched = sched[(sched['week'] == week) & (sched[type_col] == 'REG')].copy()

    if sched.empty:
        raise RuntimeError(f"No {season} week {week} games in schedule")

    rows = []
    for _, g in sched.iterrows():
        total = g.get('total_line', 44.0)
        total = 44.0 if pd.isna(total) else total
        spread = g.get('spread_line', 0.0)
        spread = 0.0 if pd.isna(spread) else spread
        home_implied = total / 2 + spread / 2
        away_implied = total / 2 - spread / 2

        for team, opp, opp_implied in [
            (g['home_team'], g['away_team'], away_implied),
            (g['away_team'], g['home_team'], home_implied),
        ]:
            # DST scales inversely with the opposing offense's strength
            raw_proj = 12.5 - 0.32 * opp_implied
            if team == g['home_team']:
                raw_proj += 0.5
            raw_proj = max(2.0, min(raw_proj, 14.0))
            rows.append({
                'team': team,
                'opp': opp,
                'raw_proj': raw_proj,
            })

    dst = pd.DataFrame(rows)
    dst['salary'] = (2500 + 1800 * (dst['raw_proj'] - dst['raw_proj'].min()) /
                     max(dst['raw_proj'].max() - dst['raw_proj'].min(), 0.01)).round(-1).astype(int)
    dst['salary'] = dst['salary'].clip(2500, 4300)
    dst['ownership'] = (0.01 + 0.05 * (dst['raw_proj'] - dst['raw_proj'].min()) /
                        max(dst['raw_proj'].max() - dst['raw_proj'].min(), 0.01)).round(4)
    dst['stddev'] = (dst['raw_proj'] * 0.55).round(2)
    dst['ceiling'] = (dst['raw_proj'] * 1.85).round(2)
    dst['name'] = dst['team'].map(TEAM_TO_NICK).fillna(dst['team'] + ' DST')
    return dst


def main():
    if len(sys.argv) < 4:
        print("Usage: python3 generate_dsts.py <projections_csv> <season> <week>")
        sys.exit(1)

    proj_path = sys.argv[1]
    season = int(sys.argv[2])
    week = int(sys.argv[3])

    proj = pd.read_csv(proj_path)
    print(f"[DST] Loaded {len(proj)} players from {proj_path}")
    print(f"  Existing positions: {proj['Position'].value_counts().to_dict()}")

    # Drop any existing DST rows to avoid duplicates
    proj = proj[proj['Position'] != 'DST'].copy()

    dst = build_dsts(season, week)
    print(f"[DST] Generated {len(dst)} DST projections")
    print(dst[['name', 'team', 'opp', 'raw_proj', 'salary', 'ownership']].to_string(index=False))

    # Match the pipeline CSV schema
    dst_rows = pd.DataFrame({
        'Player': dst['name'],
        'Position': 'DST',
        'Team': dst['team'],
        'Opponent': dst['opp'],
        'Projection': dst['raw_proj'].round(2),
        'Ceiling': dst['ceiling'],
        'Ceiling_Mult': 1.85,
        'Usage_Stability': 0.75,
        'Ceiling_Ratio': 1.85,
        'Leverage': 0.5,
        'StdDev': dst['stddev'],
        'Salary': dst['salary'],
        'AvgPointsPerGame': dst['raw_proj'].round(1),
        'Ownership': dst['ownership'],
    })

    combined = pd.concat([proj, dst_rows], ignore_index=True)
    combined.to_csv(proj_path, index=False)
    print(f"\n[DST] Wrote {len(combined)} rows back to {proj_path}")
    print(f"[DST] Final positions: {combined['Position'].value_counts().to_dict()}")


if __name__ == '__main__':
    main()