"""
injury_scraper.py

Load NFL injury reports via nflreadpy (nflverse data).

Usage:
    python3 injury_scraper.py --season 2026 --week 2 --out injury_flags.csv
"""
import argparse
import pandas as pd
import nflreadpy


def normalize_status(report_status, practice_status):
    """Combine report and practice status into a single canonical status."""
    if pd.notna(report_status):
        s = str(report_status).strip().lower()
        if 'out' in s and 'doubtful' not in s:
            return 'OUT'
        if 'doubtful' in s:
            return 'DOUBTFUL'
        if 'questionable' in s or 'quest' in s:
            return 'QUESTIONABLE'
        if 'probable' in s:
            return 'PROBABLE'
        if 'ir' in s or 'injured reserve' in s:
            return 'IR'
        if 'suspend' in s:
            return 'SUSPENDED'

    # Fallback to practice status
    if pd.notna(practice_status):
        s = str(practice_status).strip().lower()
        if 'did not' in s or 'dnp' in s:
            return 'DNP'
        if 'limited' in s:
            return 'LIMITED'
        if 'full' in s:
            return 'FULL'

    return 'Unknown'


def build_injury_flags(season, week, out_path):
    try:
        injuries = nflreadpy.load_injuries([season])
    except Exception as e:
        print(f"[Injury] load_injuries failed: {e}")
        return pd.DataFrame()

    df = injuries.to_pandas() if hasattr(injuries, "to_pandas") else pd.DataFrame(injuries)
    if df.empty:
        print(f"[Injury] No data for season {season}")
        return pd.DataFrame()

    print(f"[Injury] Loaded {len(df)} records for {season}")

    # Filter to the target week
    if 'week' in df.columns:
        df = df[df['week'] == week].copy()
        print(f"[Injury] {len(df)} records for week {week}")

    if df.empty:
        print(f"[Injury] No records for week {week}. Try a different week.")
        return pd.DataFrame()

    # Build output
    out = pd.DataFrame()
    out['player_name'] = df['full_name'].astype(str).str.strip()
    out['status'] = df.apply(
        lambda r: normalize_status(r.get('report_status'), r.get('practice_status')), axis=1
    )
    out['team'] = df['team'].astype(str).str.strip() if 'team' in df.columns else ''
    out['position'] = df['position'].astype(str).str.strip() if 'position' in df.columns else ''
    out['report_injury'] = df.get('report_primary_injury', '').fillna('')
    out['practice_injury'] = df.get('practice_primary_injury', '').fillna('')

    # Dedupe: keep the worst status per player
    status_order = {'OUT': 0, 'IR': 0, 'SUSPENDED': 0, 'DOUBTFUL': 1,
                    'DNP': 2, 'QUESTIONABLE': 3, 'LIMITED': 4,
                    'PROBABLE': 5, 'FULL': 6, 'Unknown': 7}
    out['_order'] = out['status'].map(status_order).fillna(7)
    out = out.sort_values(['player_name', '_order']).drop_duplicates('player_name', keep='first')
    out = out.drop(columns=['_order'])

    out.to_csv(out_path, index=False)
    print(f"\n[Injury] Wrote {len(out)} records to {out_path}")
    print(f"[Injury] Status breakdown: {out['status'].value_counts().to_dict()}")

    # Show skill-position players
    skill = out[out['position'].isin(['QB', 'RB', 'WR', 'TE'])]
    print(f"\n[Injury] Skill-position players ({len(skill)} total):")
    for _, r in skill.head(20).iterrows():
        inj = r['report_injury'] or r['practice_injury'] or ''
        print(f"  {r['player_name']:<25} {r['position']:<3} {r['team']:<4} "
              f"{r['status']:<12} {inj[:30]}")

    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', type=int, default=2026)
    parser.add_argument('--week', type=int, default=2)
    parser.add_argument('--out', default='injury_flags.csv')
    args = parser.parse_args()
    build_injury_flags(args.season, args.week, args.out)