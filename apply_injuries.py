"""
apply_injuries.py

Reads injury_flags.csv and the current pipeline projections, applies
status-based multipliers, and writes injury_overrides.csv that can be
appended to manual_adjustments.csv.

Usage:
    python3 apply_injuries.py --proj sim_input_projections_2026_w3.csv \
                              --injuries injury_flags.csv \
                              --out injury_overrides.csv
"""
import argparse
import pandas as pd


def norm_name(n):
    if pd.isna(n):
        return ''
    return (str(n).replace('.', '').replace("'", '').replace('-', ' ')
            .replace(' Jr', '').replace(' Sr', '').replace(' III', '')
            .lower().strip())


# Multiplier per status
STATUS_MULTIPLIERS = {
    'OUT':          0.0,
    'IR':           0.0,
    'PUP':          0.0,
    'SUSPENDED':    0.0,
    'DNP':          0.60,    # did not practice all week — significant concern
    'DOUBTFUL':     0.20,
    'QUESTIONABLE': 0.75,
    'LIMITED':      0.85,    # limited in practice — minor concern
    'PROBABLE':     0.95,
    'FULL':         1.0,     # full practice — no adjustment
}

def status_multiplier(status):
    """Case-insensitive lookup with fallback."""
    if pd.isna(status):
        return 1.0
    s = str(status).strip().upper()
    if s in STATUS_MULTIPLIERS:
        return STATUS_MULTIPLIERS[s]
    for key, mult in STATUS_MULTIPLIERS.items():
        if key in s:
            return mult
    return 1.0


def is_real_injury(row):
    """True if the injury status reflects an actual injury (not rest/personal)."""
    practice_inj = str(row.get('practice_injury', '')).lower()
    report_inj = str(row.get('report_injury', '')).lower()
    combined = practice_inj + ' ' + report_inj
    if 'not injury related' in combined:
        return False
    if 'resting' in combined:
        return False
    if 'personal' in combined:
        return False
    return True


def main(proj_path, injuries_path, out_path):
    proj = pd.read_csv(proj_path)
    injuries = pd.read_csv(injuries_path)

    proj['_key'] = proj['Player'].map(norm_name)
    injuries['_key'] = injuries['player_name'].map(norm_name)

    # Join
    merged = pd.merge(proj[['Player', 'Position', 'Team', 'Projection', '_key']],
                      injuries[['_key', 'status', 'team', 'report_injury', 'practice_injury']],
                      on='_key', how='inner')

    print(f"[Apply] Matched {len(merged)} injured players to projections")

    # Filter out non-injury statuses (rest days, personal matters)
    before = len(merged)
    merged = merged[merged.apply(is_real_injury, axis=1)].copy()
    print(f"[Apply] Filtered to {len(merged)} actual injuries (removed {before - len(merged)} non-injury)")

    # Compute multiplier per row
    merged['mult'] = merged['status'].apply(status_multiplier)
    affected = merged[merged['mult'] < 1.0].copy()

    # Report
    print(f"\n[Apply] {len(affected)} players will be adjusted:")
    for _, r in affected.sort_values('mult').iterrows():
        print(f"  {r['Player']:<25} {r['Position']:<3} "
              f"{r['status']:<12} mult={r['mult']:.2f} proj={r['Projection']:.2f}")

    # Write override CSV in the manual_adjustments.csv schema
    overrides = pd.DataFrame({
        'player_name': affected['Player'],
        'projection_mult': affected['mult'],
        'ceiling_mult': affected['mult'],
        'force_include': 'false',
        'fade': affected['status'].isin(['OUT', 'IR', 'PUP', 'SUSPENDED']).astype(str).str.lower(),
        'notes': affected['status'] + ' (auto-detected from ESPN)',
    })
    overrides.to_csv(out_path, index=False)
    print(f"\n[Apply] Wrote {len(overrides)} override rows to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--proj', required=True)
    parser.add_argument('--injuries', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    main(args.proj, args.injuries, args.out)