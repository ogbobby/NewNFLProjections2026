"""
convert_to_sim_input.py

Converts projection CSV from dfs_final_model.py into NFL_Sim format,
reconciling player names against DK's player_ids.csv.

Usage:
    python3 convert_to_sim_input.py <projection_csv> <dest_csv> [player_ids_csv]

If player_ids_csv is provided, names are reconciled against DK's canonical
names before writing. Otherwise, names pass through unchanged.
"""
import sys
import os
import re
import pandas as pd


def normalize_name(n):
    """Aggressive normalization for matching across sources."""
    if pd.isna(n):
        return ''
    s = str(n)
    # Lowercase
    s = s.lower()
    # Remove suffixes
    for suf in [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv', ' v']:
        s = s.replace(suf, '')
    # Remove punctuation
    s = re.sub(r"[.'\-]", '', s)
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def build_name_lookup(player_ids_path):
    """Return dict {normalized_name: canonical_dk_name}."""
    dk = pd.read_csv(player_ids_path)
    # DK export column is usually "Name" — may be "Name + ID" in some versions
    name_col = 'Name' if 'Name' in dk.columns else None
    if name_col is None:
        for c in dk.columns:
            if 'name' in c.lower() and 'id' not in c.lower():
                name_col = c
                break
    if name_col is None:
        print(f"[Convert] Could not find Name column in {player_ids_path}")
        print(f"[Convert] Columns: {dk.columns.tolist()}")
        return {}

    lookup = {}
    for n in dk[name_col].dropna().unique():
        lookup[normalize_name(n)] = n
    print(f"[Convert] Built name lookup with {len(lookup)} entries from DK export.")
    return lookup


def convert(src_path, dst_path, player_ids_path=None):
    print(f"[Convert] Reading {src_path}")
    df = pd.read_csv(src_path)

    required = ['Player', 'Position', 'Team', 'Salary', 'Projection', 'Ownership', 'StdDev']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required source columns: {missing}")

    # --- Name reconciliation ---
    if player_ids_path and os.path.exists(player_ids_path):
        lookup = build_name_lookup(player_ids_path)
        if lookup:
            before = df['Player'].nunique()
            mapped = df['Player'].map(lambda n: lookup.get(normalize_name(n), n))
            n_reconciled = (mapped != df['Player']).sum()
            df['Player'] = mapped
            print(f"[Convert] Reconciled {n_reconciled} player names to DK canonical form.")

            # Report any that still don't match
            unmatched = df[~df['Player'].map(lambda n: normalize_name(n)).isin(lookup.keys())]
            if len(unmatched) > 0:
                print(f"[Convert] {len(unmatched)} names still unmatched "
                      f"(sample: {unmatched['Player'].head(5).tolist()})")
    else:
        print(f"[Convert] No player_ids.csv provided — writing names as-is.")

    out = pd.DataFrame()
    out['Name']     = df['Player']
    out['Position'] = df['Position']
    out['Team']     = df['Team']
    out['Salary']   = df['Salary'].astype(int)
    out['Fpts']     = df['Projection'].round(2)
    out['Own%']     = (df['Ownership'] * 100).round(2)
    out['StdDev']   = df['StdDev'].round(2)

    if 'Ceiling' in df.columns:
        out['Ceiling'] = df['Ceiling'].round(2)
    if 'AvgPointsPerGame' in df.columns:
        out['Field Fpts'] = df['AvgPointsPerGame'].round(2)

    out = out.sort_values('Fpts', ascending=False).reset_index(drop=True)
    out.to_csv(dst_path, index=False)
    print(f"[Convert] Wrote {len(out)} players to {dst_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 convert_to_sim_input.py <projection_csv> <dest_csv> [player_ids_csv]")
        sys.exit(1)
    src = sys.argv[1]
    dst = sys.argv[2]
    pids = sys.argv[3] if len(sys.argv) > 3 else None
    convert(src, dst, pids)