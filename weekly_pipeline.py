"""
weekly_pipeline.py

Weekly DFS workflow with auto stack targeting, positional floors, and manual adjustments.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import nflreadpy

sys.path.insert(0, str(Path(__file__).parent))
from dfs_final_model import DKSimulatorDataPipeline


STATS_SEASON = 2025
TARGET_SEASON = 2026
N_LINEUPS_DEFAULT = 20
SALARY_CAP = 50000
SALARY_FLOOR = 48500
MAX_PER_TEAM = 5
MANUAL_ADJUSTMENTS_FILE = 'manual_adjustments.csv'

POSITION_FLOOR_PCT = {
    'QB': 0.35,
    'RB': 0.40,
    'WR': 0.45,
    'TE': 0.40,
    'DST': 0.50,
}


def norm_name(n):
    if pd.isna(n):
        return ''
    return (str(n).replace('.', '').replace("'", '').replace('-', ' ')
            .replace(' Jr', '').replace(' Sr', '').replace(' III', '')
            .lower().strip())


TEAM_NORMALIZE = {'JAC': 'JAX', 'WSH': 'WAS', 'LA': 'LAR', 'OAK': 'LV', 'SD': 'LAC', 'STL': 'LAR'}

def normalize_team(t):
    return TEAM_NORMALIZE.get(t, t)


def infer_slate_teams(salary_path):
    dk = pd.read_csv(salary_path)
    if 'Game Info' not in dk.columns:
        if 'TeamAbbrev' in dk.columns:
            return set(dk['TeamAbbrev'].dropna().unique())
        return None
    teams = set()
    for info in dk['Game Info'].dropna().unique():
        game = str(info).split(' ')[0]
        if '@' in game:
            away, home = game.split('@')
            teams.add(normalize_team(away.strip()))
            teams.add(normalize_team(home.strip()))
    return teams


def get_slate_games(salary_file, week):
    dk = pd.read_csv(salary_file)
    if 'Game Info' not in dk.columns:
        return pd.DataFrame()

    matchups = []
    for info in dk['Game Info'].dropna().unique():
        game_str = str(info).split(' ')[0]
        if '@' not in game_str:
            continue
        away_dk, home_dk = game_str.split('@')
        matchups.append({
            'away_dk': normalize_team(away_dk.strip()),
            'home_dk': normalize_team(home_dk.strip()),
        })

    sched = nflreadpy.load_schedules([TARGET_SEASON]).to_pandas()
    type_col = 'game_type' if 'game_type' in sched.columns else 'season_type'
    sched = sched[(sched['week'] == week) & (sched[type_col] == 'REG')].copy()

    games = []
    for m in matchups:
        away, home = m['away_dk'], m['home_dk']
        row = sched[
            ((sched['home_team'] == home) & (sched['away_team'] == away)) |
            ((sched['home_team'] == away) & (sched['away_team'] == home))
        ]
        if row.empty:
            print(f"[Warning] No schedule row for {away}@{home}")
            continue
        g = row.iloc[0]
        total = g.get('total_line', 44.0)
        total = 44.0 if pd.isna(total) else float(total)
        spread = g.get('spread_line', 0.0)
        spread = 0.0 if pd.isna(spread) else float(spread)

        nfl_home = g['home_team']
        # Positive spread_line in this dataset = home team favored
        if nfl_home == home:
            home_implied = total / 2 + spread / 2
            away_implied = total / 2 - spread / 2
        else:
            home_implied = total / 2 - spread / 2
            away_implied = total / 2 + spread / 2

        games.append({
            'matchup': f"{away}@{home}",
            'total': total, 'spread': spread,
            'home': home, 'away': away,
            'home_implied': home_implied,
            'away_implied': away_implied,
        })

    games_df = pd.DataFrame(games).sort_values('total', ascending=False)

    print("\n" + "=" * 95)
    print("SLATE GAMES — ranked by Vegas total")
    print("=" * 95)
    print(f"{'Matchup':<12} {'Total':>6} {'Spr':>6} {'Home':>5} {'Away':>5} {'H Impl':>7} {'A Impl':>7}")
    for _, r in games_df.iterrows():
        print(f"{r['matchup']:<12} {r['total']:>6.1f} {r['spread']:>+6.1f} "
              f"{r['home']:>5} {r['away']:>5} "
              f"{r['home_implied']:>7.1f} {r['away_implied']:>7.1f}")

    return games_df


def build_shortlist(proj, top_n_per_pos=None):
    if top_n_per_pos is None:
        top_n_per_pos = {'QB': 10, 'RB': 18, 'WR': 24, 'TE': 10, 'DST': 8}
    proj = proj.copy()
    proj['value'] = proj['Fpts'] / (proj['Salary'] / 1000)
    print("\n" + "=" * 95)
    print("SHORTLIST — Top plays by position (sorted by value)")
    print("=" * 95)
    for pos in ['QB', 'RB', 'WR', 'TE', 'DST']:
        sub = proj[proj['Position'] == pos].nlargest(top_n_per_pos[pos], 'value')
        if sub.empty:
            continue
        print(f"\n--- {pos} ---")
        print(f"{'Player':<25} {'Team':<5} {'Opp':<5} {'Fpts':>6} {'Sal':>6} {'Val':>5} {'Own%':>6}")
        for _, r in sub.iterrows():
            print(f"{r['Name']:<25} {r['Team']:<5} {r.get('Opp', '?'):<5} "
                  f"{r['Fpts']:>6.2f} {int(r['Salary']):>6} {r['value']:>5.2f} "
                  f"{r['Own%']:>6.2f}")


def identify_stack_candidates(proj, games_df, top_n=6):
    candidates = []
    for _, g in games_df.iterrows():
        for team, opp, implied in [
            (g['home'], g['away'], g['home_implied']),
            (g['away'], g['home'], g['away_implied']),
        ]:
            team_df = proj[proj['Team'] == team]
            qbs = team_df[team_df['Position'] == 'QB'].nlargest(1, 'Fpts')
            pcs = team_df[team_df['Position'].isin(['WR', 'TE'])].nlargest(3, 'Fpts')
            if qbs.empty or len(pcs) < 2:
                continue
            qb = qbs.iloc[0]
            stack_fpts = qb['Fpts'] + pcs['Fpts'].sum()
            score = 0.5 * stack_fpts + 0.5 * (implied * 2.5)
            candidates.append({
                'team': team, 'opp': opp, 'matchup': g['matchup'],
                'game_total': g['total'], 'team_implied': implied,
                'qb_name': qb['Name'], 'qb_fpts': round(qb['Fpts'], 2),
                'pc_names': pcs['Name'].tolist(), 'pc_fpts': round(pcs['Fpts'].sum(), 2),
                'stack_fpts': round(stack_fpts, 2), 'score': round(score, 2),
            })

    if not candidates:
        return pd.DataFrame()

    cand_df = pd.DataFrame(candidates).sort_values('score', ascending=False).head(top_n)

    print("\n" + "=" * 95)
    print("AUTOMATED STACK CANDIDATES")
    print("=" * 95)
    print(f"{'Team':<5} {'Opp':<5} {'Game':>6} {'Impl':>6} {'QB+PC':>7} {'Score':>7}  Stack")
    for _, r in cand_df.iterrows():
        print(f"{r['team']:<5} {r['opp']:<5} {r['game_total']:>6.1f} {r['team_implied']:>6.1f} "
              f"{r['stack_fpts']:>7.2f} {r['score']:>7.2f}  "
              f"{r['qb_name']} + {', '.join(r['pc_names'][:2])}")

    return cand_df


def load_adjustments(path=MANUAL_ADJUSTMENTS_FILE):
    if not os.path.exists(path):
        return {}, set(), set()
    adj = pd.read_csv(path)
    if 'player_name' not in adj.columns:
        return {}, set(), set()

    multipliers = {}
    force_include = set()
    fade = set()
    for _, r in adj.iterrows():
        name = norm_name(r['player_name'])
        if 'projection_mult' in adj.columns and pd.notna(r['projection_mult']):
            multipliers[name] = float(r['projection_mult'])
        if 'force_include' in adj.columns and str(r.get('force_include', '')).lower() == 'true':
            force_include.add(name)
        if 'fade' in adj.columns and str(r.get('fade', '')).lower() == 'true':
            fade.add(name)

    print(f"[Adjustments] {len(multipliers)} multipliers, "
          f"{len(force_include)} forced, {len(fade)} faded")
    return multipliers, force_include, fade


def apply_pool_filters(proj, salary_exempt=5000):
    """
    Apply position floors and ownership filter. Players with salary >= salary_exempt
    are exempt from BOTH filters (DK pricing says they're real starters).
    """
    print(f"[Floor] Applying positional projection floors (salary >= ${salary_exempt} exempt):")
    before = len(proj)
    frames = []
    for pos in ['QB', 'RB', 'WR', 'TE', 'DST']:
        pos_df = proj[proj['Position'] == pos]
        if pos_df.empty:
            continue
        top_proj = pos_df['Fpts'].max()
        floor_val = top_proj * POSITION_FLOOR_PCT[pos]

        above_floor = pos_df['Fpts'] >= floor_val
        high_salary = pos_df['Salary'] >= salary_exempt
        kept = pos_df[above_floor | high_salary]

        removed = len(pos_df) - len(kept)
        if removed > 0:
            print(f"  {pos}: floor={floor_val:.2f} ({POSITION_FLOOR_PCT[pos]:.0%} of top {top_proj:.2f}), "
                  f"exempt=${salary_exempt}+, removed {removed}")

        saved = pos_df[high_salary & ~above_floor]
        if not saved.empty:
            print(f"    saved by salary: {saved['Name'].tolist()}")

        frames.append(kept)

    proj = pd.concat(frames, ignore_index=True)

    # Ownership filter — exempt high-salary players
    before_own = len(proj)
    low_own = proj['Own%'] < 0.05
    low_salary = proj['Salary'] < salary_exempt
    to_remove = low_own & low_salary
    removed_names = proj[to_remove]['Name'].tolist()
    proj = proj[~to_remove].reset_index(drop=True)
    print(f"[Ownership] Removed {before_own - len(proj)} (low-owned AND under ${salary_exempt})")

    # Report which high-salary players were saved from the ownership filter
    saved_from_own = proj[(proj['Own%'] < 0.05) & (proj['Salary'] >= salary_exempt)]
    if not saved_from_own.empty:
        print(f"    exempt (high salary, low ownership): {saved_from_own['Name'].tolist()[:15]}")

    print(f"[Filter] Pool: {before} -> {len(proj)}")
    print("[Check] Position counts:", proj.groupby('Position').size().to_dict())

    eligible = []
    for team in proj['Team'].unique():
        t = proj[proj['Team'] == team]
        if (t['Position'] == 'QB').any() and len(t[t['Position'].isin(['WR','TE'])]) >= 2:
            eligible.append(team)
    print(f"[Check] {len(eligible)} stack-eligible teams: {sorted(eligible)}")

    return proj


def build_lineups(proj, stack_candidates, force_include_names, fade_names,
                  n_lineups=20, seed=42):
    """Build N lineups with flexible stack construction and bring-backs."""
    rng = np.random.default_rng(seed)
    proj = proj.reset_index(drop=True).copy()
    proj['value'] = proj['Fpts'] / (proj['Salary'] / 1000)

    forced_ids = proj[proj['Name'].map(norm_name).isin(force_include_names)].index.tolist()
    if forced_ids:
        print(f"[Core] Force include: {[proj.loc[i, 'Name'] for i in forced_ids]}")

    stack_options = {}
    for _, c in stack_candidates.iterrows():
        team = c['team']
        opp = c['opp']
        team_df = proj[proj['Team'] == team]
        qbs = team_df[team_df['Position'] == 'QB'].nlargest(1, 'Fpts')
        pcs = team_df[team_df['Position'].isin(['WR', 'TE'])].nlargest(3, 'Fpts')
        if qbs.empty or len(pcs) < 2:
            continue
        stack_options[team] = {
            'qb_id': qbs.index[0],
            'qb_name': qbs.iloc[0]['Name'],
            'pc_ids': pcs.index.tolist(),
            'pc_names': pcs['Name'].tolist(),
            'opp_team': opp,
        }

    if not stack_options:
        print("[Warning] No valid stack options.")
        return [], proj

    print(f"[Stacks] Valid stack teams: {list(stack_options.keys())}")

    def pool(pos):
        return proj[proj['Position'] == pos].sort_values('value', ascending=False)

    qb_pool = pool('QB')
    rb_pool = pool('RB')
    wr_pool = pool('WR')
    te_pool = pool('TE')
    dst_pool = pool('DST')
    flex_pool = pd.concat([rb_pool, wr_pool, te_pool]).sort_values('value', ascending=False)

    # Bring-back pools per opponent team
    def bring_back_pool(opp_team):
        return proj[
            (proj['Team'] == opp_team) &
            (proj['Position'].isin(['RB', 'WR', 'TE']))
        ].sort_values('value', ascending=False)

    def pick_from(pool_df, used_ids, top_n=15):
        sub = pool_df[~pool_df.index.isin(used_ids)]
        if sub.empty:
            return None
        return int(rng.choice(sub.head(top_n).index.values))

    lineups = []
    used_stacks = Counter()
    stack_teams = list(stack_options.keys())
    target_per_stack = max(3, int(np.ceil(n_lineups / len(stack_teams))))

    attempts = 0
    max_attempts = n_lineups * 1000
    #bring_back_count = 0

    while len(lineups) < n_lineups and attempts < max_attempts:
        attempts += 1

        under_used = [t for t in stack_teams if used_stacks[t] < target_per_stack]
        stack_team = rng.choice(under_used if under_used else stack_teams)
        opp_team = stack_options[stack_team]['opp_team']

        slots = {'QB': None, 'RB1': None, 'RB2': None,
                 'WR1': None, 'WR2': None, 'WR3': None,
                 'TE': None, 'FLEX': None, 'DST': None}
        used_ids = set()

        # QB from stack team
        slots['QB'] = int(stack_options[stack_team]['qb_id'])
        used_ids.add(slots['QB'])

        # Optional forced player
        if forced_ids:
            fp = int(rng.choice(forced_ids))
            fpos = proj.loc[fp, 'Position']
            if fpos == 'RB' and slots['RB1'] is None:
                slots['RB1'] = fp; used_ids.add(fp)
            elif fpos == 'WR' and slots['WR1'] is None:
                slots['WR1'] = fp; used_ids.add(fp)
            elif fpos == 'TE' and slots['TE'] is None:
                slots['TE'] = fp; used_ids.add(fp)
            elif fpos == 'DST' and slots['DST'] is None:
                slots['DST'] = fp; used_ids.add(fp)

        # ---- STACK: 1-3 pass-catchers with weighted probabilities ----
        # 1 PC: 35% | 2 PCs: 55% | 3 PCs: 10%
        roll = rng.random()
        if roll < 0.35:
            num_pcs = 1
        elif roll < 0.90:
            num_pcs = 2
        else:
            num_pcs = 3

        avail_pcs = [i for i in stack_options[stack_team]['pc_ids'] if i not in used_ids]
        num_pcs = min(num_pcs, len(avail_pcs))
        if num_pcs > 0:
            for pc in rng.choice(avail_pcs, size=num_pcs, replace=False):
                pc = int(pc)
                if pc in used_ids:
                    continue
                pos = proj.loc[pc, 'Position']
                if pos == 'TE' and slots['TE'] is None:
                    slots['TE'] = pc; used_ids.add(pc)
                elif pos == 'WR':
                    for s in ['WR1', 'WR2', 'WR3']:
                        if slots[s] is None:
                            slots[s] = pc; used_ids.add(pc); break

        # ---- BRING-BACK: opposing team player at 55% frequency ----
        do_bring_back = rng.random() < 0.55
        if do_bring_back:
            bb_pool = bring_back_pool(opp_team)
            bb_pool = bb_pool[~bb_pool.index.isin(used_ids)]
            if not bb_pool.empty:
                # Prefer the FLEX slot; otherwise put them in their natural position slot
                bb_pick = int(rng.choice(bb_pool.head(20).index.values))
                bb_pos = proj.loc[bb_pick, 'Position']
                placed = False
                if slots['FLEX'] is None:
                    slots['FLEX'] = bb_pick
                    used_ids.add(bb_pick)
                    placed = True
                    #bring_back_count += 1
                elif bb_pos == 'WR':
                    for s in ['WR1', 'WR2', 'WR3']:
                        if slots[s] is None:
                            slots[s] = bb_pick; used_ids.add(bb_pick); placed = True
                            #bring_back_count += 1
                            break
                elif bb_pos == 'TE' and slots['TE'] is None:
                    slots['TE'] = bb_pick; used_ids.add(bb_pick); placed = True
                    #bring_back_count += 1
                elif bb_pos == 'RB':
                    for s in ['RB1', 'RB2']:
                        if slots[s] is None:
                            slots[s] = bb_pick; used_ids.add(bb_pick); placed = True
                            #bring_back_count += 1
                            break
                # If no slot fits, skip the bring-back for this lineup

        # ---- Fill remaining slots ----
        for s in ['RB1', 'RB2']:
            if slots[s] is None:
                pick = pick_from(rb_pool, used_ids)
                if pick is not None:
                    slots[s] = pick; used_ids.add(pick)

        for s in ['WR1', 'WR2', 'WR3']:
            if slots[s] is None:
                pick = pick_from(wr_pool, used_ids)
                if pick is not None:
                    slots[s] = pick; used_ids.add(pick)

        if slots['TE'] is None:
            pick = pick_from(te_pool, used_ids)
            if pick is not None:
                slots['TE'] = pick; used_ids.add(pick)

        if slots['FLEX'] is None:
            pick = pick_from(flex_pool, used_ids, top_n=40)
            if pick is not None:
                slots['FLEX'] = pick; used_ids.add(pick)

        if slots['DST'] is None:
            pick = pick_from(dst_pool, used_ids, top_n=12)
            if pick is not None:
                slots['DST'] = pick; used_ids.add(pick)

        # ---- Validate ----
        slot_values = list(slots.values())
        if None in slot_values or len(set(slot_values)) != 9:
            continue

        checks = [
            proj.loc[slots['QB'], 'Position'] == 'QB',
            proj.loc[slots['RB1'], 'Position'] == 'RB',
            proj.loc[slots['RB2'], 'Position'] == 'RB',
            proj.loc[slots['WR1'], 'Position'] == 'WR',
            proj.loc[slots['WR2'], 'Position'] == 'WR',
            proj.loc[slots['WR3'], 'Position'] == 'WR',
            proj.loc[slots['TE'], 'Position'] == 'TE',
            proj.loc[slots['DST'], 'Position'] == 'DST',
        ]
        if not all(checks):
            continue

        salary = int(proj.loc[slot_values, 'Salary'].sum())
        if salary > SALARY_CAP or salary < SALARY_FLOOR:
            continue

        team_counts = Counter(proj.loc[i, 'Team'] for i in slot_values)
        if any(c > MAX_PER_TEAM for c in team_counts.values()):
            continue

        lineups.append(slots)
        used_stacks[stack_team] += 1

    print(f"\nBuilt {len(lineups)} lineups in {attempts} attempts")
    print(f"Stack distribution: {dict(used_stacks)}")
    #print(f"Bring-backs included: {bring_back_count} ({100*bring_back_count/max(len(lineups),1):.0f}% of lineups)")
    # Count bring-backs in the final validated lineups
    final_bb_count = 0
    for slots in lineups:
        stack_team_name = proj.loc[slots['QB'], 'Team']
        opp_team_name = stack_options[stack_team_name]['opp_team']
        for slot_name in ['FLEX', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE']:
            player_id = slots.get(slot_name)
            if player_id is not None and proj.loc[player_id, 'Team'] == opp_team_name:
                final_bb_count += 1
                break

    print(f"Bring-backs in final lineups: {final_bb_count} ({100*final_bb_count/max(len(lineups),1):.0f}% of lineups)")
    return lineups, proj


def lineups_to_df(proj, lineups):
    rows = []
    slot_order = ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE', 'FLEX', 'DST']
    for i, slots in enumerate(lineups):
        row = {'lineup_idx': i}
        for slot in slot_order:
            player_id = slots[slot]
            row[slot] = proj.loc[player_id, 'Name']
            row[f'{slot}_pos'] = proj.loc[player_id, 'Position']
        ids = list(slots.values())
        row['salary'] = int(proj.loc[ids, 'Salary'].sum())
        row['fpts_proj'] = round(float(proj.loc[ids, 'Fpts'].sum()), 2)
        row['stack_team'] = proj.loc[slots['QB'], 'Team']
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--week', type=int, required=True)
    parser.add_argument('--salary-file', required=True)
    parser.add_argument('--n-lineups', type=int, default=N_LINEUPS_DEFAULT)
    args = parser.parse_args()

    week = args.week
    out_dir = Path(f"output/week_{week}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== WEEK {week} WORKFLOW ===")
    print(f"Output directory: {out_dir}\n")

    # Step 1: Projections
    print("=" * 95)
    print("STEP 1: Running projection pipeline")
    print("=" * 95)
    model = DKSimulatorDataPipeline(
        stats_season=STATS_SEASON, target_season=TARGET_SEASON,
        target_week=week, dk_salary_csv=args.salary_file,
    )
    model.run_gpp_optimized_pipeline(injuries={})
    proj_path = f"sim_input_projections_{TARGET_SEASON}_w{week}.csv"

    # Step 2: DSTs
    print("\n" + "=" * 95)
    print("STEP 2: Generating DST projections")
    print("=" * 95)
    subprocess.run([sys.executable, 'generate_dsts.py', proj_path, str(TARGET_SEASON), str(week)], check=True)

    # Step 3: Load and filter to slate
    print("\n" + "=" * 95)
    print("STEP 3: Filtering to slate")
    print("=" * 95)
    proj = pd.read_csv(proj_path)
    proj = proj.rename(columns={'Player': 'Name', 'Projection': 'Fpts',
                                'Ownership': 'Own%', 'Opponent': 'Opp'})
    if 'Opp' not in proj.columns:
        proj['Opp'] = ''
    if proj['Own%'].max() <= 1.0:
        proj['Own%'] = proj['Own%'] * 100

    slate_teams = infer_slate_teams(args.salary_file)
    if slate_teams:
        proj = proj[proj['Team'].isin(slate_teams)].reset_index(drop=True)
        print(f"Players on slate: {len(proj)}")

    # Step 4: Adjustments
    multipliers, force_include, fade = load_adjustments(MANUAL_ADJUSTMENTS_FILE)
    if multipliers:
        for idx, row in proj.iterrows():
            key = norm_name(row['Name'])
            if key in multipliers:
                mult = multipliers[key]
                proj.at[idx, 'Fpts'] = round(row['Fpts'] * mult, 2)
                if 'Ceiling' in proj.columns:
                    proj.at[idx, 'Ceiling'] = round(row['Ceiling'] * mult, 2)

    # Step 4b: Filter
    print("\n" + "=" * 95)
    print("STEP 4b: Filtering player pool")
    print("=" * 95)
    if fade:
        before = len(proj)
        proj = proj[~proj['Name'].map(norm_name).isin(fade)].reset_index(drop=True)
        print(f"[Fade] Removed {before - len(proj)} players")
    proj = apply_pool_filters(proj)

    # Step 5: Shortlist
    build_shortlist(proj)

    # Step 6: Slate games
    games_df = get_slate_games(args.salary_file, week)

    # Step 7: Stack candidates
    stack_candidates = identify_stack_candidates(proj, games_df, top_n=8)
    if stack_candidates.empty:
        print("[ERROR] No stack candidates found.")
        return

    # Step 8: Build lineups
    print("\n" + "=" * 95)
    print(f"STEP 8: Building {args.n_lineups} lineups")
    print("=" * 95)
    lineups, proj_used = build_lineups(
        proj, stack_candidates,
        force_include_names=force_include,
        fade_names=set(),  # already applied
        n_lineups=args.n_lineups,
    )
    if not lineups:
        print("[ERROR] No lineups built.")
        return

    lineups_df = lineups_to_df(proj_used, lineups)
    out_path = out_dir / f"lineups_w{week}.csv"
    lineups_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(lineups)} lineups to {out_path}")

    print("\nSample lineups:")
    slot_order = ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE', 'FLEX', 'DST']
    for i in range(min(3, len(lineups_df))):
        row = lineups_df.iloc[i]
        print(f"\nLineup {i} (stack: {row['stack_team']}, sal: ${row['salary']}, proj: {row['fpts_proj']}):")
        for slot in slot_order:
            print(f"  {slot:<4} {row[slot]:<25} ({row[f'{slot}_pos']})")

    print(f"\nSalary check: min=${lineups_df['salary'].min()}, max=${lineups_df['salary'].max()}, over-cap={sum(lineups_df['salary'] > 50000)}")
    print(f"Stack distribution: {lineups_df['stack_team'].value_counts().to_dict()}")


if __name__ == '__main__':
    main()