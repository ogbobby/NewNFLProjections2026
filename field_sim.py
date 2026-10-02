"""
field_sim.py

Realistic NFL DFS field simulator and lineup optimizer.

Usage:
    python3 field_sim.py generate --proj projections.csv --field-size 50000 --out field.csv
    python3 field_sim.py optimize --proj projections.csv --num-lineups 20 --slate-file DKSalaries.csv --out my_lineups.csv
    python3 field_sim.py grade --proj projections.csv --field field.csv --lineups my_lineups.csv --iterations 5000
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


# ============================================================
# Constants
# ============================================================
DK_SALARY_CAP = 50000
DK_MIN_SALARY = 48000
FIELD_MAX_PER_TEAM = 4

POOL_SIZES = {
    'QB': 24, 'RB': 40, 'WR': 65, 'TE': 20, 'DST': 20,
}

# DST projections rescaled to this range
DST_PROJ_MIN = 3.0
DST_PROJ_MAX = 11.0
DST_SALARY_MIN = 2500
DST_SALARY_MAX = 4300


# ============================================================
# Name normalization
# ============================================================
def norm_name(n):
    if pd.isna(n):
        return ''
    return (str(n).replace('.', '').replace("'", '').replace('-', ' ')
            .replace(' Jr', '').replace(' Sr', '').replace(' III', '')
            .lower().strip())


# ============================================================
# Slate filtering
# ============================================================
def infer_slate_teams(dk_salary_path):
    dk = pd.read_csv(dk_salary_path)
    if 'Game Info' not in dk.columns:
        return None
    teams = set()
    for info in dk['Game Info'].dropna().unique():
        game = str(info).split(' ')[0]
        if '@' in game:
            away, home = game.split('@')
            teams.add(away.strip())
            teams.add(home.strip())
    return teams


# ============================================================
# Load projections with all fixes applied
# ============================================================
def load_projections(path, pool_size_mult=1.0, only_teams=None, top_pct=None):
    df = pd.read_csv(path)

    # Normalize column names — accept the pipeline schema or the field_sim schema
    column_aliases = {
        'Player': 'Name',
        'Projection': 'Fpts',
        'Ownership': 'Own%',
        'Opponent': 'Opp',
        'AvgPointsPerGame': 'AvgPPG',
    }
    for old, new in column_aliases.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # Ownership scale — pipeline writes fractions (0.15), field_sim may expect percent (15.0)
    if 'Own%' in df.columns and df['Own%'].max() <= 1.0:
        df['Own%'] = df['Own%'] * 100

    required = ['Name', 'Position', 'Team', 'Salary', 'Fpts', 'StdDev']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Projections CSV missing: {missing}\nAvailable: {df.columns.tolist()}")

    # Own% is optional — fill with a default if missing
    if 'Own%' not in df.columns:
        df['Own%'] = 5.0  # neutral default

    df = df[df['Position'].isin(['QB', 'RB', 'WR', 'TE', 'DST'])].copy()
    df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce').fillna(0).astype(int)
    df['Fpts'] = pd.to_numeric(df['Fpts'], errors='coerce').fillna(0)
    df['StdDev'] = pd.to_numeric(df['StdDev'], errors='coerce').fillna(5.0)
    df['Own%'] = pd.to_numeric(df['Own%'], errors='coerce').fillna(0)

    # DST projection rescale (if compressed)
    dst_mask = df['Position'] == 'DST'
    if dst_mask.any():
        dst_df = df[dst_mask]
        old_min, old_max = dst_df['Fpts'].min(), dst_df['Fpts'].max()
        if old_max - old_min > 0 and old_max < 10.0:
            df.loc[dst_mask, 'Fpts'] = (
                3.0 + (dst_df['Fpts'] - old_min) * (11.0 - 3.0) / (old_max - old_min)
            ).round(2)

    if only_teams:
        df = df[df['Team'].isin(only_teams)].copy()

    df = df[df['Salary'] > 0].copy()
    df['_key'] = df['Name'].map(norm_name)
    df = df.drop_duplicates('_key').reset_index(drop=True)

    return df
#def load_projections(path, pool_size_mult=1.0, only_teams=None, top_pct=None):
#    df = pd.read_csv(path)
#    required = ['Name', 'Position', 'Team', 'Salary', 'Fpts', 'Own%', 'StdDev']
#    missing = [c for c in required if c not in df.columns]
#    if missing:
#        raise ValueError(f"Projections CSV missing: {missing}")
#
#    df = df[df['Position'].isin(['QB', 'RB', 'WR', 'TE', 'DST'])].copy()
#    df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce').fillna(0).astype(int)
#    df['Fpts'] = pd.to_numeric(df['Fpts'], errors='coerce').fillna(0)
#    df['StdDev'] = pd.to_numeric(df['StdDev'], errors='coerce').fillna(5.0)
#    df['Own%'] = pd.to_numeric(df['Own%'], errors='coerce').fillna(0)
#
#    # --- FIX 1: Rescale DST projections if compressed ---
#    dst_mask = df['Position'] == 'DST'
#    if dst_mask.any():
#        dst_df = df[dst_mask]
#        old_min, old_max = dst_df['Fpts'].min(), dst_df['Fpts'].max()
#        if old_max - old_min > 0 and old_max < DST_PROJ_MAX - 1.0:
#            df.loc[dst_mask, 'Fpts'] = (
#                DST_PROJ_MIN + (dst_df['Fpts'] - old_min) *
#                (DST_PROJ_MAX - DST_PROJ_MIN) / (old_max - old_min)
#            ).round(2)
#            print(f"[Fix DST] Rescaled projections to [{DST_PROJ_MIN}, {DST_PROJ_MAX}]")
#
#    # --- FIX 2: Synthesize DST salaries if all identical ---
#    if dst_mask.any() and df.loc[dst_mask, 'Salary'].nunique() == 1:
#        dst_df = df[dst_mask].copy()
#        rank = dst_df['Fpts'].rank(pct=True)
#        dst_df['Salary'] = (DST_SALARY_MIN + (DST_SALARY_MAX - DST_SALARY_MIN) * rank).round(-1).astype(int)
#        df.loc[dst_mask, 'Salary'] = dst_df['Salary']
#        print(f"[Fix DST] Synthesized salaries ${DST_SALARY_MIN}-${DST_SALARY_MAX}")
#
#    # Slate filter
#    if only_teams:
#        before = len(df)
#        df = df[df['Team'].isin(only_teams)].copy()
#        print(f"[Slate] {before} -> {len(df)} players")
#
#    # Percentile floor (scale-free)
#    if top_pct is not None:
#        def pct_filter(group):
#            if len(group) < 3:
#                return group
#            threshold = group['Fpts'].quantile(1 - top_pct)
#            return group[group['Fpts'] >= threshold]
#        before = len(df)
#        df = df.groupby('Position', group_keys=False).apply(pct_filter).reset_index(drop=True)
#        print(f"[Floor] Top {top_pct*100:.0f}% per position -> {before} -> {len(df)} players")
#
#    df = df[df['Salary'] > 0].copy()
#    df['_key'] = df['Name'].map(norm_name)
#    df = df.drop_duplicates('_key').reset_index(drop=True)
#
#    # Position pool filter (top N by a blend of projection + ownership)
#    def filt(group):
#        n = int(POOL_SIZES.get(group.name, 30) * pool_size_mult)
#        group = group.copy()
#        group['_rank_score'] = group['Fpts'].rank(ascending=False) + group['Own%'].rank(ascending=False)
#        return group.nsmallest(n, '_rank_score')
#
#    df = df.groupby('Position', group_keys=False).apply(filt).reset_index(drop=True)
#    return df


# ============================================================
# Field generator
# ============================================================
class FieldGenerator:
    def __init__(self, proj, seed=42):
        self.proj = proj.reset_index(drop=True)
        self.rng = np.random.default_rng(seed)
        self.pools = {}
        for pos in ['QB', 'RB', 'WR', 'TE', 'DST']:
            pool = self.proj[self.proj['Position'] == pos].copy()
            if pool.empty:
                self.pools[pos] = None
                continue
            fpts_rank = pool['Fpts'].rank(ascending=False)
            own_rank = pool['Own%'].rank(ascending=False)
            sal_rank = pool['Salary'].rank(ascending=False)
            pool['_score'] = fpts_rank + 0.5 * own_rank + 0.5 * sal_rank
            pool['_w'] = 1.0 / pool['_score'].clip(lower=1)
            pool['_w'] = pool['_w'] / pool['_w'].sum()
            self.pools[pos] = pool
        self.team_qbs = defaultdict(list)
        self.team_pass_catchers = defaultdict(list)
        for i, row in self.proj.iterrows():
            if row['Position'] == 'QB':
                self.team_qbs[row['Team']].append(i)
            elif row['Position'] in ('WR', 'TE'):
                self.team_pass_catchers[row['Team']].append(i)

    def _weighted_pick(self, pool_df, n, exclude_ids=None):
        if pool_df is None or pool_df.empty:
            return []
        sub = pool_df
        if exclude_ids:
            sub = pool_df[~pool_df.index.isin(exclude_ids)]
        if sub.empty:
            return []
        n = min(n, len(sub))
        w = sub['_w'].values.astype(float)
        if w.sum() <= 0:
            w = np.ones(len(sub))
        w = w / w.sum()
        return list(self.rng.choice(sub.index.values, size=n, replace=False, p=w))

    def _try_one(self):
        ids = []
        teams = []
        qb_picks = self._weighted_pick(self.pools['QB'], 1)
        if not qb_picks:
            return None
        ids.append(qb_picks[0])
        teams.append(self.proj.loc[qb_picks[0], 'Team'])
        stack_team = teams[0]

        for p in self._weighted_pick(self.pools['RB'], 2, exclude_ids=ids):
            ids.append(p); teams.append(self.proj.loc[p, 'Team'])

        need_wr = 3
        stack_wrs = self.pools['WR']
        if stack_wrs is not None:
            stack_wrs = stack_wrs[stack_wrs['Team'] == stack_team]
            for p in self._weighted_pick(stack_wrs, min(need_wr, 2), exclude_ids=ids):
                ids.append(p); teams.append(self.proj.loc[p, 'Team']); need_wr -= 1
        if need_wr > 0:
            for p in self._weighted_pick(self.pools['WR'], need_wr, exclude_ids=ids):
                ids.append(p); teams.append(self.proj.loc[p, 'Team'])

        te_pool = self.pools['TE']
        stack_te = te_pool[te_pool['Team'] == stack_team] if te_pool is not None else None
        picks = self._weighted_pick(stack_te, 1, exclude_ids=ids) if stack_te is not None and not stack_te.empty else []
        if not picks:
            picks = self._weighted_pick(self.pools['TE'], 1, exclude_ids=ids)
        for p in picks:
            ids.append(p); teams.append(self.proj.loc[p, 'Team'])

        flex_pools = [p for p in [self.pools['RB'], self.pools['WR'], self.pools['TE']] if p is not None]
        if flex_pools:
            flex = pd.concat(flex_pools)
            flex = flex[~flex.index.isin(ids)].copy()
            if flex.empty:
                return None
            flex['_w'] = flex['Own%'].clip(lower=0.1)
            flex['_w'] = flex['_w'] / flex['_w'].sum()
            for p in self._weighted_pick(flex, 1):
                ids.append(p); teams.append(self.proj.loc[p, 'Team'])

        for p in self._weighted_pick(self.pools['DST'], 1, exclude_ids=ids):
            ids.append(p); teams.append(self.proj.loc[p, 'Team'])

        if len(set(ids)) != 9:
            return None
        salary = int(self.proj.loc[ids, 'Salary'].sum())
        if salary > DK_SALARY_CAP or salary < DK_MIN_SALARY:
            return None
        if any(c > FIELD_MAX_PER_TEAM for c in Counter(teams).values()):
            return None
        return ids

    def generate(self, field_size, verbose=True):
        lineups = []
        attempts = 0
        max_attempts = field_size * 300
        last_report = 0
        while len(lineups) < field_size and attempts < max_attempts:
            attempts += 1
            ids = self._try_one()
            if ids is not None:
                lineups.append(ids)
            if verbose and len(lineups) > 0 and len(lineups) - last_report >= 5000:
                print(f"  generated {len(lineups)} / {field_size}")
                last_report = len(lineups)
        if verbose:
            print(f"  done: {len(lineups)} lineups from {attempts} attempts")
        return lineups


# ============================================================
# Lineup optimizer with progressive constraint relaxation
# ============================================================
class LineupOptimizer:
    def __init__(self, proj, seed=42):
        self.proj = proj.reset_index(drop=True)
        self.rng = np.random.default_rng(seed)
        self.current_min_salary = DK_MIN_SALARY
        self.current_max_per_team = FIELD_MAX_PER_TEAM

    def _projected_total(self, ids):
        return self.proj.loc[ids, 'Fpts'].sum()

    def _salary_total(self, ids):
        return int(self.proj.loc[ids, 'Salary'].sum())

    def _is_valid(self, ids):
        if len(set(ids)) != 9:
            return False
        s = self._salary_total(ids)
        if s > DK_SALARY_CAP or s < self.current_min_salary:
            return False
        teams = Counter(self.proj.loc[i, 'Team'] for i in ids)
        if any(c > self.current_max_per_team for c in teams.values()):
            return False
        return True

    def optimize_one(self, jitter_pct=0.10):
        proj = self.proj.copy()
        jitter = self.rng.normal(0, jitter_pct, len(proj)) * proj['Fpts'].values
        proj['_adj_fpts'] = proj['Fpts'] + jitter
        proj['_value'] = proj['_adj_fpts'] / (proj['Salary'] / 1000)

        pools = {}
        for pos in ['QB', 'RB', 'WR', 'TE', 'DST']:
            pool = proj[proj['Position'] == pos].sort_values('_value', ascending=False)
            pools[pos] = pool.index.tolist()

        qb_candidates = list(proj[proj['Position'] == 'QB']
                              .sort_values('_value', ascending=False).index)[:15]
        self.rng.shuffle(qb_candidates)

        for qb_choice in qb_candidates:
            ids = [qb_choice]
            teams = [proj.loc[qb_choice, 'Team']]
            stack_team = proj.loc[qb_choice, 'Team']

            rb_pool = proj[proj['Position'] == 'RB'].sort_values('_value', ascending=False)
            for rb_i in rb_pool.index:
                if rb_i in ids: continue
                if len([i for i in ids if proj.loc[i, 'Position'] == 'RB']) >= 2: break
                ids.append(rb_i); teams.append(proj.loc[rb_i, 'Team'])

            wr_pool = proj[proj['Position'] == 'WR'].sort_values('_value', ascending=False)
            stack_wrs = [i for i in wr_pool.index if proj.loc[i, 'Team'] == stack_team and i not in ids]
            for w in stack_wrs:
                if len([i for i in ids if proj.loc[i, 'Position'] == 'WR']) >= 3: break
                ids.append(w); teams.append(proj.loc[w, 'Team'])
            for w in wr_pool.index:
                if len([i for i in ids if proj.loc[i, 'Position'] == 'WR']) >= 3: break
                if w in ids: continue
                ids.append(w); teams.append(proj.loc[w, 'Team'])

            te_pool = proj[proj['Position'] == 'TE'].sort_values('_value', ascending=False)
            stack_tes = [i for i in te_pool.index if proj.loc[i, 'Team'] == stack_team and i not in ids]
            picked = False
            for t in stack_tes:
                ids.append(t); teams.append(proj.loc[t, 'Team']); picked = True; break
            if not picked:
                for t in te_pool.index:
                    if t in ids: continue
                    ids.append(t); teams.append(proj.loc[t, 'Team']); break

            flex_pool = proj[proj['Position'].isin(['RB', 'WR', 'TE'])].sort_values('_value', ascending=False)
            for f in flex_pool.index:
                if f in ids: continue
                ids.append(f); teams.append(proj.loc[f, 'Team']); break

            dst_pool = proj[proj['Position'] == 'DST'].sort_values('_value', ascending=False)
            for d in dst_pool.index:
                if d in ids: continue
                ids.append(d); teams.append(proj.loc[d, 'Team']); break

            if len(ids) != 9 or len(set(ids)) != 9: continue
            if not self._is_valid(ids): continue

            improved = True
            it = 0
            while improved and it < 15:
                improved = False; it += 1
                current_score = self._projected_total(ids)
                for slot_i, old_id in enumerate(ids):
                    pos = proj.loc[old_id, 'Position']
                    candidate_pool = pools.get(pos, [])
                    for new_id in candidate_pool[:40]:
                        if new_id in ids: continue
                        trial = ids.copy(); trial[slot_i] = new_id
                        if not self._is_valid(trial): continue
                        if self._projected_total(trial) > current_score:
                            ids = trial; improved = True; break
                    if improved: break
            return ids
        return None

    def optimize(self, num_lineups, max_overlap=5, jitter_pct=0.10,
                 qb_cap=None, dst_cap=None, verbose=True):
        if qb_cap is None: qb_cap = max(2, int(np.ceil(num_lineups * 0.35)))
        if dst_cap is None: dst_cap = max(2, int(np.ceil(num_lineups * 0.35)))

        lineups = []
        lineups_sets = []
        qb_counts = Counter()
        dst_counts = Counter()
        attempts = 0
        max_attempts = num_lineups * 200
        relaxation_step = 0

        while len(lineups) < num_lineups and attempts < max_attempts:
            attempts += 1
            ids = self.optimize_one(jitter_pct=jitter_pct)

            if ids is None:
                # Progressive relaxation: every 100 failures, loosen a constraint
                if attempts % 100 == 0 and relaxation_step < 3:
                    relaxation_step += 1
                    if relaxation_step == 1:
                        self.current_min_salary = max(DK_SALARY_CAP - 4000, self.current_min_salary - 1000)
                        print(f"  [Relax] min salary -> ${self.current_min_salary}")
                    elif relaxation_step == 2:
                        self.current_max_per_team += 1
                        print(f"  [Relax] max per team -> {self.current_max_per_team}")
                    elif relaxation_step == 3:
                        jitter_pct += 0.05
                        print(f"  [Relax] jitter -> {jitter_pct:.2f}")
                continue

            qb_id = next((i for i in ids if self.proj.loc[i, 'Position'] == 'QB'), None)
            dst_id = next((i for i in ids if self.proj.loc[i, 'Position'] == 'DST'), None)
            if qb_id is None or dst_id is None: continue
            if qb_counts[qb_id] >= qb_cap: continue
            if dst_counts[dst_id] >= dst_cap: continue

            ids_set = set(ids)
            if any(len(ids_set & prev) > max_overlap for prev in lineups_sets): continue

            lineups.append(ids)
            lineups_sets.append(ids_set)
            qb_counts[qb_id] += 1
            dst_counts[dst_id] += 1
            if verbose and len(lineups) % 5 == 0:
                print(f"  optimized {len(lineups)} / {num_lineups}")

        if verbose:
            print(f"  done: {len(lineups)} lineups in {attempts} attempts "
                  f"(relaxation step {relaxation_step})")
            print(f"  distinct QBs: {len(qb_counts)} | distinct DSTs: {len(dst_counts)}")
        return lineups


# ============================================================
# Simulation and grading — unchanged
# ============================================================
def simulate_field(proj, field_lineup_ids, iterations, seed=42):
    """
    Simulate weekly outcomes for a set of lineups using correlated sampling.

    Three levels of correlation:
      1. Player-level noise (base)
      2. Team-level shock (same-team players move together)
      3. Game-level shock (both teams in a shootout move together)

    This makes stack lineups appropriately volatile.
    """
    rng = np.random.default_rng(seed)
    n_players = len(proj)

    means = proj['Fpts'].values
    stds = proj['StdDev'].values
    # Widened StdDev clip — real NFL weekly variance is higher than [0.30, 0.55]
    stds = np.clip(
        stds,
        np.maximum(means * 0.40, 1.0),
        np.maximum(means * 0.75, 3.0)
    )

    teams = proj['Team'].values
    positions = proj['Position'].values

    # Build opponent map from the 'Opp' column if present, else fall back to team-only
    if 'Opp' in proj.columns:
        opponents = proj['Opp'].fillna('').values
    else:
        opponents = np.array([''] * n_players)

    # Game ID: canonical pair of team codes
    game_ids = np.array([
        f"{min(t, o)}_{max(t, o)}" if o and o != t else t
        for t, o in zip(teams, opponents)
    ])
    unique_games = np.unique(game_ids)

    n_lineups = len(field_lineup_ids)
    scores = np.zeros((n_lineups, iterations))
    lineup_arrays = [np.array(ids) for ids in field_lineup_ids]

    for it in range(iterations):
        # Team-level shocks: a team's offense as a whole
        team_shocks = {t: rng.normal(0, 1) for t in np.unique(teams)}
        # Game-level shocks: the game environment (shootout vs defensive battle)
        game_shocks = {g: rng.normal(0, 1) for g in unique_games}

        player_scores = np.zeros(n_players)
        for i in range(n_players):
            m, s = means[i], stds[i]
            pos, team = positions[i], teams[i]
            gid = game_ids[i]
            base = rng.normal(0, 1)
            t_shock = team_shocks[team]
            g_shock = game_shocks[gid]

            if pos == 'QB':
                z = 0.45 * base + 0.35 * t_shock + 0.20 * g_shock
            elif pos in ('WR', 'TE'):
                z = 0.55 * base + 0.30 * t_shock + 0.15 * g_shock
            elif pos == 'RB':
                z = 0.70 * base + 0.20 * t_shock + 0.10 * g_shock
            else:  # DST — benefits from a low-scoring game, so the game shock is negative
                z = 0.60 * base + 0.25 * t_shock - 0.15 * g_shock

            player_scores[i] = max(0.0, m + s * z)

        for li, arr in enumerate(lineup_arrays):
            scores[li, it] = player_scores[arr].sum()

    return scores


def grade_lineups(proj, field_lineup_ids, my_lineup_ids, iterations=5000, verbose=True):
    if verbose:
        print(f"Simulating {iterations} iterations...")
    field_scores = simulate_field(proj, field_lineup_ids, iterations)
    my_scores = simulate_field(proj, my_lineup_ids, iterations)
    n_field = len(field_lineup_ids)
    results = []
    for mi in range(len(my_lineup_ids)):
        my_col = my_scores[mi]
        ranks = np.array([1 + int((field_scores[:, it] > my_col[it]).sum()) for it in range(iterations)])
        results.append({
            'lineup_idx': mi,
            'mean_pts': round(float(my_col.mean()), 2),
            'mean_rank': round(float(ranks.mean()), 1),
            'median_rank': int(np.median(ranks)),
            'best_rank': int(ranks.min()),
            'top_1pct_rate': round(float((ranks <= n_field * 0.01).mean()), 4),
            'top_5pct_rate': round(float((ranks <= n_field * 0.05).mean()), 4),
            'top_10pct_rate': round(float((ranks <= n_field * 0.10).mean()), 4),
            'top_25pct_rate': round(float((ranks <= n_field * 0.25).mean()), 4),
        })
    return pd.DataFrame(results)


# ============================================================
# I/O
# ============================================================
def save_lineups(proj, lineups, path):
    rows = []
    for ids in lineups:
        names = [proj.loc[i, 'Name'] for i in ids]
        row = {f'slot{i}': n for i, n in enumerate(names)}
        row['Salary'] = int(proj.loc[ids, 'Salary'].sum())
        row['Fpts'] = round(float(proj.loc[ids, 'Fpts'].sum()), 2)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def load_lineups_csv(proj, path):
    import re
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return []
    if df.empty:
        return []

    print(f"[Loader] Loaded {len(df)} rows")
    print(f"[Loader] All columns: {df.columns.tolist()}")

    name_to_idx = dict(zip(proj['_key'], proj.index))

    # Match slot0..slot8 or QB/RB1/...
    slot_cols = sorted([c for c in df.columns if re.match(r'^slot\d+$', c)],
                       key=lambda c: int(c[4:]))
    if not slot_cols:
        candidates = ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE', 'FLEX', 'DST']
        slot_cols = [c for c in candidates if c in df.columns]

    print(f"[Loader] Using {len(slot_cols)} slot columns: {slot_cols}")

    lineups = []
    failed_names = []
    for i, row in df.iterrows():
        names = [str(row[c]) for c in slot_cols]
        ids = [name_to_idx.get(norm_name(n)) for n in names]
        missing = [names[j] for j, x in enumerate(ids) if x is None]
        if missing:
            failed_names.extend(missing)
            continue
        if len(set(ids)) != 9:
            continue
        lineups.append(ids)

    if failed_names:
        unique_missing = sorted(set(failed_names))
        print(f"[Loader] {len(unique_missing)} unmatched names: {unique_missing[:15]}")

    print(f"[Loader] Loaded {len(lineups)}/{len(df)} valid lineups")
    return lineups
#def load_lineups_csv(proj, path):
#    try:
#        df = pd.read_csv(path)
#    except pd.errors.EmptyDataError:
#        return []
#    if df.empty:
#        return []
#    name_to_idx = dict(zip(proj['_key'], proj.index))
#    slot_cols = [c for c in df.columns if c.startswith('slot')]
#    lineups = []
#    for _, row in df.iterrows():
#        names = [str(row[c]) for c in slot_cols]
#        ids = [name_to_idx.get(norm_name(n)) for n in names]
#        if len(ids) == 9 and all(i is not None for i in ids) and len(set(ids)) == 9:
#            lineups.append(ids)
#    return lineups


# ============================================================
# CLI
# ============================================================
def _apply_slate(args, proj):
    if args.slate_file and os.path.exists(args.slate_file):
        teams = infer_slate_teams(args.slate_file)
        if teams:
            proj = proj[proj['Team'].isin(teams)].reset_index(drop=True)
            print(f"[Slate] {len(proj)} players on slate")
    return proj


def cmd_generate(args):
    proj = load_projections(args.proj, top_pct=args.top_pct)
    proj = _apply_slate(args, proj)
    print(f"Loaded {len(proj)} players")
    gen = FieldGenerator(proj, seed=args.seed)
    print(f"Generating {args.field_size} field lineups...")
    lineups = gen.generate(args.field_size, verbose=True)
    save_lineups(proj, lineups, args.out)
    print(f"Saved {len(lineups)} lineups to {args.out}")


def cmd_optimize(args):
    proj = load_projections(args.proj, top_pct=args.top_pct)
    proj = _apply_slate(args, proj)
    print(f"Loaded {len(proj)} players")
    print(f"  QBs: {sum(proj['Position']=='QB')}, RBs: {sum(proj['Position']=='RB')}, "
          f"WRs: {sum(proj['Position']=='WR')}, TEs: {sum(proj['Position']=='TE')}, "
          f"DSTs: {sum(proj['Position']=='DST')}")
    opt = LineupOptimizer(proj, seed=args.seed)
    print(f"Optimizing {args.num_lineups} lineups...")
    lineups = opt.optimize(
        args.num_lineups,
        max_overlap=args.max_overlap,
        jitter_pct=args.jitter,
        qb_cap=args.qb_cap,
        dst_cap=args.dst_cap,
        verbose=True,
    )
    save_lineups(proj, lineups, args.out)
    print(f"Saved {len(lineups)} lineups to {args.out}")


def cmd_grade(args):
    proj = load_projections(args.proj, top_pct=args.top_pct)
    proj = _apply_slate(args, proj)
    print(f"Loaded {len(proj)} players")

    if args.field and os.path.exists(args.field):
        field_df = pd.read_csv(args.field)
        name_to_idx = dict(zip(proj['_key'], proj.index))
        slot_cols = [c for c in field_df.columns if c.startswith('slot')]
        field_lineups = []
        for _, row in field_df.iterrows():
            ids = [name_to_idx.get(norm_name(str(row[c]))) for c in slot_cols]
            if all(i is not None for i in ids):
                field_lineups.append(ids)
        print(f"  {len(field_lineups)} field lineups loaded")
    else:
        gen = FieldGenerator(proj, seed=args.seed)
        field_lineups = gen.generate(args.field_size, verbose=True)

    my_lineups = load_lineups_csv(proj, args.lineups)
    print(f"Loaded {len(my_lineups)} of my lineups")
    if not my_lineups:
        print("No lineups to grade.")
        return

    results = grade_lineups(proj, field_lineups, my_lineups, iterations=args.iterations)
    print("\n=== Grading Results ===")
    print(results.to_string(index=False))
    results.to_csv('grading_results.csv', index=False)

    for mi, ids in enumerate(my_lineups):
        names = [proj.loc[i, 'Name'] for i in ids]
        r = results.iloc[mi]
        print(f"\nLineup {mi}: proj={r['mean_pts']:.1f} mean_rank={r['mean_rank']:.0f} "
              f"top5%={r['top_5pct_rate']:.1%} top10%={r['top_10pct_rate']:.1%}")
        print(f"  {', '.join(names)}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')

    g = sub.add_parser('generate')
    g.add_argument('--proj', required=True)
    g.add_argument('--field-size', type=int, default=50000)
    g.add_argument('--out', required=True)
    g.add_argument('--seed', type=int, default=42)
    g.add_argument('--top-pct', type=float, default=None)
    g.add_argument('--slate-file', default=None)
    g.set_defaults(func=cmd_generate)

    o = sub.add_parser('optimize')
    o.add_argument('--proj', required=True)
    o.add_argument('--num-lineups', type=int, default=20)
    o.add_argument('--max-overlap', type=int, default=5)
    o.add_argument('--jitter', type=float, default=0.10)
    o.add_argument('--qb-cap', type=int, default=None)
    o.add_argument('--dst-cap', type=int, default=None)
    o.add_argument('--out', required=True)
    o.add_argument('--seed', type=int, default=42)
    o.add_argument('--top-pct', type=float, default=None)
    o.add_argument('--slate-file', default=None)
    o.set_defaults(func=cmd_optimize)

    gr = sub.add_parser('grade')
    gr.add_argument('--proj', required=True)
    gr.add_argument('--field', default=None)
    gr.add_argument('--field-size', type=int, default=20000)
    gr.add_argument('--lineups', required=True)
    gr.add_argument('--iterations', type=int, default=5000)
    gr.add_argument('--seed', type=int, default=42)
    gr.add_argument('--top-pct', type=float, default=None)
    gr.add_argument('--slate-file', default=None)
    gr.set_defaults(func=cmd_grade)

    args = parser.parse_args()
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == '__main__':
    main()