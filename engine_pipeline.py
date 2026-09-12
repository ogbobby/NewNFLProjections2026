import pandas as pd
import numpy as np
import os
from engine_core import DKCoreDataEngine


POSITION_MEAN_EFF = {
    'pass': 0.52,
    'rush': 1.00,
    'target': 1.65,
}
SHRINKAGE_K = 8


class DKProjectionPipeline:
    def __init__(self, stats_season, target_season, target_week, dk_salary_csv=None):
        self.core = DKCoreDataEngine(stats_season, target_season, target_week)
        self.core.dk_salary_csv = dk_salary_csv
        self.dk_salary_csv = dk_salary_csv

    def _shrink(self, eff_series, n_games_series, pos_mean):
        w = n_games_series / (n_games_series + SHRINKAGE_K)
        return (w * eff_series + (1 - w) * pos_mean).fillna(pos_mean)

    def allocate_and_synthesize(self, team_volumes, injured_players=None):
        print("[Pipeline] Computing micro volume market shares...")
        hist_data = self.core.master_weekly.copy().sort_values(by=['player_id', 'week'])
        injuries = injured_players if injured_players else {}

        if hist_data.empty:
            raise RuntimeError("hist_data is empty.")

        hist_data['player_pass_share']   = (hist_data['pass_attempts'] / hist_data['team_pass_attempts']).fillna(0.0).clip(0, 1)
        hist_data['player_rush_share']   = (hist_data['rush_attempts'] / hist_data['team_rush_attempts']).fillna(0.0).clip(0, 1)
        hist_data['player_target_share'] = (hist_data['targets']       / hist_data['team_targets']).fillna(0.0).clip(0, 1)

        rec_points_earned  = hist_data['receptions'] * 1.0 + hist_data['receiving_yards'] * 0.1 + hist_data['receiving_tds'] * 6.0
        rush_points_earned = hist_data['rushing_yards'] * 0.1 + hist_data['rushing_tds'] * 6.0
        pass_points_earned = (hist_data['passing_yards'] * 0.04
                              + hist_data['passing_tds'] * 4.0
                              - hist_data['passing_interceptions'] * 1.0)
        fumbles = (hist_data['sack_fumbles_lost']
                   + hist_data['rushing_fumbles_lost']
                   + hist_data['receiving_fumbles_lost'])

        hist_data['rec_points_earned']  = rec_points_earned
        hist_data['rush_points_earned'] = rush_points_earned - fumbles * 2.0
        hist_data['pass_points_earned'] = pass_points_earned

        # ============================================================
        # WEIGHTED-AVERAGE EFFICIENCY (season totals, not per-game means)
        # ============================================================
        player_agg = hist_data.groupby('player_id').agg(
            total_pass_att=('pass_attempts', 'sum'),
            total_pass_pts=('pass_points_earned', 'sum'),
            total_rush_att=('rush_attempts', 'sum'),
            total_rush_pts=('rush_points_earned', 'sum'),
            total_targets=('targets', 'sum'),
            total_rec_pts=('rec_points_earned', 'sum'),
            total_games=('week', 'count'),
        ).reset_index()

        player_agg['eff_pass_raw']   = (player_agg['total_pass_pts']   / player_agg['total_pass_att'].replace(0, np.nan)).fillna(0)
        player_agg['eff_rush_raw']   = (player_agg['total_rush_pts']   / player_agg['total_rush_att'].replace(0, np.nan)).fillna(0)
        player_agg['eff_target_raw'] = (player_agg['total_rec_pts']    / player_agg['total_targets'].replace(0, np.nan)).fillna(0)

        player_agg['eff_pass']   = self._shrink(player_agg['eff_pass_raw'],   player_agg['total_games'], POSITION_MEAN_EFF['pass'])
        player_agg['eff_rush']   = self._shrink(player_agg['eff_rush_raw'],   player_agg['total_games'], POSITION_MEAN_EFF['rush'])
        player_agg['eff_target'] = self._shrink(player_agg['eff_target_raw'], player_agg['total_games'], POSITION_MEAN_EFF['target'])

        # ============================================================
        # TEAM PASSING EFFICIENCY (for QB context)
        # ============================================================
        team_week_eff = (
            hist_data.groupby(['team', 'week'])
            .apply(lambda g: g['pass_points_earned'].sum() / max(g['pass_attempts'].sum(), 1))
            .reset_index(name='team_pass_eff')
        )
        team_pass_eff = team_week_eff.groupby('team')['team_pass_eff'].mean().to_dict()

        # ============================================================
        # PER-POSITION LOOP
        # ============================================================
        player_records = []
        for pos, span in self.core.POSITION_WINDOWS.items():
            pos_df = hist_data[hist_data['position'] == pos].copy()
            print(f"[Debug] position={pos}: {len(pos_df)} rows")
            if pos_df.empty:
                continue
            
            pos_df['games_played'] = pos_df.groupby('player_id')['week'].transform('count')

            pos_df['base_pass_share']   = pos_df.groupby('player_id')['player_pass_share'].transform(lambda x: x.ewm(span=span, min_periods=1).mean())
            pos_df['base_rush_share']   = pos_df.groupby('player_id')['player_rush_share'].transform(lambda x: x.ewm(span=span, min_periods=1).mean())
            pos_df['base_target_share'] = pos_df.groupby('player_id')['player_target_share'].transform(lambda x: x.ewm(span=span, min_periods=1).mean())

            pos_df['pass_share_std']   = pos_df.groupby('player_id')['player_pass_share'].transform('std').fillna(0)
            pos_df['rush_share_std']   = pos_df.groupby('player_id')['player_rush_share'].transform('std').fillna(0)
            pos_df['target_share_std'] = pos_df.groupby('player_id')['player_target_share'].transform('std').fillna(0)

            latest = pos_df.groupby('player_id').last().reset_index()
            latest = latest.merge(
                player_agg[['player_id', 'eff_pass', 'eff_rush', 'eff_target']],
                on='player_id', how='left'
            )
            latest['team_pass_eff'] = latest['team'].map(team_pass_eff).fillna(POSITION_MEAN_EFF['pass'])

            player_records.append(latest[[
                'player_id', 'player_name', 'position', 'team',
                'base_pass_share', 'base_rush_share', 'base_target_share',
                'eff_pass', 'eff_rush', 'eff_target', 'team_pass_eff',
                'pass_share_std', 'rush_share_std', 'target_share_std'
            ]])

        if not player_records:
            raise RuntimeError("player_records is empty.")

        df_players = pd.concat(player_records, ignore_index=True).rename(columns={'team': 'recent_team'})
        print(f"[Debug] df_players rows: {len(df_players)}")

        TEAM_ALIASES = {
            'LA': 'LAR', 'STL': 'LAR', 'SL': 'LAR',
            'SD': 'LAC', 'OAK': 'LV', 'LVR': 'LV',
            'WSH': 'WAS', 'JAC': 'JAX', 'ARZ': 'ARI',
            'BLT': 'BAL', 'CLV': 'CLE', 'HST': 'HOU',
        }
        df_players['recent_team'] = df_players['recent_team'].replace(TEAM_ALIASES)
        team_volumes = team_volumes.copy()
        team_volumes['recent_team'] = team_volumes['recent_team'].replace(TEAM_ALIASES)

        for injured_name, details in injuries.items():
            if injured_name in df_players['player_name'].values:
                match = df_players[df_players['player_name'] == injured_name].iloc[0]
                team = match['recent_team']
                p_share = match['base_pass_share']
                r_share = match['base_rush_share']
                c_share = match['base_target_share']

                df_players.loc[df_players['player_name'] == injured_name,
                               ['base_pass_share', 'base_rush_share', 'base_target_share']] = 0.0
                team_mask = (df_players['recent_team'] == team) & (df_players['player_name'] != injured_name)

                if details['pos'] in ['WR', 'TE']:
                    df_players.loc[team_mask & df_players['position'].isin(['WR', 'TE']), 'base_pass_share'] += p_share * 0.70
                    df_players.loc[team_mask & (df_players['position'] == 'RB'), 'base_pass_share'] += p_share * 0.30
                    df_players.loc[team_mask & df_players['position'].isin(['WR', 'TE']), 'base_target_share'] += c_share * 0.70
                    df_players.loc[team_mask & (df_players['position'] == 'RB'), 'base_target_share'] += c_share * 0.30
                elif details['pos'] == 'RB':
                    df_players.loc[team_mask & (df_players['position'] == 'RB'), 'base_rush_share'] += r_share * 0.80
                    df_players.loc[team_mask & df_players['position'].isin(['WR', 'TE']), 'base_rush_share'] += r_share * 0.20
                    df_players.loc[team_mask & df_players['position'].isin(['WR', 'TE']), 'base_target_share'] += c_share * 0.55
                    df_players.loc[team_mask & (df_players['position'] == 'RB'), 'base_target_share'] += c_share * 0.45

        merged = pd.merge(df_players, team_volumes, on='recent_team', how='left')

        defaults = {'proj_plays': 64.0, 'proj_team_pass': 38.4, 'proj_team_rec': 24.2,
                    'proj_team_rush': 25.6, 'implied_total': 22.0, 'completion_rate': 0.62}
        for col, default in defaults.items():
            if col in merged.columns:
                merged[col] = merged[col].fillna(default)
            else:
                merged[col] = default

        if 'opponent_team' in merged.columns:
            merged['opponent_team'] = merged['opponent_team'].fillna('UNK')
        else:
            merged['opponent_team'] = 'UNK'
        if 'is_fav' in merged.columns:
            merged['is_fav'] = merged['is_fav'].fillna(False)
        else:
            merged['is_fav'] = False

        # ============================================================
        # BLEND TEAM EFFICIENCY INTO QB EFFICIENCY
        # ============================================================
        qb_mask = merged['position'] == 'QB'
        merged.loc[qb_mask, 'eff_pass'] = (
            0.5 * merged.loc[qb_mask, 'eff_pass']
            + 0.5 * merged.loc[qb_mask, 'team_pass_eff']
        )

        # Position-aware team-quality scaling
        base_quality = (merged['implied_total'] / 22.0).clip(0.70, 1.40)
        merged['team_quality_mod'] = np.where(
            merged['position'] == 'QB',
            base_quality ** 1.6,
            np.where(
                merged['position'].isin(['WR', 'TE']),
                base_quality ** 1.4,
                base_quality ** 1.1
            )
        )

        high_usage = (
            (merged['position'].isin(['WR', 'TE']) & (merged['base_target_share'] >= 0.22)) |
            ((merged['position'] == 'RB') & (merged['base_rush_share'] >= 0.55))
        )
        elite_pass_catcher = (
            (merged['position'].isin(['WR', 'TE']) & (merged['base_target_share'] >= 0.25)) |
            ((merged['position'] == 'RB') & (merged['base_target_share'] >= 0.15) & (merged['base_rush_share'] >= 0.55))
        )

        merged['team_quality_mod'] = np.where(
            high_usage,
            np.maximum(merged['team_quality_mod'], 0.90),
            merged['team_quality_mod']
        )
        merged['team_quality_mod'] = np.where(
            elite_pass_catcher,
            np.maximum(merged['team_quality_mod'], 0.95) * 1.05,
            merged['team_quality_mod']
        )

        merged['proj_pass_volume'] = merged['proj_team_pass'] * merged['base_pass_share'] * merged['team_quality_mod']
        merged['proj_rush_volume'] = merged['proj_team_rush'] * merged['base_rush_share'] * merged['team_quality_mod']
        merged['proj_team_targets'] = merged['proj_team_rec'] / merged['completion_rate'].replace(0, 0.62)
        merged['proj_target_volume'] = merged['proj_team_targets'] * merged['base_target_share'] * merged['team_quality_mod']

        merged['raw_projection'] = (
            merged['proj_pass_volume'] * merged['eff_pass']
            + merged['proj_rush_volume'] * merged['eff_rush']
            + merged['proj_target_volume'] * merged['eff_target']
        )
        print(f"[Debug] raw_projection range: {merged['raw_projection'].min():.2f}–{merged['raw_projection'].max():.2f}")

        def ceiling_multiplier(row):
            pos = row['position']
            if pos in ['WR', 'TE']:
                share = row['base_target_share']
                std   = row['target_share_std']
                if share >= 0.28:   base = 1.55
                elif share >= 0.22: base = 1.40
                elif share >= 0.16: base = 1.25
                elif share >= 0.10: base = 1.15
                else:               base = 1.05
                base += (0.15 - min(std, 0.15)) * 0.5
            elif pos == 'RB':
                rush = row['base_rush_share']
                tgt  = row['base_target_share']
                std  = row['rush_share_std']
                if rush >= 0.60 and tgt >= 0.12: base = 1.50
                elif rush >= 0.50:               base = 1.35
                elif rush >= 0.35:               base = 1.22
                elif rush >= 0.20:               base = 1.12
                else:                            base = 1.02
                base += (0.15 - min(std, 0.15)) * 0.5
            elif pos == 'QB':
                rush = row['base_rush_share']
                if rush >= 0.15:   base = 1.45
                elif rush >= 0.10: base = 1.32
                elif rush >= 0.05: base = 1.22
                else:              base = 1.12
            else:
                base = 1.15
            return round(base, 3)

        merged['ceiling_multiplier'] = merged.apply(ceiling_multiplier, axis=1)
        merged['ceiling_projection'] = (merged['raw_projection'] * merged['ceiling_multiplier']).round(2)

        def stability_score(row):
            if row['position'] in ['WR', 'TE']:
                std = row['target_share_std']
            elif row['position'] == 'RB':
                std = (row['rush_share_std'] + row['target_share_std']) / 2
            else:
                std = row['pass_share_std']
            return round(max(0.0, 1.0 - min(std, 0.25) / 0.25), 3)

        merged['usage_stability'] = merged.apply(stability_score, axis=1)
        print(f"[Debug] ceiling_projection range: {merged['ceiling_projection'].min():.2f}–{merged['ceiling_projection'].max():.2f}")
        return merged

    #def allocate_and_synthesize(self, team_volumes, injured_players=None):
    #    print("[Pipeline] Computing micro volume market shares...")
    #    hist_data = self.core.master_weekly.copy().sort_values(by=['player_id', 'week'])
    #    injuries = injured_players if injured_players else {}
#
    #    if hist_data.empty:
    #        raise RuntimeError("hist_data is empty.")
#
    #    hist_data['player_pass_share']   = (hist_data['pass_attempts'] / hist_data['team_pass_attempts']).fillna(0.0).clip(0, 1)
    #    hist_data['player_rush_share']   = (hist_data['rush_attempts'] / hist_data['team_rush_attempts']).fillna(0.0).clip(0, 1)
    #    hist_data['player_target_share'] = (hist_data['targets']       / hist_data['team_targets']).fillna(0.0).clip(0, 1)
#
    #    rec_points_earned  = hist_data['receptions'] * 1.0 + hist_data['receiving_yards'] * 0.1 + hist_data['receiving_tds'] * 6.0
    #    rush_points_earned = hist_data['rushing_yards'] * 0.1 + hist_data['rushing_tds'] * 6.0
    #    pass_points_earned = (hist_data['passing_yards'] * 0.04
    #                          + hist_data['passing_tds'] * 4.0
    #                          - hist_data['passing_interceptions'] * 1.0)
    #    fumbles = (hist_data['sack_fumbles_lost']
    #               + hist_data['rushing_fumbles_lost']
    #               + hist_data['receiving_fumbles_lost'])
#
    #    hist_data['pts_per_pass_att'] = (pass_points_earned / hist_data['pass_attempts']).replace([np.inf, -np.inf], 0).fillna(0)
    #    hist_data['pts_per_rush_att'] = ((rush_points_earned - fumbles * 2.0) / hist_data['rush_attempts']).replace([np.inf, -np.inf], 0).fillna(0)
    #    hist_data['pts_per_target']   = (rec_points_earned / hist_data['targets']).replace([np.inf, -np.inf], 0).fillna(0)
#
    #    player_records = []
    #    for pos, span in self.core.POSITION_WINDOWS.items():
    #        pos_df = hist_data[hist_data['position'] == pos].copy()
    #        print(f"[Debug] position={pos}: {len(pos_df)} rows")
    #        if pos_df.empty:
    #            continue
#
    #        pos_df['games_played'] = pos_df.groupby('player_id')['week'].transform('count')
#
    #        pos_df['base_pass_share']   = pos_df.groupby('player_id')['player_pass_share'].transform(lambda x: x.ewm(span=span, min_periods=1).mean())
    #        pos_df['base_rush_share']   = pos_df.groupby('player_id')['player_rush_share'].transform(lambda x: x.ewm(span=span, min_periods=1).mean())
    #        pos_df['base_target_share'] = pos_df.groupby('player_id')['player_target_share'].transform(lambda x: x.ewm(span=span, min_periods=1).mean())
#
    #        raw_pass_eff   = pos_df.groupby('player_id')['pts_per_pass_att'].transform(lambda x: x.ewm(span=span, min_periods=1).mean())
    #        raw_rush_eff   = pos_df.groupby('player_id')['pts_per_rush_att'].transform(lambda x: x.ewm(span=span, min_periods=1).mean())
    #        raw_target_eff = pos_df.groupby('player_id')['pts_per_target'].transform(lambda x: x.ewm(span=span, min_periods=1).mean())
#
    #        pos_df['eff_pass']   = self._shrink(raw_pass_eff,   pos_df['games_played'], POSITION_MEAN_EFF['pass'])
    #        pos_df['eff_rush']   = self._shrink(raw_rush_eff,   pos_df['games_played'], POSITION_MEAN_EFF['rush'])
    #        pos_df['eff_target'] = self._shrink(raw_target_eff, pos_df['games_played'], POSITION_MEAN_EFF['target'])
#
    #        pos_df['pass_share_std']   = pos_df.groupby('player_id')['player_pass_share'].transform('std').fillna(0)
    #        pos_df['rush_share_std']   = pos_df.groupby('player_id')['player_rush_share'].transform('std').fillna(0)
    #        pos_df['target_share_std'] = pos_df.groupby('player_id')['player_target_share'].transform('std').fillna(0)
#
    #        latest = pos_df.groupby('player_id').last().reset_index()
    #        player_records.append(latest[[
    #            'player_id', 'player_name', 'position', 'team',
    #            'base_pass_share', 'base_rush_share', 'base_target_share',
    #            'eff_pass', 'eff_rush', 'eff_target',
    #            'pass_share_std', 'rush_share_std', 'target_share_std'
    #        ]])
#
    #    if not player_records:
    #        raise RuntimeError("player_records is empty.")
#
    #    df_players = pd.concat(player_records, ignore_index=True).rename(columns={'team': 'recent_team'})
    #    print(f"[Debug] df_players rows: {len(df_players)}")
#
    #    TEAM_ALIASES = {
    #        'LA': 'LAR', 'STL': 'LAR', 'SL': 'LAR',
    #        'SD': 'LAC', 'OAK': 'LV', 'LVR': 'LV',
    #        'WSH': 'WAS', 'JAC': 'JAX', 'ARZ': 'ARI',
    #        'BLT': 'BAL', 'CLV': 'CLE', 'HST': 'HOU',
    #    }
    #    df_players['recent_team'] = df_players['recent_team'].replace(TEAM_ALIASES)
    #    team_volumes = team_volumes.copy()
    #    team_volumes['recent_team'] = team_volumes['recent_team'].replace(TEAM_ALIASES)
#
    #    for injured_name, details in injuries.items():
    #        if injured_name in df_players['player_name'].values:
    #            match = df_players[df_players['player_name'] == injured_name].iloc[0]
    #            team = match['recent_team']
    #            p_share = match['base_pass_share']
    #            r_share = match['base_rush_share']
    #            c_share = match['base_target_share']
#
    #            df_players.loc[df_players['player_name'] == injured_name,
    #                           ['base_pass_share', 'base_rush_share', 'base_target_share']] = 0.0
    #            team_mask = (df_players['recent_team'] == team) & (df_players['player_name'] != injured_name)
#
    #            if details['pos'] in ['WR', 'TE']:
    #                df_players.loc[team_mask & df_players['position'].isin(['WR', 'TE']), 'base_pass_share'] += p_share * 0.70
    #                df_players.loc[team_mask & (df_players['position'] == 'RB'), 'base_pass_share'] += p_share * 0.30
    #                df_players.loc[team_mask & df_players['position'].isin(['WR', 'TE']), 'base_target_share'] += c_share * 0.70
    #                df_players.loc[team_mask & (df_players['position'] == 'RB'), 'base_target_share'] += c_share * 0.30
    #            elif details['pos'] == 'RB':
    #                df_players.loc[team_mask & (df_players['position'] == 'RB'), 'base_rush_share'] += r_share * 0.80
    #                df_players.loc[team_mask & df_players['position'].isin(['WR', 'TE']), 'base_rush_share'] += r_share * 0.20
    #                df_players.loc[team_mask & df_players['position'].isin(['WR', 'TE']), 'base_target_share'] += c_share * 0.55
    #                df_players.loc[team_mask & (df_players['position'] == 'RB'), 'base_target_share'] += c_share * 0.45
#
    #    merged = pd.merge(df_players, team_volumes, on='recent_team', how='left')
#
    #    defaults = {'proj_plays': 64.0, 'proj_team_pass': 38.4, 'proj_team_rec': 24.2,
    #                'proj_team_rush': 25.6, 'implied_total': 22.0, 'completion_rate': 0.62}
    #    for col, default in defaults.items():
    #        if col in merged.columns:
    #            merged[col] = merged[col].fillna(default)
    #        else:
    #            merged[col] = default
#
    #    if 'opponent_team' in merged.columns:
    #        merged['opponent_team'] = merged['opponent_team'].fillna('UNK')
    #    else:
    #        merged['opponent_team'] = 'UNK'
    #    if 'is_fav' in merged.columns:
    #        merged['is_fav'] = merged['is_fav'].fillna(False)
    #    else:
    #        merged['is_fav'] = False
#
    #    # ============================================================
    #    # POSITION-AWARE TEAM-QUALITY SCALING
    #    # ============================================================
    #    base_quality = (merged['implied_total'] / 22.0).clip(0.70, 1.40)
    #    merged['team_quality_mod'] = np.where(
    #        merged['position'] == 'QB',
    #        base_quality ** 1.6,
    #        np.where(
    #            merged['position'].isin(['WR', 'TE']),
    #            base_quality ** 1.4,
    #            base_quality ** 1.1
    #        )
    #    )
#
    #    # High-usage pass-catchers are protected from bad team signals
    #    #high_usage = (
    #    #    (merged['position'].isin(['WR', 'TE']) & (merged['base_target_share'] >= 0.22)) |
    #    #    ((merged['position'] == 'RB') & (merged['base_rush_share'] >= 0.55))
    #    #)
    #    #merged['team_quality_mod'] = np.where(
    #    #    high_usage,
    #    #    np.maximum(merged['team_quality_mod'], 0.90),
    #    #    merged['team_quality_mod']
    #    #)
    #    high_usage = (
    #        (merged['position'].isin(['WR', 'TE']) & (merged['base_target_share'] >= 0.22)) |
    #        ((merged['position'] == 'RB') & (merged['base_rush_share'] >= 0.55))
    #    )
    #    elite_pass_catcher = (
    #        (merged['position'].isin(['WR', 'TE']) & (merged['base_target_share'] >= 0.25)) |
    #        ((merged['position'] == 'RB') & (merged['base_target_share'] >= 0.15) & (merged['base_rush_share'] >= 0.55))
    #    )
    #    
    #    merged['team_quality_mod'] = np.where(
    #        high_usage,
    #        np.maximum(merged['team_quality_mod'], 0.90),
    #        merged['team_quality_mod']
    #    )
    #    merged['team_quality_mod'] = np.where(
    #        elite_pass_catcher,
    #        np.maximum(merged['team_quality_mod'], 0.95) * 1.05,
    #        merged['team_quality_mod']
    #    )
#
    #    merged['proj_pass_volume'] = merged['proj_team_pass'] * merged['base_pass_share'] * merged['team_quality_mod']
    #    merged['proj_rush_volume'] = merged['proj_team_rush'] * merged['base_rush_share'] * merged['team_quality_mod']
    #    merged['proj_team_targets'] = merged['proj_team_rec'] / merged['completion_rate'].replace(0, 0.62)
    #    merged['proj_target_volume'] = merged['proj_team_targets'] * merged['base_target_share'] * merged['team_quality_mod']
#
    #    merged['raw_projection'] = (
    #        merged['proj_pass_volume'] * merged['eff_pass']
    #        + merged['proj_rush_volume'] * merged['eff_rush']
    #        + merged['proj_target_volume'] * merged['eff_target']
    #    )
    #    print(f"[Debug] raw_projection range: {merged['raw_projection'].min():.2f}–{merged['raw_projection'].max():.2f}")
#
    #    def ceiling_multiplier(row):
    #        pos = row['position']
    #        if pos in ['WR', 'TE']:
    #            share = row['base_target_share']
    #            std   = row['target_share_std']
    #            if share >= 0.28:   base = 1.55
    #            elif share >= 0.22: base = 1.40
    #            elif share >= 0.16: base = 1.25
    #            elif share >= 0.10: base = 1.15
    #            else:               base = 1.05
    #            base += (0.15 - min(std, 0.15)) * 0.5
    #        elif pos == 'RB':
    #            rush = row['base_rush_share']
    #            tgt  = row['base_target_share']
    #            std  = row['rush_share_std']
    #            if rush >= 0.60 and tgt >= 0.12: base = 1.50
    #            elif rush >= 0.50:               base = 1.35
    #            elif rush >= 0.35:               base = 1.22
    #            elif rush >= 0.20:               base = 1.12
    #            else:                            base = 1.02
    #            base += (0.15 - min(std, 0.15)) * 0.5
    #        elif pos == 'QB':
    #            rush = row['base_rush_share']
    #            if rush >= 0.15:   base = 1.45
    #            elif rush >= 0.10: base = 1.32
    #            elif rush >= 0.05: base = 1.22
    #            else:              base = 1.12
    #        else:
    #            base = 1.15
    #        return round(base, 3)
#
    #    merged['ceiling_multiplier'] = merged.apply(ceiling_multiplier, axis=1)
    #    merged['ceiling_projection'] = (merged['raw_projection'] * merged['ceiling_multiplier']).round(2)
#
    #    def stability_score(row):
    #        if row['position'] in ['WR', 'TE']:
    #            std = row['target_share_std']
    #        elif row['position'] == 'RB':
    #            std = (row['rush_share_std'] + row['target_share_std']) / 2
    #        else:
    #            std = row['pass_share_std']
    #        return round(max(0.0, 1.0 - min(std, 0.25) / 0.25), 3)
#
    #    merged['usage_stability'] = merged.apply(stability_score, axis=1)
    #    print(f"[Debug] ceiling_projection range: {merged['ceiling_projection'].min():.2f}–{merged['ceiling_projection'].max():.2f}")
    #    return merged

    def run_full_pipeline(self, injured_players_dict=None):
        self.core.fetch_and_clean_data()
        dvp_matrix = self.core.calculate_matchup_dvp()
        team_volumes = self.core.project_macro_team_volume()

        print(f"[Debug] team_volumes rows: {len(team_volumes)}")

        final_df = self.allocate_and_synthesize(team_volumes, injured_players_dict)
        final_df = pd.merge(
            final_df, dvp_matrix,
            left_on=['opponent_team', 'position'],
            right_on=['defense_team', 'position'],
            how='left'
        ).fillna({'dvp_multiplier': 1.0})

        final_df = final_df[final_df['opponent_team'] != 'UNK'].copy()
        print(f"[Debug] after bye filter: {len(final_df)} rows")

        dvp_effect = 1 + (final_df['dvp_multiplier'] - 1) * 0.7
        final_df['final_projection']   = final_df['raw_projection'] * dvp_effect
        final_df['ceiling_projection'] = final_df['ceiling_projection'] * dvp_effect

        final_df.loc[(final_df['is_fav'] == True) & (final_df['position'] == 'RB'), 'final_projection'] *= 1.05
        final_df.loc[(final_df['is_fav'] == False) & (final_df['position'].isin(['WR', 'TE'])), 'final_projection'] *= 1.03
        final_df.loc[(final_df['is_fav'] == True) & (final_df['position'] == 'RB'), 'ceiling_projection'] *= 1.05
        final_df.loc[(final_df['is_fav'] == False) & (final_df['position'].isin(['WR', 'TE'])), 'ceiling_projection'] *= 1.03

        final_df['final_projection']   = final_df['final_projection'].clip(lower=0.0).round(2)
        final_df['ceiling_projection'] = final_df['ceiling_projection'].clip(lower=0.0).round(2)

        output_cols = ['player_name', 'position', 'recent_team', 'opponent_team',
                       'implied_total', 'final_projection', 'ceiling_projection',
                       'ceiling_multiplier', 'usage_stability']
        output_df = final_df[output_cols].sort_values(by='final_projection', ascending=False)
        print(f"[Debug] output_df rows from pipeline: {len(output_df)}")
        return output_df