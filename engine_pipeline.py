import pandas as pd
import numpy as np
import os
from engine_core import DKCoreDataEngine


class DKProjectionPipeline:
    def __init__(self, stats_season, target_season, target_week, dk_salary_csv=None):
        self.core = DKCoreDataEngine(stats_season, target_season, target_week)
        self.core.dk_salary_csv = dk_salary_csv
        self.dk_salary_csv = dk_salary_csv

    def allocate_and_synthesize(self, team_volumes, injured_players=None):
        print("[Pipeline] Computing micro volume market shares...")
        hist_data = self.core.master_weekly.copy().sort_values(by=['player_id', 'week'])
        injuries = injured_players if injured_players else {}

        if hist_data.empty:
            raise RuntimeError("hist_data is empty.")

        # --- Shares (using real targets) ---
        hist_data['player_pass_share']   = (hist_data['pass_attempts'] / hist_data['team_pass_attempts']).fillna(0.0).clip(0, 1)
        hist_data['player_rush_share']   = (hist_data['rush_attempts'] / hist_data['team_rush_attempts']).fillna(0.0).clip(0, 1)
        hist_data['player_target_share'] = (hist_data['targets']       / hist_data['team_targets']).fillna(0.0).clip(0, 1)

        # --- Category-specific points ---
        rec_points_earned  = hist_data['receptions'] * 1.0 + hist_data['receiving_yards'] * 0.1 + hist_data['receiving_tds'] * 6.0
        rush_points_earned = hist_data['rushing_yards'] * 0.1 + hist_data['rushing_tds'] * 6.0
        pass_points_earned = (hist_data['passing_yards'] * 0.04
                              + hist_data['passing_tds'] * 4.0
                              - hist_data['passing_interceptions'] * 1.0)
        fumbles = (hist_data['sack_fumbles_lost']
                   + hist_data['rushing_fumbles_lost']
                   + hist_data['receiving_fumbles_lost'])

        hist_data['pts_per_pass_att'] = (pass_points_earned / hist_data['pass_attempts']).replace([np.inf, -np.inf], 0).fillna(0)
        hist_data['pts_per_rush_att'] = ((rush_points_earned - fumbles * 2.0) / hist_data['rush_attempts']).replace([np.inf, -np.inf], 0).fillna(0)
        hist_data['pts_per_target']   = (rec_points_earned / hist_data['targets']).replace([np.inf, -np.inf], 0).fillna(0)

        player_records = []
        for pos, span in self.core.POSITION_WINDOWS.items():
            pos_df = hist_data[hist_data['position'] == pos].copy()
            print(f"[Debug] position={pos}: {len(pos_df)} rows")
            if pos_df.empty:
                continue

            pos_df['base_pass_share']   = pos_df.groupby('player_id')['player_pass_share'].transform(lambda x: x.ewm(span=span, min_periods=1).mean())
            pos_df['base_rush_share']   = pos_df.groupby('player_id')['player_rush_share'].transform(lambda x: x.ewm(span=span, min_periods=1).mean())
            pos_df['base_target_share'] = pos_df.groupby('player_id')['player_target_share'].transform(lambda x: x.ewm(span=span, min_periods=1).mean())
            pos_df['eff_pass']   = pos_df.groupby('player_id')['pts_per_pass_att'].transform(lambda x: x.ewm(span=span, min_periods=1).mean())
            pos_df['eff_rush']   = pos_df.groupby('player_id')['pts_per_rush_att'].transform(lambda x: x.ewm(span=span, min_periods=1).mean())
            pos_df['eff_target'] = pos_df.groupby('player_id')['pts_per_target'].transform(lambda x: x.ewm(span=span, min_periods=1).mean())

            # Usage stability: how consistent were shares week to week?
            pos_df['pass_share_std']   = pos_df.groupby('player_id')['player_pass_share'].transform('std').fillna(0)
            pos_df['rush_share_std']   = pos_df.groupby('player_id')['player_rush_share'].transform('std').fillna(0)
            pos_df['target_share_std'] = pos_df.groupby('player_id')['player_target_share'].transform('std').fillna(0)

            latest = pos_df.groupby('player_id').last().reset_index()
            player_records.append(latest[[
                'player_id', 'player_name', 'position', 'team',
                'base_pass_share', 'base_rush_share', 'base_target_share',
                'eff_pass', 'eff_rush', 'eff_target',
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

        # --- Injury redistribution ---
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

        print(f"[Debug] df_players teams: {sorted(df_players['recent_team'].dropna().unique())}")
        print(f"[Debug] team_volumes teams: {sorted(team_volumes['recent_team'].dropna().unique())}")

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

        merged['proj_pass_volume'] = merged['proj_team_pass'] * merged['base_pass_share']
        merged['proj_rush_volume'] = merged['proj_team_rush'] * merged['base_rush_share']
        merged['proj_team_targets'] = merged['proj_team_rec'] / merged['completion_rate'].replace(0, 0.62)
        merged['proj_target_volume'] = merged['proj_team_targets'] * merged['base_target_share']

        merged['raw_projection'] = (
            merged['proj_pass_volume'] * merged['eff_pass']
            + merged['proj_rush_volume'] * merged['eff_rush']
            + merged['proj_target_volume'] * merged['eff_target']
        )
        print(f"[Debug] raw_projection range: {merged['raw_projection'].min():.2f}–{merged['raw_projection'].max():.2f}")

        # ============================================================
        # CEILING PROJECTION LAYER
        # ============================================================
        # Ceiling = median projection * a usage-dependent multiplier.
        # Higher target / rush share = higher ceiling because the player has
        # more opportunities for spike weeks. Lower share volatility = higher
        # floor AND higher ceiling, because we trust the usage to repeat.

        def ceiling_multiplier(row):
            pos = row['position']
            if pos in ['WR', 'TE']:
                share = row['base_target_share']
                std   = row['target_share_std']
                if share >= 0.28:
                    base = 1.55
                elif share >= 0.22:
                    base = 1.40
                elif share >= 0.16:
                    base = 1.25
                elif share >= 0.10:
                    base = 1.15
                else:
                    base = 1.05
                # Lower volatility = higher multiplier (trust the ceiling)
                base += (0.15 - min(std, 0.15)) * 0.5
            elif pos == 'RB':
                rush = row['base_rush_share']
                tgt  = row['base_target_share']
                std  = row['rush_share_std']
                if rush >= 0.60 and tgt >= 0.12:
                    base = 1.50  # true bellcow
                elif rush >= 0.50:
                    base = 1.35
                elif rush >= 0.35:
                    base = 1.22
                elif rush >= 0.20:
                    base = 1.12
                else:
                    base = 1.02
                base += (0.15 - min(std, 0.15)) * 0.5
            elif pos == 'QB':
                rush = row['base_rush_share']
                if rush >= 0.15:
                    base = 1.45  # dual-threat
                elif rush >= 0.10:
                    base = 1.32
                elif rush >= 0.05:
                    base = 1.22
                else:
                    base = 1.12
            else:
                base = 1.15
            return round(base, 3)

        merged['ceiling_multiplier'] = merged.apply(ceiling_multiplier, axis=1)
        merged['ceiling_projection'] = (merged['raw_projection'] * merged['ceiling_multiplier']).round(2)

        # Usage stability score: 0 = volatile, 1 = rock solid
        # Derived from the standard deviation of shares. Lower std = higher stability.
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

        final_df['final_projection'] = final_df['raw_projection'] * final_df['dvp_multiplier']
        # Ceiling also benefits from DvP: a soft matchup raises the ceiling more than the median
        final_df['ceiling_projection'] = final_df['ceiling_projection'] * (1 + (final_df['dvp_multiplier'] - 1) * 1.15)

        final_df.loc[(final_df['is_fav'] == True) & (final_df['position'] == 'RB'), 'final_projection'] *= 1.05
        final_df.loc[(final_df['is_fav'] == False) & (final_df['position'].isin(['WR', 'TE'])), 'final_projection'] *= 1.03
        final_df.loc[(final_df['is_fav'] == True) & (final_df['position'] == 'RB'), 'ceiling_projection'] *= 1.05
        final_df.loc[(final_df['is_fav'] == False) & (final_df['position'].isin(['WR', 'TE'])), 'ceiling_projection'] *= 1.03

        final_df['final_projection']   = final_df['final_projection'].round(2)
        final_df['ceiling_projection'] = final_df['ceiling_projection'].round(2)

        output_cols = ['player_name', 'position', 'recent_team', 'opponent_team',
                       'implied_total', 'final_projection', 'ceiling_projection',
                       'ceiling_multiplier', 'usage_stability']
        output_df = final_df[output_cols].sort_values(by='final_projection', ascending=False)

        print(f"[Debug] output_df rows before salary merge: {len(output_df)}")

        if self.dk_salary_csv and os.path.exists(self.dk_salary_csv):
            dk_sal_raw = pd.read_csv(self.dk_salary_csv)
            print(f"[Debug] DK CSV columns: {dk_sal_raw.columns.tolist()}")

            if 'Name' not in dk_sal_raw.columns or 'Salary' not in dk_sal_raw.columns:
                print("[Warning] DK CSV missing 'Name' or 'Salary'. Skipping salary merge.")
            else:
                dk_sal = dk_sal_raw[['Name', 'Salary']].copy()
                dk_sal.columns = ['player_name', 'salary']
                dk_sal['player_name'] = dk_sal['player_name'].astype(str).str.strip()
                output_df['player_name'] = output_df['player_name'].astype(str).str.strip()

                def norm(n):
                    return (n.replace('.', '').replace("'", '').replace('-', ' ')
                             .replace(' Jr', '').replace(' Sr', '').replace(' III', '')
                             .lower().strip())

                output_df['_key'] = output_df['player_name'].map(norm)
                dk_sal['_key']    = dk_sal['player_name'].map(norm)

                matched = pd.merge(
                    output_df[['_key']].drop_duplicates(),
                    dk_sal[['_key', 'salary']].drop_duplicates('_key'),
                    on='_key', how='inner'
                )
                print(f"[Debug] salary key matches: {len(matched)}")

                if len(matched) == 0:
                    print("[Warning] No salary matches. Returning projections without salaries.")
                    output_df = output_df.drop(columns=['_key'], errors='ignore')
                else:
                    output_df = pd.merge(output_df, dk_sal[['_key', 'salary']].drop_duplicates('_key'),
                                         on='_key', how='left')
                    output_df = output_df.drop(columns=['_key'])
                    output_df['value'] = ((output_df['final_projection'] / output_df['salary']) * 1000).round(2)
                    output_df = output_df.sort_values(by='value', ascending=False)

        return output_df