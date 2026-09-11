import pandas as pd
import numpy as np
import os
from engine_core import DKCoreDataEngine


class DKProjectionPipeline:
    def __init__(self, stats_season, target_season, target_week, dk_salary_csv=None):
        self.core = DKCoreDataEngine(stats_season, target_season, target_week)
        self.core.dk_salary_csv = dk_salary_csv   # so core can fall back to DK Game Info
        self.dk_salary_csv = dk_salary_csv

    def allocate_and_synthesize(self, team_volumes, injured_players=None):
        print("[Pipeline] Computing micro volume market shares...")
        hist_data = self.core.master_weekly.copy().sort_values(by=['player_id', 'week'])
        injuries = injured_players if injured_players else {}

        print(f"[Debug] hist_data rows: {len(hist_data)}")
        if hist_data.empty:
            raise RuntimeError("hist_data is empty — no stats rows found.")

        hist_data['player_pass_share'] = (hist_data['pass_attempts'] / hist_data['team_pass_attempts']).fillna(0.0)
        hist_data['player_rush_share'] = (hist_data['rush_attempts'] / hist_data['team_rush_attempts']).fillna(0.0)
        hist_data['player_rec_share']  = (hist_data['receptions']    / hist_data['team_receptions']).fillna(0.0)

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
        hist_data['pts_per_rec']      = (rec_points_earned / hist_data['receptions']).replace([np.inf, -np.inf], 0).fillna(0)

        player_records = []
        for pos, span in self.core.POSITION_WINDOWS.items():
            pos_df = hist_data[hist_data['position'] == pos].copy()
            print(f"[Debug] position={pos}: {len(pos_df)} rows")
            if pos_df.empty:
                continue

            pos_df['base_pass_share'] = pos_df.groupby('player_id')['player_pass_share'].transform(lambda x: x.ewm(span=span, min_periods=1).mean())
            pos_df['base_rush_share'] = pos_df.groupby('player_id')['player_rush_share'].transform(lambda x: x.ewm(span=span, min_periods=1).mean())
            pos_df['base_rec_share']  = pos_df.groupby('player_id')['player_rec_share'].transform(lambda x: x.ewm(span=span, min_periods=1).mean())
            pos_df['eff_pass'] = pos_df.groupby('player_id')['pts_per_pass_att'].transform(lambda x: x.ewm(span=span, min_periods=1).mean())
            pos_df['eff_rush'] = pos_df.groupby('player_id')['pts_per_rush_att'].transform(lambda x: x.ewm(span=span, min_periods=1).mean())
            pos_df['eff_rec']  = pos_df.groupby('player_id')['pts_per_rec'].transform(lambda x: x.ewm(span=span, min_periods=1).mean())

            latest = pos_df.groupby('player_id').last().reset_index()
            player_records.append(latest[[
                'player_id', 'player_name', 'position', 'team',
                'base_pass_share', 'base_rush_share', 'base_rec_share',
                'eff_pass', 'eff_rush', 'eff_rec'
            ]])

        if not player_records:
            raise RuntimeError("player_records is empty — no positions matched.")

        df_players = pd.concat(player_records, ignore_index=True)
        df_players = df_players.rename(columns={'team': 'recent_team'})
        print(f"[Debug] df_players rows: {len(df_players)}")

        # --- Normalize team codes ---
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
                c_share = match['base_rec_share']

                df_players.loc[df_players['player_name'] == injured_name,
                               ['base_pass_share', 'base_rush_share', 'base_rec_share']] = 0.0
                team_mask = (df_players['recent_team'] == team) & (df_players['player_name'] != injured_name)

                if details['pos'] in ['WR', 'TE']:
                    df_players.loc[team_mask & df_players['position'].isin(['WR', 'TE']), 'base_pass_share'] += p_share * 0.70
                    df_players.loc[team_mask & (df_players['position'] == 'RB'), 'base_pass_share'] += p_share * 0.30
                    df_players.loc[team_mask & df_players['position'].isin(['WR', 'TE']), 'base_rec_share']  += c_share * 0.70
                    df_players.loc[team_mask & (df_players['position'] == 'RB'), 'base_rec_share']  += c_share * 0.30
                elif details['pos'] == 'RB':
                    df_players.loc[team_mask & (df_players['position'] == 'RB'), 'base_rush_share'] += r_share * 0.80
                    df_players.loc[team_mask & df_players['position'].isin(['WR', 'TE']), 'base_rush_share'] += r_share * 0.20
                    df_players.loc[team_mask & df_players['position'].isin(['WR', 'TE']), 'base_rec_share']  += c_share * 0.55
                    df_players.loc[team_mask & (df_players['position'] == 'RB'), 'base_rec_share']  += c_share * 0.45

        # --- Merge with team volumes (left join + fallback) ---
        print(f"[Debug] df_players teams: {sorted(df_players['recent_team'].dropna().unique())}")
        print(f"[Debug] team_volumes teams: {sorted(team_volumes['recent_team'].dropna().unique())}")

        merged = pd.merge(df_players, team_volumes, on='recent_team', how='left')

        defaults = {
            'proj_plays': 64.0,
            'proj_team_pass': 38.4,
            'proj_team_rec': 24.2,
            'proj_team_rush': 25.6,
            'implied_total': 22.0,
        }
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

        print(f"[Debug] merged rows after team_volumes join: {len(merged)}")

        merged['proj_pass_volume'] = merged['proj_team_pass'] * merged['base_pass_share']
        merged['proj_rush_volume'] = merged['proj_team_rush'] * merged['base_rush_share']
        merged['proj_rec_volume']  = merged['proj_team_rec']  * merged['base_rec_share']

        merged['raw_projection'] = (
            merged['proj_pass_volume'] * merged['eff_pass']
            + merged['proj_rush_volume'] * merged['eff_rush']
            + merged['proj_rec_volume']  * merged['eff_rec']
        )
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

        # Drop players on bye (no opponent assigned)
        final_df = final_df[final_df['opponent_team'] != 'UNK'].copy()
        print(f"[Debug] after bye filter: {len(final_df)} rows")

        final_df['final_projection'] = final_df['raw_projection'] * final_df['dvp_multiplier']
        final_df.loc[(final_df['is_fav'] == True) & (final_df['position'] == 'RB'), 'final_projection'] *= 1.05
        final_df.loc[(final_df['is_fav'] == False) & (final_df['position'].isin(['WR', 'TE'])), 'final_projection'] *= 1.03
        final_df['final_projection'] = final_df['final_projection'].round(2)

        output_cols = ['player_name', 'position', 'recent_team', 'opponent_team',
                       'implied_total', 'final_projection']
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

                print(f"[Debug] output_df keys sample: {output_df['_key'].head(5).tolist()}")
                print(f"[Debug] dk_sal keys sample:    {dk_sal['_key'].head(5).tolist()}")

                matched = pd.merge(
                    output_df[['_key']].drop_duplicates(),
                    dk_sal[['_key', 'salary']].drop_duplicates('_key'),
                    on='_key', how='inner'
                )
                print(f"[Debug] salary key matches: {len(matched)}")

                if len(matched) == 0:
                    print("[Warning] No salary matches found. Returning projections without salaries.")
                    output_df = output_df.drop(columns=['_key'], errors='ignore')
                else:
                    output_df = pd.merge(output_df, dk_sal[['_key', 'salary']].drop_duplicates('_key'),
                                         on='_key', how='left')
                    output_df = output_df.drop(columns=['_key'])
                    output_df['value'] = ((output_df['final_projection'] / output_df['salary']) * 1000).round(2)
                    output_df = output_df.sort_values(by='value', ascending=False)

        return output_df