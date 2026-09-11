import pandas as pd
import numpy as np
import os
from engine_pipeline import DKProjectionPipeline
from redzone_engine import RZOpportunityEngine


class DKSimulatorDataPipeline:
    def __init__(self, stats_season, target_season, target_week, dk_salary_csv=None):
        self.stats_season = stats_season
        self.target_season = target_season
        self.week = target_week
        self.dk_salary_csv = dk_salary_csv

        self.base_pipeline = DKProjectionPipeline(
            stats_season=stats_season,
            target_season=target_season,
            target_week=target_week,
            dk_salary_csv=dk_salary_csv,
        )
        self.rz_engine = RZOpportunityEngine(season=stats_season)

    def calculate_player_variance(self):
        print("[Engine] Processing historical player standard deviations...")
        if not self.base_pipeline.core._loaded:
            self.base_pipeline.core.fetch_and_clean_data()

        hist_data = self.base_pipeline.core.master_weekly.copy()

        player_std = hist_data.groupby('player_id')['dk_points'].std().reset_index(name='historical_std')
        position_fallback_std = hist_data.groupby('position')['dk_points'].std().to_dict()
        player_pos = hist_data.groupby('player_id')['position'].last().reset_index()

        variance_df = pd.merge(player_std, player_pos, on='player_id', how='left')
        variance_df['historical_std'] = variance_df.apply(
            lambda row: position_fallback_std.get(row['position'], 7.5)
            if pd.isna(row['historical_std']) else row['historical_std'],
            axis=1
        )
        return variance_df[['player_id', 'historical_std']]

    def model_algorithmic_ownership(self, dataframe):
        print("[Engine] Calculating market consensus field ownership curves...")

        if dataframe is None or dataframe.empty:
            return dataframe
        if 'position' not in dataframe.columns or 'salary' not in dataframe.columns:
            dataframe['Ownership'] = 0.05
            return dataframe

        df = dataframe.copy()
        df['salary'] = pd.to_numeric(df['salary'], errors='coerce').fillna(0)
        df = df[df['salary'] > 0].copy()
        if df.empty:
            dataframe['Ownership'] = 0.05
            return dataframe

        # Value metric blends median projection and ceiling. Tournament fields
        # reward ceiling, but ownership still tracks value per dollar.
        df['value_metric'] = ((df['gpp_projection'] * 0.65 + df['ceiling_projection'] * 0.35) / df['salary']) * 1000

        pos_scales = {'QB': 0.12, 'RB': 0.28, 'WR': 0.45, 'TE': 0.15}
        pieces = []

        for pos, weight in pos_scales.items():
            pos_df = df[df['position'] == pos].copy()
            if pos_df.empty:
                continue

            pos_df['value_rank'] = pos_df['value_metric'].rank(ascending=False, method='min')
            decay = 0.85 ** (pos_df['value_rank'] - 1)
            pos_df['Ownership'] = (decay / decay.sum()) * weight

            max_cap = 0.45 if pos in ['RB', 'WR'] else 0.25
            pos_df['Ownership'] = pos_df['Ownership'].clip(lower=0.005, upper=max_cap)
            pieces.append(pos_df)

        if not pieces:
            dataframe['Ownership'] = 0.05
            return dataframe

        result = pd.concat(pieces, ignore_index=True)
        print(f"[Engine] Ownership range: {result['Ownership'].min():.4f}–{result['Ownership'].max():.4f}")
        return result

    def run_gpp_optimized_pipeline(self, injuries=None):
        base_projections = self.base_pipeline.run_full_pipeline(injured_players_dict=injuries)
        print(f"[Debug] base_projections rows: {len(base_projections)}")

        if base_projections is None or base_projections.empty:
            raise RuntimeError("base_projections is empty.")

        rz_matrix = self.rz_engine.extract_redzone_shares()
        variance_matrix = self.calculate_player_variance()

        hist_map = (
            self.base_pipeline.core.master_weekly
            .groupby('player_name')['player_id']
            .last()
            .reset_index()
        )

        final_df = pd.merge(base_projections, hist_map, on='player_name', how='left')
        print(f"[Debug] after hist_map merge: {len(final_df)} rows, "
              f"{final_df['player_id'].isna().sum()} missing player_id")

        final_df = pd.merge(final_df, rz_matrix, on='player_name', how='left')
        final_df[['rz_carry_share', 'rz_target_share']] = (
            final_df[['rz_carry_share', 'rz_target_share']].fillna(0.0)
        )

        final_df = pd.merge(final_df, variance_matrix, on='player_id', how='left')
        final_df['historical_std'] = final_df['historical_std'].fillna(7.5)

        # --- Touchdown equity applies to BOTH median and ceiling ---
        def apply_touchdown_equity(row):
            current_projection = row['final_projection']
            current_ceiling    = row['ceiling_projection']
            if row['position'] == 'RB' and row['rz_carry_share'] >= 0.40:
                return current_projection * 1.12, current_ceiling * 1.15
            if row['position'] in ['WR', 'TE'] and row['rz_target_share'] >= 0.25:
                return current_projection * 1.08, current_ceiling * 1.12
            return current_projection, current_ceiling

        touched = final_df.apply(apply_touchdown_equity, axis=1, result_type='expand')
        final_df['gpp_projection']   = touched[0].round(2)
        final_df['ceiling_projection'] = touched[1].round(2)

        # --- Ceiling-to-median ratio flags tournament leverage plays ---
        # A high ratio means the median is low but the ceiling is high — exactly
        # the players you want in a GPP but that the field under-rosters.
        final_df['ceiling_ratio'] = (final_df['ceiling_projection'] / final_df['gpp_projection'].replace(0, np.nan)).round(3)
        final_df['ceiling_ratio'] = final_df['ceiling_ratio'].fillna(1.0)

        final_df = self.model_algorithmic_ownership(final_df)

        if final_df is None or final_df.empty:
            raise RuntimeError("Ownership model returned an empty dataframe.")

        # --- Leverage score: high ceiling vs. low ownership = tournament gold ---
        final_df['leverage_score'] = (
            (final_df['ceiling_projection'] / final_df['ceiling_projection'].max())
            / final_df['Ownership'].replace(0, np.nan)
        ).round(3)
        final_df['leverage_score'] = final_df['leverage_score'].fillna(0)

        sim_input_cols = {
            'player_name': 'Player', 'position': 'Position', 'recent_team': 'Team',
            'opponent_team': 'Opponent', 'gpp_projection': 'Projection',
            'ceiling_projection': 'Ceiling',
            'ceiling_multiplier': 'Ceiling_Mult',
            'usage_stability': 'Usage_Stability',
            'ceiling_ratio': 'Ceiling_Ratio',
            'leverage_score': 'Leverage',
            'historical_std': 'StdDev', 'salary': 'Salary', 'Ownership': 'Ownership'
        }

        available_keys = [k for k in sim_input_cols.keys() if k in final_df.columns]
        sim_export = final_df[available_keys].rename(
            columns={k: v for k, v in sim_input_cols.items() if k in available_keys}
        )

        for col in ['Ownership', 'Leverage']:
            if col in sim_export.columns:
                sim_export[col] = sim_export[col].round(4)
        if 'StdDev' in sim_export.columns:
            sim_export['StdDev'] = sim_export['StdDev'].round(2)

        sim_export = sim_export.sort_values(by='Projection', ascending=False)

        csv_sim_filename = f"sim_input_projections_{self.target_season}_w{self.week}.csv"
        sim_export.to_csv(csv_sim_filename, index=False)
        print(f"\n--> SUCCESS! Clean simulation-ready file generated: {csv_sim_filename} "
              f"({len(sim_export)} players)")
        return sim_export


if __name__ == "__main__":
    model = DKSimulatorDataPipeline(
        stats_season=2025,
        target_season=2026,
        target_week=1,
        dk_salary_csv="DKSalaries.csv",
    )
    active_injuries = {}
    clean_sim_file = model.run_gpp_optimized_pipeline(injuries=active_injuries)
    print(clean_sim_file.head(10))