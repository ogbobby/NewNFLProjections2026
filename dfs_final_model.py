import pandas as pd
import numpy as np
import os
from engine_pipeline import DKProjectionPipeline
from redzone_engine import RZOpportunityEngine


class DKSimulatorDataPipeline:
    def __init__(self, stats_season, target_season, target_week, dk_salary_csv=None):
        """
        stats_season:  year to pull historical stats from (e.g. 2025)
        target_season: year of the slate you're building (e.g. 2026)
        target_week:   week of the slate (e.g. 1)
        """
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
            print("[Warning] Ownership model received an empty dataframe. Skipping.")
            return dataframe

        if 'position' not in dataframe.columns or 'salary' not in dataframe.columns:
            print("[Warning] Missing 'position' or 'salary'. Assigning flat 5% ownership.")
            dataframe['Ownership'] = 0.05
            return dataframe

        df = dataframe.copy()
        df['salary'] = pd.to_numeric(df['salary'], errors='coerce').fillna(0)
        df = df[df['salary'] > 0].copy()
        if df.empty:
            print("[Warning] All salaries were zero. Assigning flat 5% ownership.")
            dataframe['Ownership'] = 0.05
            return dataframe

        df['value_metric'] = (df['gpp_projection'] / df['salary']) * 1000
        df['raw_bias'] = df['value_metric'].clip(lower=0) ** 1.85

        pos_scales = {'QB': 0.12, 'RB': 0.28, 'WR': 0.45, 'TE': 0.15}
        final_ownership_list = []

        for pos, weight in pos_scales.items():
            pos_df = df[df['position'] == pos].copy()
            if pos_df.empty:
                print(f"[Warning] No players at position {pos}; skipping.")
                continue

            total_bias = pos_df['raw_bias'].sum()
            if total_bias > 0:
                pos_df['Ownership'] = (pos_df['raw_bias'] / total_bias) * weight
            else:
                pos_df['Ownership'] = weight / len(pos_df)

            max_cap = 0.48 if pos in ['RB', 'WR'] else 0.28
            pos_df['Ownership'] = pos_df['Ownership'].clip(lower=0.01, upper=max_cap)
            final_ownership_list.append(pos_df)

        if not final_ownership_list:
            print("[Warning] Ownership produced no rows. Falling back to flat 5%.")
            dataframe['Ownership'] = 0.05
            return dataframe

        return pd.concat(final_ownership_list, ignore_index=True)

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

        def apply_touchdown_equity(row):
            current_projection = row['final_projection']
            if row['position'] == 'RB' and row['rz_carry_share'] >= 0.40:
                return current_projection * 1.12
            if row['position'] in ['WR', 'TE'] and row['rz_target_share'] >= 0.25:
                return current_projection * 1.08
            return current_projection

        final_df['gpp_projection'] = final_df.apply(apply_touchdown_equity, axis=1).round(2)
        final_df = self.model_algorithmic_ownership(final_df)

        if final_df is None or final_df.empty:
            raise RuntimeError("Ownership model returned an empty dataframe.")

        sim_input_cols = {
            'player_name': 'Player', 'position': 'Position', 'recent_team': 'Team',
            'opponent_team': 'Opponent', 'gpp_projection': 'Projection',
            'historical_std': 'StdDev', 'salary': 'Salary', 'Ownership': 'Ownership'
        }

        available_keys = [k for k in sim_input_cols.keys() if k in final_df.columns]
        sim_export = final_df[available_keys].rename(
            columns={k: v for k, v in sim_input_cols.items() if k in available_keys}
        )

        sim_export['Ownership'] = sim_export['Ownership'].round(4)
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