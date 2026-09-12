import pandas as pd
import numpy as np


class NFLSimConfigRunner:
    def __init__(self, projection_csv_path):
        self.csv_path = projection_csv_path
        self.player_pool = None

    def load_and_initialize_pool(self):
        print(f"[Sim Config] Ingesting source matrix: {self.csv_path}")
        try:
            self.player_pool = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not locate {self.csv_path}. Please run dfs_final_model.py first."
            )

        print(f"[Sim Config] Detected columns: {self.player_pool.columns.tolist()}")

        # --- Normalize column names: accept TitleCase or snake_case ---
        aliases = {
            'Player':          ['Player', 'player_name', 'name', 'full_name'],
            'Position':        ['Position', 'position', 'pos'],
            'Team':            ['Team', 'recent_team', 'team'],
            'Opponent':        ['Opponent', 'opponent_team', 'opp'],
            'Projection':      ['Projection', 'gpp_projection', 'final_projection', 'proj'],
            'Ceiling':         ['Ceiling', 'ceiling_projection'],
            'StdDev':          ['StdDev', 'historical_std', 'std_dev', 'stddev'],
            'Salary':          ['Salary', 'salary'],
            'Ownership':       ['Ownership', 'ownership'],
            'Leverage':        ['Leverage', 'leverage_score', 'leverage'],
            'Ceiling_Ratio':   ['Ceiling_Ratio', 'ceiling_ratio'],
            'Usage_Stability': ['Usage_Stability', 'usage_stability'],
            'AvgPointsPerGame':['AvgPointsPerGame', 'avg_points_per_game'],
        }

        rename_map = {}
        for canonical, candidates in aliases.items():
            if canonical in self.player_pool.columns:
                continue
            for cand in candidates:
                if cand in self.player_pool.columns:
                    rename_map[cand] = canonical
                    break
        if rename_map:
            print(f"[Sim Config] Renaming columns: {rename_map}")
            self.player_pool = self.player_pool.rename(columns=rename_map)

        # --- Verify required fields ---
        required = ['Player', 'Position', 'Team', 'Opponent',
                    'Projection', 'StdDev', 'Salary', 'Ownership']
        missing = [f for f in required if f not in self.player_pool.columns]
        if missing:
            raise ValueError(
                f"Missing required simulation fields after normalization: {missing}\n"
                f"Available columns: {self.player_pool.columns.tolist()}\n"
                f"Fix this by ensuring dfs_final_model.py exports a 'player_name' "
                f"and 'salary' column — see the [Debug] pre-export columns line "
                f"in the dfs_final_model.py run output."
            )

        # --- Coerce numerics ---
        for col in ['Projection', 'Ceiling', 'StdDev', 'Salary', 'Ownership',
                    'Leverage', 'Ceiling_Ratio', 'Usage_Stability', 'AvgPointsPerGame']:
            if col in self.player_pool.columns:
                self.player_pool[col] = pd.to_numeric(self.player_pool[col], errors='coerce').fillna(0.0)

        print(f"[Sim Config] Loaded {len(self.player_pool)} players.")
        return self.player_pool

    def get_game_stacks_matrix(self):
        if self.player_pool is None:
            self.load_and_initialize_pool()

        self.player_pool['Game_ID'] = self.player_pool.apply(
            lambda r: f"{min(r['Team'], r['Opponent'])}_{max(r['Team'], r['Opponent'])}",
            axis=1
        )
        return self.player_pool

    def execute_mock_simulation_trial(self, iterations=10000):
        df = self.get_game_stacks_matrix()
        print(f"[Sim Core] Running {iterations} iterations...")

        projections = df['Projection'].values
        std_devs = df['StdDev'].values

        np.random.seed(42)
        sim_results = np.random.normal(
            loc=projections[:, np.newaxis],
            scale=std_devs[:, np.newaxis],
            size=(len(projections), iterations)
        )
        sim_results = np.clip(sim_results, a_min=0, a_max=None)

        df['Simulated_Mean'] = np.mean(sim_results, axis=1).round(2)
        df['Simulated_Median'] = np.median(sim_results, axis=1).round(2)
        df['Simulated_85th'] = np.percentile(sim_results, 85, axis=1).round(2)
        df['Ceiling_95th'] = np.percentile(sim_results, 95, axis=1).round(2)
        df['Ceiling_99th'] = np.percentile(sim_results, 99, axis=1).round(2)
        df['Boom_Percentage'] = np.mean(sim_results >= 25, axis=1).round(4) * 100

        df = df.sort_values(by='Ceiling_95th', ascending=False)

        output_name = "sim_final_tournament_metrics.csv"
        df.to_csv(output_name, index=False)
        print(f"--> [Sim Complete] Results written to: {output_name}")
        return df


if __name__ == "__main__":
    runner = NFLSimConfigRunner(
        projection_csv_path="sim_input_projections_2026_w1.csv"
    )
    results = runner.execute_mock_simulation_trial(iterations=5000)
    print("\nTop 12 by 95th percentile ceiling:")
    display_cols = ['Player', 'Position', 'Projection', 'Ceiling',
                    'Ceiling_95th', 'Ceiling_99th', 'Boom_Percentage',
                    'Ownership', 'Leverage']
    display_cols = [c for c in display_cols if c in results.columns]
    print(results[display_cols].head(12))