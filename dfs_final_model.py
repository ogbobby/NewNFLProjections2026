import pandas as pd
import numpy as np
import os
from engine_pipeline import DKProjectionPipeline
from redzone_engine import RZOpportunityEngine


POSITION_CALIBRATION = {
    'QB': 0.80,
    'RB': 0.97,
    'WR': 0.85,
    'TE': 0.88,
}

POSITION_CAPS = {
    'QB': 26.0,
    'RB': 32.0,
    'WR': 30.0,
    'TE': 24.0,
}

MANUAL_ADJUSTMENTS_FILE = 'manual_adjustments.csv'


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

    def _prune_to_starters(self, dataframe):
        """Keep only plausible starters per team by salary rank."""
        if 'salary' not in dataframe.columns:
            print("[Engine] No salary column — skipping starter pruning.")
            return dataframe

        df = dataframe.copy()
        df['salary'] = pd.to_numeric(df['salary'], errors='coerce').fillna(0)
        pos_limits = {'QB': 1, 'RB': 2, 'WR': 5, 'TE': 2}
        keep_pieces = []

        for pos, limit in pos_limits.items():
            pos_df = df[df['position'] == pos].copy()
            if pos_df.empty:
                continue
            pos_df['team_rank'] = pos_df.groupby('recent_team')['salary'].rank(
                ascending=False, method='first'
            )
            before = len(pos_df)
            pos_df = pos_df[pos_df['team_rank'] <= limit].drop(columns=['team_rank'])
            print(f"[Engine] Pruned {pos}: {before} → {len(pos_df)}")
            keep_pieces.append(pos_df)

        if not keep_pieces:
            return df
        return pd.concat(keep_pieces, ignore_index=True)

    def _apply_backup_rb_discount(self, dataframe):
        """
        For teams with multiple RBs in the pool, discount backups based on
        salary rank. RB2 gets 0.65×, RB3 gets 0.45×. RB1 is untouched.
        """
        print("[Engine] Applying backup RB discount...")
        df = dataframe.copy()
        rb_mask = df['position'] == 'RB'
        if rb_mask.sum() == 0:
            return df

        df.loc[rb_mask, 'rb_team_rank'] = df.loc[rb_mask].groupby('recent_team')['salary'].rank(
            ascending=False, method='first'
        )

        multipliers = {1: 1.00, 2: 0.65, 3: 0.45}
        n_discounted = 0
        for rank, mult in multipliers.items():
            if mult == 1.0:
                continue
            mask = rb_mask & (df['rb_team_rank'] == rank)
            count = mask.sum()
            if count > 0:
                df.loc[mask, 'gpp_projection']     = (df.loc[mask, 'gpp_projection']     * mult).round(2)
                df.loc[mask, 'ceiling_projection'] = (df.loc[mask, 'ceiling_projection'] * mult).round(2)
                n_discounted += count

        df = df.drop(columns=['rb_team_rank'])
        print(f"[Engine] Discounted {n_discounted} backup RBs.")
        return df

    def _apply_manual_adjustments(self, dataframe):
        """
        Read manual_adjustments.csv and apply per-player multipliers.
        File format: player_name, projection_mult, ceiling_mult, notes
        Missing multipliers are treated as 1.0.
        """
        if not os.path.exists(MANUAL_ADJUSTMENTS_FILE):
            print(f"[Engine] No {MANUAL_ADJUSTMENTS_FILE} found — skipping manual overrides.")
            return dataframe

        print(f"[Engine] Applying manual overrides from {MANUAL_ADJUSTMENTS_FILE}...")
        adj = pd.read_csv(MANUAL_ADJUSTMENTS_FILE)
        required = {'player_name', 'projection_mult', 'ceiling_mult'}
        if not required.issubset(adj.columns):
            print(f"[Warning] {MANUAL_ADJUSTMENTS_FILE} missing required columns. Found: {adj.columns.tolist()}")
            return dataframe

        df = dataframe.copy()
        n_applied = 0
        n_unmatched = 0

        for _, row in adj.iterrows():
            name = str(row['player_name']).strip()
            mask = df['player_name'].astype(str).str.strip() == name
            if mask.sum() == 0:
                print(f"  [Warning] Manual override for '{name}' did not match any player.")
                n_unmatched += 1
                continue

            proj_mult = float(row['projection_mult']) if pd.notna(row['projection_mult']) else 1.0
            ceil_mult = float(row['ceiling_mult']) if pd.notna(row['ceiling_mult']) else 1.0

            df.loc[mask, 'gpp_projection']     = (df.loc[mask, 'gpp_projection']     * proj_mult).round(2)
            df.loc[mask, 'ceiling_projection'] = (df.loc[mask, 'ceiling_projection'] * ceil_mult).round(2)

            note = row.get('notes', '')
            print(f"  {name}: proj × {proj_mult}, ceil × {ceil_mult}{'  (' + str(note) + ')' if note else ''}")
            n_applied += 1

        print(f"[Engine] Applied {n_applied} overrides ({n_unmatched} unmatched).")
        return df

    def model_algorithmic_ownership(self, dataframe):
        print("[Engine] Calculating market-consensus field ownership curves...")

        if dataframe is None or dataframe.empty:
            return dataframe
        if 'position' not in dataframe.columns or 'salary' not in dataframe.columns:
            dataframe['Ownership'] = 0.005
            return dataframe

        df = dataframe.copy()
        df['salary'] = pd.to_numeric(df['salary'], errors='coerce').fillna(0)
        df = df[df['salary'] > 0].copy()
        if df.empty:
            dataframe['Ownership'] = 0.005
            return dataframe

        if 'AvgPointsPerGame' in df.columns:
            df['AvgPointsPerGame'] = pd.to_numeric(df['AvgPointsPerGame'], errors='coerce').fillna(0)
            df['market_signal'] = df['AvgPointsPerGame'].clip(lower=0)
        else:
            df['market_signal'] = df['salary']

        df['model_value'] = ((df['gpp_projection'] * 0.70 + df['ceiling_projection'] * 0.30)
                             / df['salary']) * 1000

        df['market_signal_norm'] = df.groupby('position')['market_signal'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
        )
        df['model_value_norm'] = df.groupby('position')['model_value'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
        )

        MARKET_WEIGHT = 0.40
        df['blended_bias'] = ((1 - MARKET_WEIGHT) * df['model_value_norm'] +
                              MARKET_WEIGHT * df['market_signal_norm']) + 0.05

        pos_scales = {'QB': 0.12, 'RB': 0.28, 'WR': 0.45, 'TE': 0.15}
        pieces = []

        for pos, weight in pos_scales.items():
            pos_df = df[df['position'] == pos].copy()
            if pos_df.empty:
                continue

            pos_df['value_rank']  = pos_df['model_value'].rank(ascending=False, method='min')
            pos_df['market_rank'] = pos_df['market_signal'].rank(ascending=False, method='min')
            pos_df['blended_rank'] = 0.60 * pos_df['value_rank'] + 0.40 * pos_df['market_rank']

            decay = 0.75 ** (pos_df['blended_rank'] - 1)
            pos_df['Ownership'] = (decay / decay.sum()) * weight

            max_cap = 0.35 if pos in ['RB', 'WR'] else 0.20
            pos_df['Ownership'] = pos_df['Ownership'].clip(upper=max_cap)

            excess = weight - pos_df['Ownership'].sum()
            if excess > 0:
                uncapped = pos_df[pos_df['Ownership'] < max_cap]
                if not uncapped.empty:
                    pos_df.loc[uncapped.index, 'Ownership'] += (
                        (uncapped['Ownership'] / uncapped['Ownership'].sum()) * excess
                    )

            pieces.append(pos_df)

        if not pieces:
            dataframe['Ownership'] = 0.005
            return dataframe

        result = pd.concat(pieces, ignore_index=True)

        print("[Engine] Ownership sum by position:")
        for pos in ['QB', 'RB', 'WR', 'TE']:
            sub = result[result['position'] == pos]['Ownership']
            if not sub.empty:
                print(f"  {pos}: sum={sub.sum():.4f}, max={sub.max():.4f}, median={sub.median():.5f}")

        total = result['Ownership'].sum()
        print(f"[Engine] Total ownership sum: {total:.4f}")
        return result

    def _apply_qb_market_prior(self, final_df):
        qb_mask = final_df['position'] == 'QB'
        if qb_mask.sum() == 0 or 'salary' not in final_df.columns:
            return final_df

        qb_salaries = final_df.loc[qb_mask, 'salary'].fillna(5000)
        sal_min, sal_max = qb_salaries.min(), qb_salaries.max()
        if sal_max > sal_min:
            qb_norm = (qb_salaries - sal_min) / (sal_max - sal_min)
        else:
            qb_norm = pd.Series(0.5, index=qb_salaries.index)

        market_proj = 12.0 + qb_norm * 14.0
        MARKET_WEIGHT_QB = 0.40

        print("[Engine] Applying QB market prior (40% salary-implied blend)...")
        final_df.loc[qb_mask, 'gpp_projection'] = (
            (1 - MARKET_WEIGHT_QB) * final_df.loc[qb_mask, 'gpp_projection']
            + MARKET_WEIGHT_QB * market_proj
        ).round(2)
        final_df.loc[qb_mask, 'ceiling_projection'] = (
            (1 - MARKET_WEIGHT_QB) * final_df.loc[qb_mask, 'ceiling_projection']
            + MARKET_WEIGHT_QB * market_proj * 1.45
        ).round(2)
        return final_df

    def _apply_soft_caps(self, final_df):
        print("[Engine] Applying soft position caps...")
        for pos, cap in POSITION_CAPS.items():
            mask = (final_df['position'] == pos) & (final_df['gpp_projection'] > cap)
            if mask.any():
                excess = final_df.loc[mask, 'gpp_projection'] - cap
                final_df.loc[mask, 'gpp_projection'] = (cap + np.log1p(excess) * 2.0).round(2)

            ceil_cap = cap * 1.55
            mask_c = (final_df['position'] == pos) & (final_df['ceiling_projection'] > ceil_cap)
            if mask_c.any():
                excess = final_df.loc[mask_c, 'ceiling_projection'] - ceil_cap
                final_df.loc[mask_c, 'ceiling_projection'] = (ceil_cap + np.log1p(excess) * 2.5).round(2)

            n_proj = mask.sum()
            n_ceil = mask_c.sum()
            if n_proj or n_ceil:
                print(f"  {pos}: capped {n_proj} projections, {n_ceil} ceilings")
        return final_df

    def run_gpp_optimized_pipeline(self, injuries=None):
        base_projections = self.base_pipeline.run_full_pipeline(injured_players_dict=injuries)
        print(f"[Debug] base_projections rows: {len(base_projections)}")

        if base_projections is None or base_projections.empty:
            raise RuntimeError("base_projections is empty.")

        if 'player_name' not in base_projections.columns:
            for alt in ['Player', 'name', 'full_name', 'player', 'player_display_name']:
                if alt in base_projections.columns:
                    base_projections = base_projections.rename(columns={alt: 'player_name'})
                    break
            else:
                raise RuntimeError(f"No player-name column. Available: {base_projections.columns.tolist()}")

        rz_matrix = self.rz_engine.extract_redzone_shares()
        variance_matrix = self.calculate_player_variance()

        hist_map = (
            self.base_pipeline.core.master_weekly
            .groupby('player_name')['player_id']
            .last()
            .reset_index()
        )

        if 'player_id' in base_projections.columns:
            base_projections = base_projections.drop(columns=['player_id'])

        final_df = pd.merge(base_projections, hist_map, on='player_name', how='left')

        if 'player_name' not in final_df.columns:
            for c in final_df.columns:
                if c.startswith('player_name_'):
                    final_df = final_df.rename(columns={c: 'player_name'})
                    break

        final_df = pd.merge(final_df, rz_matrix, on='player_name', how='left')
        final_df[['rz_carry_share', 'rz_target_share']] = (
            final_df[['rz_carry_share', 'rz_target_share']].fillna(0.0)
        )

        final_df = pd.merge(final_df, variance_matrix, on='player_id', how='left')
        final_df['historical_std'] = final_df['historical_std'].fillna(7.5)

        def apply_touchdown_equity(row):
            proj = row['final_projection']
            ceil = row['ceiling_projection']
            if row['position'] == 'RB' and row['rz_carry_share'] >= 0.40:
                return proj * 1.12, ceil * 1.15
            if row['position'] in ['WR', 'TE'] and row['rz_target_share'] >= 0.25:
                return proj * 1.08, ceil * 1.12
            return proj, ceil

        touched = final_df.apply(apply_touchdown_equity, axis=1, result_type='expand')
        final_df['gpp_projection']   = touched[0].round(2)
        final_df['ceiling_projection'] = touched[1].round(2)

        final_df['ceiling_ratio'] = (final_df['ceiling_projection'] / final_df['gpp_projection'].replace(0, np.nan)).round(3)
        final_df['ceiling_ratio'] = final_df['ceiling_ratio'].fillna(1.0)

        # --- Salary merge ---
        if self.dk_salary_csv and os.path.exists(self.dk_salary_csv):
            dk_sal_raw = pd.read_csv(self.dk_salary_csv)
            keep_cols = [c for c in ['Name', 'Salary', 'AvgPointsPerGame'] if c in dk_sal_raw.columns]
            if 'Name' not in keep_cols or 'Salary' not in keep_cols:
                print("[Warning] DK CSV missing 'Name' or 'Salary'.")
            else:
                dk_sal = dk_sal_raw[keep_cols].copy()
                dk_sal = dk_sal.rename(columns={'Name': 'player_name', 'Salary': 'salary'})
                dk_sal['player_name'] = dk_sal['player_name'].astype(str).str.strip()
                final_df['player_name'] = final_df['player_name'].astype(str).str.strip()

                def norm(n):
                    return (n.replace('.', '').replace("'", '').replace('-', ' ')
                             .replace(' Jr', '').replace(' Sr', '').replace(' III', '')
                             .lower().strip())

                final_df['_key'] = final_df['player_name'].map(norm)
                dk_sal['_key']    = dk_sal['player_name'].map(norm)

                for c in ['salary', 'AvgPointsPerGame']:
                    if c in final_df.columns:
                        final_df = final_df.drop(columns=[c])

                matched = pd.merge(
                    final_df[['_key']].drop_duplicates(),
                    dk_sal[['_key']].drop_duplicates('_key'),
                    on='_key', how='inner'
                )
                print(f"[Debug] salary key matches: {len(matched)}")

                if len(matched) > 0:
                    merge_cols = ['_key', 'salary'] + (['AvgPointsPerGame'] if 'AvgPointsPerGame' in dk_sal.columns else [])
                    final_df = pd.merge(final_df, dk_sal[merge_cols].drop_duplicates('_key'),
                                        on='_key', how='left')
                final_df = final_df.drop(columns=['_key'], errors='ignore')

        if 'player_name' not in final_df.columns:
            for c in final_df.columns:
                if c.startswith('player_name_'):
                    final_df = final_df.rename(columns={c: 'player_name'})
                    break

        # ============================================================
        # FIX A — prune to starters by salary
        # ============================================================
        print(f"[Engine] Player pool before pruning: {len(final_df)}")
        final_df = self._prune_to_starters(final_df)
        print(f"[Engine] Player pool after pruning: {len(final_df)}")

        # ============================================================
        # POSITION CALIBRATION
        # ============================================================
        print("[Engine] Applying position calibration...")
        for pos, mult in POSITION_CALIBRATION.items():
            mask = final_df['position'] == pos
            final_df.loc[mask, 'gpp_projection']     = (final_df.loc[mask, 'gpp_projection'] * mult).round(2)
            final_df.loc[mask, 'ceiling_projection'] = (final_df.loc[mask, 'ceiling_projection'] * mult).round(2)

        # ============================================================
        # FIX B — QB market prior
        # ============================================================
        final_df = self._apply_qb_market_prior(final_df)

        # ============================================================
        # FIX C — soft position caps
        # ============================================================
        final_df = self._apply_soft_caps(final_df)

        # ============================================================
        # FIX D — backup RB discount (automatic)
        # ============================================================
        final_df = self._apply_backup_rb_discount(final_df)

        # ============================================================
        # FIX E — manual overrides (from CSV)
        # ============================================================
        final_df = self._apply_manual_adjustments(final_df)

        # ============================================================
        # STDDEV clip
        # ============================================================
        print("[Engine] Clipping standard deviations...")
        final_df['historical_std'] = np.where(
            final_df['gpp_projection'] > 0,
            np.clip(final_df['historical_std'],
                    final_df['gpp_projection'] * 0.30,
                    final_df['gpp_projection'] * 0.55),
            final_df['historical_std']
        )
        print(f"[Debug] StdDev range after clip: {final_df['historical_std'].min():.2f}–{final_df['historical_std'].max():.2f}")

        # ============================================================
        # OWNERSHIP
        # ============================================================
        final_df = self.model_algorithmic_ownership(final_df)

        if final_df is None or final_df.empty:
            raise RuntimeError("Ownership model returned empty.")

        final_df['leverage_score'] = (
            (final_df['ceiling_projection'] / final_df['ceiling_projection'].max())
            / final_df['Ownership'].replace(0, np.nan)
        ).round(3)
        final_df['leverage_score'] = final_df['leverage_score'].fillna(0)

        # ============================================================
        # EXPORT
        # ============================================================
        sim_input_cols = {
            'player_name': 'Player', 'position': 'Position', 'recent_team': 'Team',
            'opponent_team': 'Opponent', 'gpp_projection': 'Projection',
            'ceiling_projection': 'Ceiling',
            'ceiling_multiplier': 'Ceiling_Mult',
            'usage_stability': 'Usage_Stability',
            'ceiling_ratio': 'Ceiling_Ratio',
            'leverage_score': 'Leverage',
            'historical_std': 'StdDev', 'salary': 'Salary',
            'AvgPointsPerGame': 'AvgPointsPerGame',
            'Ownership': 'Ownership'
        }

        available_keys = [k for k in sim_input_cols.keys() if k in final_df.columns]
        sim_export = final_df[available_keys].rename(
            columns={k: v for k, v in sim_input_cols.items() if k in available_keys}
        )

        if 'Player' not in sim_export.columns:
            raise RuntimeError(f"Export missing Player.")

        for col in ['Ownership', 'Leverage']:
            if col in sim_export.columns:
                sim_export[col] = sim_export[col].round(4)
        if 'StdDev' in sim_export.columns:
            sim_export['StdDev'] = sim_export['StdDev'].round(2)

        sim_export = sim_export.sort_values(by='Projection', ascending=False)

        csv_sim_filename = f"sim_input_projections_{self.target_season}_w{self.week}.csv"
        sim_export.to_csv(csv_sim_filename, index=False)
        print(f"\n--> SUCCESS! {csv_sim_filename} ({len(sim_export)} players)")
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