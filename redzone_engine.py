import pandas as pd
import nflreadpy


class RZOpportunityEngine:
    def __init__(self, season, week=None):
        self.season = season
        self.week = week

    def extract_redzone_shares(self):
        print(f"[RZ Engine] Analyzing red-zone play-by-play from {self.season}...")

        pbp_raw = nflreadpy.load_pbp([self.season])
        pbp = pbp_raw.to_pandas() if hasattr(pbp_raw, "to_pandas") else pd.DataFrame(pbp_raw)

        pbp = pbp[
            (pbp['season_type'] == 'REG') &
            (pbp['yardline_100'] <= 10)
        ].copy()

        pbp['is_rz_carry'] = (pbp['rush_attempt'] == 1) & (pbp['qb_kneel'] == 0)
        pbp['is_rz_target'] = (pbp['pass_attempt'] == 1) & (pbp['sack'] == 0)

        # --- Name lookup from rosters ---
        try:
            roster_raw = nflreadpy.load_rosters([self.season])
            roster = roster_raw.to_pandas() if hasattr(roster_raw, "to_pandas") else pd.DataFrame(roster_raw)
            name_cols = [c for c in ['football_name', 'player_name', 'short_name'] if c in roster.columns]
            full_col = next((c for c in ['full_name', 'player_display_name'] if c in roster.columns), None)
            if name_cols and full_col:
                lookup = (roster[[name_cols[0], full_col]]
                          .dropna()
                          .drop_duplicates(name_cols[0])
                          .set_index(name_cols[0])[full_col]
                          .to_dict())
            else:
                lookup = {}
        except Exception as e:
            print(f"[Warning] Could not load rosters: {e}")
            lookup = {}

        def to_full(name):
            if pd.isna(name):
                return name
            return lookup.get(name, name)

        team_carries = (pbp[pbp['is_rz_carry']]
                        .groupby(['posteam', 'week']).size()
                        .reset_index(name='team_rz_carries'))
        team_targets = (pbp[pbp['is_rz_target']]
                        .groupby(['posteam', 'week']).size()
                        .reset_index(name='team_rz_targets'))

        rz_carry = pbp[pbp['is_rz_carry']].copy()
        rz_carry['rz_player'] = rz_carry['rusher_player_name'].map(to_full)

        rz_target = pbp[pbp['is_rz_target']].copy()
        rz_target['rz_player'] = rz_target['receiver_player_name'].map(to_full)

        player_carries = (rz_carry
                          .groupby(['rz_player', 'posteam', 'week']).size()
                          .reset_index(name='player_rz_carries'))
        player_targets = (rz_target
                          .groupby(['rz_player', 'posteam', 'week']).size()
                          .reset_index(name='player_rz_targets'))

        carry_merge = pd.merge(player_carries, team_carries, on=['posteam', 'week'], how='left')
        carry_merge['rz_carry_share'] = carry_merge['player_rz_carries'] / carry_merge['team_rz_carries']

        target_merge = pd.merge(player_targets, team_targets, on=['posteam', 'week'], how='left')
        target_merge['rz_target_share'] = target_merge['player_rz_targets'] / target_merge['team_rz_targets']

        rz_rush_profile = (carry_merge.groupby('rz_player')['rz_carry_share']
                           .mean().reset_index())
        rz_rec_profile = (target_merge.groupby('rz_player')['rz_target_share']
                          .mean().reset_index())

        rz_master = pd.merge(rz_rush_profile, rz_rec_profile,
                             on='rz_player', how='outer').fillna(0.0)
        rz_master.columns = ['player_name', 'rz_carry_share', 'rz_target_share']

        print(f"[RZ Engine] Extracted profiles for {len(rz_master)} skill assets.")
        return rz_master


if __name__ == "__main__":
    extractor = RZOpportunityEngine(season=2025)
    print(extractor.extract_redzone_shares().head(10))