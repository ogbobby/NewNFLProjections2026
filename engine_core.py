import pandas as pd
import numpy as np
import nflreadpy


class DKCoreDataEngine:
    def __init__(self, stats_season, target_season, target_week):
        self.stats_season = stats_season      # year to pull historical stats from
        self.target_season = target_season    # year of the slate you're building
        self.week = target_week               # week of the slate
        self.POSITION_WINDOWS = {'QB': 6, 'RB': 6, 'WR': 3, 'TE': 3}
        self.master_weekly = None
        self.schedule = None
        self._loaded = False

    def fetch_and_clean_data(self):
        if self._loaded:
            return
        print(f"[Core] Pulling {self.stats_season} stats and {self.target_season} schedule...")
        weekly_raw = nflreadpy.load_player_stats([self.stats_season])
        try:
            schedule_raw = nflreadpy.load_schedules([self.target_season])
        except Exception as e:
            print(f"[Warning] Could not load {self.target_season} schedule: {e}")
            schedule_raw = pd.DataFrame()

        self.master_weekly = weekly_raw.to_pandas() if hasattr(weekly_raw, "to_pandas") else pd.DataFrame(weekly_raw)
        self.schedule = schedule_raw.to_pandas() if hasattr(schedule_raw, "to_pandas") else pd.DataFrame(schedule_raw)

        df = self.master_weekly
        df = df[df['position'].isin(['QB', 'RB', 'WR', 'TE'])].copy()

        # --- Use full display name so joins against DK work ---
        if 'player_display_name' in df.columns:
            df['player_name'] = df['player_display_name']
        elif 'full_name' in df.columns:
            df['player_name'] = df['full_name']

        # --- Normalize column names ---
        rename_map = {
            'passing_attempts': 'pass_attempts',
            'attempts':         'pass_attempts',
            'carries':          'rush_attempts',
            'rushing_attempts': 'rush_attempts',
            'interceptions':    'passing_interceptions',
        }
        df = df.rename(columns={k: v for k, v in rename_map.items()
                                if k in df.columns and v not in df.columns})

        # --- Safe numeric fallbacks ---
        for col in ['pass_attempts', 'rush_attempts', 'receptions',
                    'passing_yards', 'passing_tds', 'passing_interceptions',
                    'rushing_yards', 'rushing_tds',
                    'receiving_yards', 'receiving_tds',
                    'sack_fumbles_lost', 'rushing_fumbles_lost', 'receiving_fumbles_lost']:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # --- Team denominators for shares ---
        df['team_pass_attempts'] = df.groupby(['team', 'week'])['pass_attempts'].transform('sum').replace(0, np.nan)
        df['team_rush_attempts'] = df.groupby(['team', 'week'])['rush_attempts'].transform('sum').replace(0, np.nan)
        df['team_receptions']    = df.groupby(['team', 'week'])['receptions'].transform('sum').replace(0, np.nan)

        # --- DK scoring ---
        pass_pts = df['passing_yards'] * 0.04 + df['passing_tds'] * 4.0 - df['passing_interceptions'] * 1.0
        rush_pts = df['rushing_yards'] * 0.1 + df['rushing_tds'] * 6.0
        rec_pts  = df['receptions'] * 1.0 + df['receiving_yards'] * 0.1 + df['receiving_tds'] * 6.0
        fumbles_lost = (df['sack_fumbles_lost'] + df['rushing_fumbles_lost'] + df['receiving_fumbles_lost'])

        df['dk_points'] = pass_pts + rush_pts + rec_pts + (fumbles_lost * -2.0)
        self.master_weekly = df
        self._loaded = True

    def calculate_matchup_dvp(self):
        print("[Core] Calculating Defense vs Position (DvP) indices...")
        hist_data = self.master_weekly.copy()
        position_avg = hist_data.groupby('position')['dk_points'].mean().to_dict()
        def_allowed = hist_data.groupby(['opponent_team', 'position'])['dk_points'].mean().reset_index()
        def_allowed.columns = ['defense_team', 'position', 'avg_points_allowed']
        def_allowed['dvp_multiplier'] = def_allowed.apply(
            lambda r: r['avg_points_allowed'] / position_avg[r['position']] if position_avg[r['position']] > 0 else 1.0,
            axis=1
        )
        return def_allowed

    def project_macro_team_volume(self):
        print("[Core] Extracting Vegas totals and playbook profiles...")

        # --- Try the schedule for the target season/week ---
        sched = self.schedule.copy() if self.schedule is not None else pd.DataFrame()
        if not sched.empty and 'week' in sched.columns:
            sched = sched[sched['week'] == self.week].copy()
            type_col = next((c for c in ('season_type', 'game_type') if c in sched.columns), None)
            if type_col:
                sched = sched[sched[type_col] == 'REG'].copy()

        vegas_rows = []
        if not sched.empty:
            for _, g in sched.iterrows():
                total = g.get('total_line', 44.0)
                total = 44.0 if pd.isna(total) else total
                spread = g.get('spread_line', 0.0)
                spread = 0.0 if pd.isna(spread) else spread
                home, away = g['home_team'], g['away_team']
                home_implied = total / 2 - spread / 2
                away_implied = total / 2 + spread / 2
                vegas_rows.append({'team': home, 'opponent_team': away,
                                   'implied_total': home_implied, 'is_fav': spread < 0})
                vegas_rows.append({'team': away, 'opponent_team': home,
                                   'implied_total': away_implied, 'is_fav': spread > 0})
        else:
            # --- Fallback: read matchups from the DK salary file's Game Info ---
            if getattr(self, 'dk_salary_csv', None) and pd.io.common.file_exists(self.dk_salary_csv):
                dk = pd.read_csv(self.dk_salary_csv)
                if 'Game Info' in dk.columns and 'TeamAbbrev' in dk.columns:
                    matchups = dk[['Game Info', 'TeamAbbrev']].drop_duplicates()
                    for _, r in matchups.iterrows():
                        game = r['Game Info'].split(' ')[0]
                        if '@' in game:
                            away, home = game.split('@')
                            vegas_rows.append({'team': home, 'opponent_team': away,
                                               'implied_total': 22.0, 'is_fav': False})
                            vegas_rows.append({'team': away, 'opponent_team': home,
                                               'implied_total': 22.0, 'is_fav': False})
                    # Dedupe by team
                    if vegas_rows:
                        vegas_rows = pd.DataFrame(vegas_rows).drop_duplicates('team').to_dict('records')

        if not vegas_rows:
            print(f"[Warning] No schedule found for {self.target_season} week {self.week}; using defaults.")
            teams = self.master_weekly['team'].unique()
            return pd.DataFrame({
                'recent_team': teams,
                'opponent_team': ['UNK'] * len(teams),
                'implied_total': 22.0,
                'proj_plays': 64.0,
                'proj_team_pass': 38.4,
                'proj_team_rec': 24.2,
                'proj_team_rush': 25.6,
                'is_fav': False,
            })

        vegas = pd.DataFrame(vegas_rows)
        vegas['v_mod'] = vegas['implied_total'] / 22.0

        # --- Historical play-volume + pass-rate baselines (stats_season) ---
        hist = self.master_weekly.copy()
        team_games = (hist.groupby(['team', 'week'])
                      .agg(pass_att=('pass_attempts', 'sum'),
                           rush_att=('rush_attempts', 'sum'))
                      .reset_index())
        team_games['plays'] = team_games['pass_att'] + team_games['rush_att']
        team_games['pass_rate'] = team_games['pass_att'] / team_games['plays'].replace(0, pd.NA)

        baselines = (team_games.groupby('team')
                     .agg(avg_plays=('plays', 'mean'),
                          pass_rate_identity=('pass_rate', 'mean'))
                     .reset_index())

        volume = pd.merge(baselines, vegas, on='team', how='right')
        volume['avg_plays'] = volume['avg_plays'].fillna(64.0)
        volume['pass_rate_identity'] = volume['pass_rate_identity'].fillna(0.60)

        volume['proj_plays'] = volume['avg_plays'] * (1 + (volume['v_mod'] - 1) * 0.3)
        volume['proj_team_pass'] = volume['proj_plays'] * volume['pass_rate_identity']
        volume['proj_team_rec']  = volume['proj_team_pass'] * 0.63
        volume['proj_team_rush'] = volume['proj_plays'] * (1 - volume['pass_rate_identity'])

        volume = volume.rename(columns={'team': 'recent_team'})
        return volume[['recent_team', 'opponent_team', 'implied_total',
                       'proj_plays', 'proj_team_pass', 'proj_team_rec',
                       'proj_team_rush', 'is_fav']]