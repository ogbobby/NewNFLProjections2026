import pandas as pd
import numpy as np
import nflreadpy


class DKCoreDataEngine:
    def __init__(self, stats_season, target_season, target_week):
        self.stats_season = stats_season
        self.target_season = target_season
        self.week = target_week
        self.POSITION_WINDOWS = {'QB': 6, 'RB': 6, 'WR': 3, 'TE': 3}
        self.master_weekly = None
        self.schedule = None
        self.dk_salary_csv = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Team-code normalization used across the engine
    # ------------------------------------------------------------------
    TEAM_ALIASES = {
        'LA': 'LAR', 'STL': 'LAR', 'SL': 'LAR',
        'SD': 'LAC', 'OAK': 'LV', 'LVR': 'LV',
        'WSH': 'WAS', 'JAC': 'JAX', 'ARZ': 'ARI',
        'BLT': 'BAL', 'CLV': 'CLE', 'HST': 'HOU',
    }

    def _normalize_team(self, series):
        return series.replace(self.TEAM_ALIASES)

    def _norm_name(self, n):
        """Normalize player names for joins across sources."""
        if pd.isna(n):
            return ''
        return (str(n).replace('.', '').replace("'", '').replace('-', ' ')
                .replace(' Jr', '').replace(' Sr', '').replace(' III', '')
                .lower().strip())

    # ------------------------------------------------------------------
    # Build a player_id -> current_team lookup
    # ------------------------------------------------------------------
    def _build_current_team_lookup(self):
        """Return a dict {player_id: current_team} or {normalized_name: current_team}.

        Tries nflreadpy rosters first (most authoritative for target_season).
        Falls back to the DK salary CSV's TeamAbbrev column, which is always
        correct for the specific slate being projected.
        """
        lookup_by_id = {}
        lookup_by_name = {}

        # --- Primary source: nflreadpy rosters for the target season ---
        try:
            roster_raw = nflreadpy.load_rosters([self.target_season])
            roster = roster_raw.to_pandas() if hasattr(roster_raw, "to_pandas") else pd.DataFrame(roster_raw)

            if not roster.empty and 'team' in roster.columns:
                # Prefer most recent week per player
                if 'week' in roster.columns:
                    roster = roster.sort_values('week', ascending=False)
                roster = roster.drop_duplicates('player_id') if 'player_id' in roster.columns else roster

                if 'player_id' in roster.columns:
                    lookup_by_id = (
                        roster[roster['team'].notna()]
                        .set_index('player_id')['team']
                        .to_dict()
                    )
                    print(f"[Core] Roster lookup (by id) built for {len(lookup_by_id)} players.")
        except Exception as e:
            print(f"[Warning] nflreadpy.load_rosters([{self.target_season}]) failed: {e}")

        # --- Fallback / supplement: DK salary CSV TeamAbbrev ---
        if self.dk_salary_csv and pd.io.common.file_exists(self.dk_salary_csv):
            try:
                dk = pd.read_csv(self.dk_salary_csv)
                if 'TeamAbbrev' in dk.columns and 'Name' in dk.columns:
                    dk_teams = dk[['Name', 'TeamAbbrev']].dropna().drop_duplicates('Name')
                    lookup_by_name = {
                        self._norm_name(name): team
                        for name, team in zip(dk_teams['Name'], dk_teams['TeamAbbrev'])
                    }
                    print(f"[Core] DK salary team lookup built for {len(lookup_by_name)} players.")
            except Exception as e:
                print(f"[Warning] Could not read DK salary team abbreviations: {e}")

        return lookup_by_id, lookup_by_name

    # ------------------------------------------------------------------
    # Data fetch & clean
    # ------------------------------------------------------------------
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
        # --- Full display name for joins against DK ---
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
        # --- Numeric fallbacks ---
        for col in ['pass_attempts', 'rush_attempts', 'receptions', 'targets',
                    'passing_yards', 'passing_tds', 'passing_interceptions',
                    'rushing_yards', 'rushing_tds',
                    'receiving_yards', 'receiving_tds',
                    'sack_fumbles_lost', 'rushing_fumbles_lost', 'receiving_fumbles_lost']:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        # ============================================================
        # ROSTER REMAPPING
        # ============================================================
        print(f"[Core] Remapping player teams using {self.target_season} rosters and DK salary data...")
        lookup_by_id, lookup_by_name = self._build_current_team_lookup()
        team_col = 'team' if 'team' in df.columns else ('recent_team' if 'recent_team' in df.columns else None)
        if team_col is None:
            raise RuntimeError("No team column found in player stats.")
        # IMPORTANT: preserve the original historical team
        df['historical_team'] = df[team_col].copy()
        # Build current-team lookup
        current_team = pd.Series([None] * len(df), index=df.index, dtype='object')
        if 'player_id' in df.columns and lookup_by_id:
            current_team = df['player_id'].map(lookup_by_id)
        unmatched_mask = current_team.isna()
        if unmatched_mask.any() and lookup_by_name:
            norm_names = df.loc[unmatched_mask, 'player_name'].map(self._norm_name)
            dk_teams = norm_names.map(lookup_by_name)
            current_team.loc[unmatched_mask] = dk_teams
        remapped = (current_team.notna() & (current_team != df['historical_team'])).sum()
        filled = current_team.notna().sum()
        print(f"[Core] Teams resolved for {filled}/{len(df)} rows; {remapped} changed vs historical team.")
        # Apply remap. Where we have a current team, use it. Otherwise historical.
        df['team'] = current_team.fillna(df['historical_team'])
        # Normalize team codes on both columns
        df['team'] = self._normalize_team(df['team'])
        df['historical_team'] = self._normalize_team(df['historical_team'])
        if 'opponent_team' in df.columns:
            df['opponent_team'] = self._normalize_team(df['opponent_team'])
        # ============================================================
        # TEAM DENOMINATORS — group by HISTORICAL team, not current team
        # ============================================================
        # A player's 2025 share should be measured against the 2025 offense
        # they actually played in, not the 2026 team they'll play for.
        df['team_pass_attempts'] = df.groupby(['historical_team', 'week'])['pass_attempts'].transform('sum').replace(0, np.nan)
        df['team_rush_attempts'] = df.groupby(['historical_team', 'week'])['rush_attempts'].transform('sum').replace(0, np.nan)
        df['team_receptions']    = df.groupby(['historical_team', 'week'])['receptions'].transform('sum').replace(0, np.nan)
        df['team_targets']       = df.groupby(['historical_team', 'week'])['targets'].transform('sum').replace(0, np.nan)
        # --- DK scoring ---
        pass_pts = df['passing_yards'] * 0.04 + df['passing_tds'] * 4.0 - df['passing_interceptions'] * 1.0
        rush_pts = df['rushing_yards'] * 0.1 + df['rushing_tds'] * 6.0
        rec_pts  = df['receptions'] * 1.0 + df['receiving_yards'] * 0.1 + df['receiving_tds'] * 6.0
        fumbles_lost = (df['sack_fumbles_lost'] + df['rushing_fumbles_lost'] + df['receiving_fumbles_lost'])
        df['dk_points'] = pass_pts + rush_pts + rec_pts + (fumbles_lost * -2.0)
        self.master_weekly = df
        self._loaded = True

    # ------------------------------------------------------------------
    # DvP — unchanged from your working version
    # ------------------------------------------------------------------
    def calculate_matchup_dvp(self):
        print("[Core] Calculating EPA-based Defense vs Position (DvP) indices...")

        try:
            pbp_raw = nflreadpy.load_pbp([self.stats_season])
            pbp = pbp_raw.to_pandas() if hasattr(pbp_raw, "to_pandas") else pd.DataFrame(pbp_raw)
        except Exception as e:
            print(f"[Warning] Could not load PBP: {e}. Falling back to points-based DvP.")
            return self._fallback_dvp()

        if pbp.empty:
            return self._fallback_dvp()

        def find_col(candidates):
            for c in candidates:
                if c in pbp.columns:
                    return c
            return None

        offense_col  = find_col(['posteam', 'pos_team', 'offense_team', 'offense'])
        defense_col  = find_col(['defteam', 'def_team', 'defense_team', 'defense'])
        pass_col     = find_col(['pass_attempt', 'pass_attempts', 'is_pass'])
        rush_col     = find_col(['rush_attempt', 'rush_attempts', 'is_rush'])
        sack_col     = find_col(['sack', 'is_sack'])
        kneel_col    = find_col(['qb_kneel', 'kneel'])
        epa_col      = find_col(['epa', 'EPA'])
        season_ty    = find_col(['season_type', 'game_type'])

        if not offense_col or not defense_col or not epa_col:
            print("[Warning] PBP missing required columns. Falling back.")
            return self._fallback_dvp()

        if season_ty:
            pbp = pbp[pbp[season_ty] == 'REG'].copy()

        if pass_col:
            pass_mask = pbp[pass_col] == 1
            if sack_col:
                pass_mask = pass_mask & (pbp[sack_col] == 0)
            pass_plays = pbp[pass_mask & pbp[epa_col].notna()].copy()
        else:
            pass_plays = pd.DataFrame()

        if rush_col:
            rush_mask = pbp[rush_col] == 1
            if kneel_col:
                rush_mask = rush_mask & (pbp[kneel_col] == 0)
            rush_plays = pbp[rush_mask & pbp[epa_col].notna()].copy()
        else:
            rush_plays = pd.DataFrame()

        off_pass = pd.DataFrame()
        off_rush = pd.DataFrame()
        if not pass_plays.empty:
            off_pass = pass_plays.groupby(offense_col)[epa_col].mean().reset_index()
            off_pass.columns = ['offense_team', 'off_pass_epa_per_play']
        if not rush_plays.empty:
            off_rush = rush_plays.groupby(offense_col)[epa_col].mean().reset_index()
            off_rush.columns = ['offense_team', 'off_rush_epa_per_play']

        pass_def_adj = pd.DataFrame()
        rush_def_adj = pd.DataFrame()

        if not pass_plays.empty and not off_pass.empty:
            pass_plays_adj = pass_plays.rename(columns={offense_col: 'offense_team'}).merge(
                off_pass, on='offense_team', how='left'
            )
            pass_plays_adj['epa_over_expected'] = pass_plays_adj[epa_col] - pass_plays_adj['off_pass_epa_per_play']
            pass_def_adj = pass_plays_adj.groupby(defense_col)['epa_over_expected'].mean().reset_index()
            pass_def_adj.columns = ['defense_team', 'pass_epa_adj']

        if not rush_plays.empty and not off_rush.empty:
            rush_plays_adj = rush_plays.rename(columns={offense_col: 'offense_team'}).merge(
                off_rush, on='offense_team', how='left'
            )
            rush_plays_adj['epa_over_expected'] = rush_plays_adj[epa_col] - rush_plays_adj['off_rush_epa_per_play']
            rush_def_adj = rush_plays_adj.groupby(defense_col)['epa_over_expected'].mean().reset_index()
            rush_def_adj.columns = ['defense_team', 'rush_epa_adj']

        if pass_def_adj.empty and rush_def_adj.empty:
            return self._fallback_dvp()

        if not pass_def_adj.empty:
            league_pass = pass_def_adj['pass_epa_adj'].mean()
            pass_def_adj['pass_dvp_mult'] = (1.0 + (pass_def_adj['pass_epa_adj'] - league_pass) * 1.5).clip(0.75, 1.25)

        if not rush_def_adj.empty:
            league_rush = rush_def_adj['rush_epa_adj'].mean()
            rush_def_adj['rush_dvp_mult'] = (1.0 + (rush_def_adj['rush_epa_adj'] - league_rush) * 1.5).clip(0.75, 1.25)

        dvp_rows = []
        for _, r in pass_def_adj.iterrows():
            for pos in ['QB', 'WR', 'TE']:
                dvp_rows.append({'defense_team': r['defense_team'], 'position': pos,
                                 'dvp_multiplier': r['pass_dvp_mult']})

        for _, r in rush_def_adj.iterrows():
            pass_row = pass_def_adj[pass_def_adj['defense_team'] == r['defense_team']]
            pass_mult = pass_row['pass_dvp_mult'].iloc[0] if not pass_row.empty else 1.0
            rb_mult = 0.6 * r['rush_dvp_mult'] + 0.4 * pass_mult
            dvp_rows.append({'defense_team': r['defense_team'], 'position': 'RB',
                             'dvp_multiplier': round(rb_mult, 3)})

        dvp = pd.DataFrame(dvp_rows)
        if dvp.empty:
            return self._fallback_dvp()

        print(f"[Core] EPA DvP computed for {len(dvp)} defense-position pairs.")
        return dvp

    def _fallback_dvp(self):
        hist_data = self.master_weekly.copy()
        position_avg = hist_data.groupby('position')['dk_points'].mean().to_dict()
        def_allowed = hist_data.groupby(['opponent_team', 'position'])['dk_points'].mean().reset_index()
        def_allowed.columns = ['defense_team', 'position', 'avg_points_allowed']
        def_allowed['dvp_multiplier'] = def_allowed.apply(
            lambda r: r['avg_points_allowed'] / position_avg[r['position']] if position_avg[r['position']] > 0 else 1.0,
            axis=1
        )
        return def_allowed

    # ------------------------------------------------------------------
    # Team volumes
    # ------------------------------------------------------------------
    def project_macro_team_volume(self):
        print("[Core] Extracting Vegas totals and playbook profiles...")

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
            if self.dk_salary_csv and pd.io.common.file_exists(self.dk_salary_csv):
                try:
                    dk = pd.read_csv(self.dk_salary_csv)
                    if 'Game Info' in dk.columns:
                        matchups = dk['Game Info'].drop_duplicates()
                        for info in matchups:
                            game = str(info).split(' ')[0]
                            if '@' in game:
                                away, home = game.split('@')
                                vegas_rows.append({'team': home, 'opponent_team': away,
                                                   'implied_total': 22.0, 'is_fav': False})
                                vegas_rows.append({'team': away, 'opponent_team': home,
                                                   'implied_total': 22.0, 'is_fav': False})
                        if vegas_rows:
                            vegas_rows = pd.DataFrame(vegas_rows).drop_duplicates('team').to_dict('records')
                except Exception as e:
                    print(f"[Warning] Could not parse DK Game Info: {e}")

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
                'completion_rate': 0.62,
            })

        vegas = pd.DataFrame(vegas_rows)
        vegas['team'] = self._normalize_team(vegas['team'])
        vegas['opponent_team'] = self._normalize_team(vegas['opponent_team'])
        vegas['v_mod'] = vegas['implied_total'] / 22.0

        hist = self.master_weekly.copy()
        # Use historical_team here: we want to know how many plays each 2025 offense
        # ran, not how many 2026 teams' combined rosters ran in 2025.
        team_games = (hist.groupby(['historical_team', 'week'])
                      .agg(pass_att=('pass_attempts', 'sum'),
                           receptions=('receptions', 'sum'),
                           rush_att=('rush_attempts', 'sum'))
                      .reset_index())
        #hist = self.master_weekly.copy()
        #team_games = (hist.groupby(['team', 'week'])
        #              .agg(pass_att=('pass_attempts', 'sum'),
        #                   receptions=('receptions', 'sum'),
        #                   rush_att=('rush_attempts', 'sum'))
        #              .reset_index())
        team_games['plays'] = team_games['pass_att'] + team_games['rush_att']
        team_games['pass_rate'] = team_games['pass_att'] / team_games['plays'].replace(0, pd.NA)
        team_games['completion_rate'] = team_games['receptions'] / team_games['pass_att'].replace(0, pd.NA)

        #baselines = (team_games.groupby('team')
        #             .agg(avg_plays=('plays', 'mean'),
        #                  pass_rate_identity=('pass_rate', 'mean'),
        #                  completion_rate=('completion_rate', 'mean'))
        #             .reset_index())
        baselines = (team_games.groupby('historical_team')
             .agg(avg_plays=('plays', 'mean'),
                  pass_rate_identity=('pass_rate', 'mean'),
                  completion_rate=('completion_rate', 'mean'))
             .reset_index())
        baselines = baselines.rename(columns={'historical_team': 'team'})

        volume = pd.merge(baselines, vegas, on='team', how='right')
        volume['avg_plays'] = volume['avg_plays'].fillna(64.0)
        volume['pass_rate_identity'] = volume['pass_rate_identity'].fillna(0.60)
        volume['completion_rate'] = volume['completion_rate'].fillna(0.62)

        volume['proj_plays'] = volume['avg_plays'] * (1 + (volume['v_mod'] - 1) * 0.3)
        volume['proj_team_pass'] = volume['proj_plays'] * volume['pass_rate_identity']
        volume['proj_team_rec']  = (volume['proj_team_pass'] * volume['completion_rate']).clip(upper=26.0)
        volume['proj_team_rush'] = volume['proj_plays'] * (1 - volume['pass_rate_identity'])

        volume = volume.rename(columns={'team': 'recent_team'})
        return volume[['recent_team', 'opponent_team', 'implied_total',
                       'proj_plays', 'proj_team_pass', 'proj_team_rec',
                       'proj_team_rush', 'is_fav', 'completion_rate']]