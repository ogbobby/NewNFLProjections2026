Complete pipeline walkthrough

Here's the full system, in the order things run, with what each file does.
The files

Core projection files (the model):
File	Purpose
engine_core.py	Loads NFL stats, remaps teams, computes DvP, Vegas totals
engine_pipeline.py	Player shares, efficiency, shrinkage, ceiling projections
redzone_engine.py	Red-zone opportunity data from play-by-play
dfs_final_model.py	Orchestrator — runs the pipeline, applies calibration, market prior, manual overrides, ownership
generate_dsts.py	Adds DST projections (pipeline doesn't produce them)

Analysis and validation files:
File	Purpose
dk_contest_analyzer.py	Parses DK contest results, computes ROI, writes last_week_results.csv
validate_wk1.py	Compares pipeline projections to actual results for a past week
injury_scraper.py	Pulls injury data from nflreadpy (not yet wired in)
apply_injuries.py	Converts injury flags to multiplier overrides (not yet wired in)

Lineup construction:
File	Purpose
weekly_pipeline.py	Master script — runs projections, filters to slate, builds lineups, writes output

Data files (inputs and outputs):
File	Direction	Purpose
DKSalaries.csv	Input	Current week's DK salary file
manual_adjustments.csv	Input	Your fade/core/multiplier overrides
last_week_results.csv	Input	Previous week's actual DK points
sim_input_projections_2026_wN.csv	Output	The projection file the pipeline produces
output/week_N/lineups_wN.csv	Output	Final lineups
The complete weekly workflow
Step 1 — Get the current week's DK salary file

Go to DraftKings, find the contest you're entering, click Export → Export Salaries. Save as DKSalaries.csv in the project directory.

This file has salary, ownership, Game Info (which teams are playing), and AvgPointsPerGame for each player. Every downstream step reads it.
Step 2 — Update manual_adjustments.csv

Open the file. Add any players you want to fade, force-include, or adjust. Format:
csv

player_name,projection_mult,ceiling_mult,force_include,fade,notes
Kyler Murray,0.55,0.55,false,true,hurt
Kenny Gainwell,0.65,0.65,false,false,RB2

    projection_mult — multiply the model's projection by this (1.0 = no change)

    fade: true — remove the player from the pool entirely

    force_include: true — try to include the player in each lineup

This is where your subjective judgment goes: injuries the model doesn't know about, players you want to avoid, etc.
Step 3 — Run the projection pipeline
bash

python3 weekly_pipeline.py --week 3 --salary-file DKSalaries.csv

That single command runs everything:

    Loads the DK salary file (identifies slate teams, salary data)

    Runs dfs_final_model.py, which orchestrates:

        engine_core.py loads 2024+2025 stats, remaps to 2026 teams, computes DvP and Vegas totals

        engine_pipeline.py computes player shares, efficiency, shrinkage, projections

        redzone_engine.py adds red-zone data

        dfs_final_model.py applies calibration, QB market prior, soft caps, manual overrides, ownership model

        Writes sim_input_projections_2026_wN.csv

    Runs generate_dsts.py to add DST projections

    Loads the projections, filters to slate teams

    Applies manual_adjustments.csv (fades, multipliers)

    Filters the pool (position floors with salary exemption, ownership threshold)

    Identifies stack candidates from Vegas totals

    Builds 20 lineups with stacks and bring-backs

    Writes output/week_N/lineups_wN.csv

The console shows all the intermediate steps. You're watching for:

    [Engine] QB projections rebuilt — QB ordering matches market

    [Check] N stack-eligible teams — teams that survive the pool filter

    [AUTOMATED STACK CANDIDATES] — the top stack targets

    Built 20 lineups — success

    Bring-backs in final lineups: N — should be ~10-12 for 20 lineups

Step 4 — Review the lineups

Open output/week_N/lineups_wN.csv. Each row is a lineup with 9 player names, salary, and projected points. The columns are QB, RB1, RB2, WR1, WR2, WR3, TE, FLEX, DST.

Two things to check:

    No faded players. If Kyler Murray is faded and shows up, the fade isn't working.

    No obvious mistakes. If a player who's been ruled out is in a lineup, add them to manual_adjustments.csv and rerun.

Step 5 — Submit lineups to DK

Copy the lineups into DK's entry form. Or, if you have a script for bulk entry, feed it lineups_wN.csv.
Step 6 — After the games (Monday)

Download the contest results from DK:

    Contests → History → Results → Export Lineups to CSV

    Save as contest-standings-WEEK_N.csv

Run the analyzer:
bash

python3 dk_contest_analyzer.py \
  contest-standings-WEEK_N.csv \
  --username runninjewels \
  --save-ewma

That does two things:

    Prints a summary: cash line, top lineups, your lineups' percentiles, biggest wins and losses

    Writes last_week_results.csv with each player's actual DK points

That last_week_results.csv file is the input to two downstream steps:

    The EWMA blend in the next pipeline run (already wired into dfs_final_model.py)

    The weekly validation

Step 7 — (Optional) Validate against the week

If you want to see how your projections held up:
bash

python3 validate_wk1.py \
  sim_input_projections_2026_wN.csv \
  last_week_results.csv

Replace wk1 with the week you're validating. The output shows MAE, bias, and correlation by position. This is how you know if the model is calibrated.

Note: validate_wk1.py is my name for the script. If your file is named differently, adjust. And if you want to validate the current week's projections against actuals from the same week, you need to regenerate the projections before the EWMA blend kicks in (otherwise you're validating a projection that already knows the answer). The way around this is to temporarily rename last_week_results.csv before generating, then validate, then restore.
The full sequence for a new week
text

1. Download DKSalaries.csv from the contest
2. Update manual_adjustments.csv with any new overrides
3. Run:  python3 weekly_pipeline.py --week N --salary-file DKSalaries.csv
4. Review output/week_N/lineups_wN.csv
5. Submit lineups to DK
6. (After games) Download contest CSV
7. Run:  python3 dk_contest_analyzer.py contest-standings-WEEK_N.csv --username X --save-ewma
8. (Optional) Run:  python3 validate_wk1.py sim_input_projections_2026_wN.csv last_week_results.csv

What each command does under the hood

weekly_pipeline.py is the master. It:

    Calls dfs_final_model.py's run_gpp_optimized_pipeline method

    That method calls engine_core, engine_pipeline, redzone_engine in sequence

    Then applies its post-processing (calibration, market prior, ownership)

    Writes the projection CSV

    Back in weekly_pipeline.py: adds DSTs, filters pool, identifies stacks, builds lineups

You can run the projection part alone (without lineup building) if you just want the CSV:
bash

python3 dfs_final_model.py

But that skips the DST generation and the slate filtering. weekly_pipeline.py is the complete workflow.
What's currently automated vs. manual

Automated:

    Data loading, team remapping, DvP

    Projection calculation with shrinkage, ceiling, leverage

    Position calibration and QB market prior

    Ownership model

    DST generation

    Slate filtering

    Stack identification

    Lineup construction with bring-backs

Manual (per week):

    Downloading DK salary file

    Reading injury news, updating manual_adjustments.csv

    Running the commands

    Reviewing lineups before submission

    Downloading contest results after games

Manual (once):

    Setting POSITION_CALIBRATION constants (done)

    Setting POSITION_FLOOR_PCT (done)

    Deciding how many lineups to build (--n-lineups)

The two files you'll edit most

manual_adjustments.csv — your weekly subjective input. 5-10 minutes of review per week.

DKSalaries.csv — replaced every week from DK's site.

Everything else is code that runs.
What to run right now

If you want to test the pipeline end to end with the current week:
bash

python3 weekly_pipeline.py --week 3 --salary-file DKSalaries.csv

If DKSalaries.csv is for week 3, this runs the full workflow. If it's still for week 2, use --week 2. The pipeline reads the salary file's Game Info to know which teams are playing, so the week number and the salary file need to match.
