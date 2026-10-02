import os
import glob
import shutil
"""
This first function will move and rename the DKsalary file to salaryfiles directory 
------Stuff that still needs to be added---
1. Old field lineups need to be moved or deleted, not sure if we need to keep them. There is a FieldLineups dir (---field_w3.csv)
2. The grading results csv(grading_results.csv) needs to be deleted.
3. Not sure where these my_lineups(my_lineups_w2.csv) files came from, may have been from early testing days.
4. Old projection files(sim_input_projections_2026_w3.csv) need moved to directory OldProjectionFiles.

"""

def rename_and_move_salary_file(download_dir, target_dir, new_filename="dk_salaries_current.csv"):
    """
    Finds the newest CSV file in download_dir, renames it, 
    and moves it to target_dir.
    """
    # 1. Look for any CSV file starting with 'DKSalaries' in the source directory
    search_path = os.path.join(download_dir, "DKSalaries*.csv")
    csv_files = glob.glob(search_path)
    
    if not csv_files:
        print(f"❌ No DraftKings CSV files found in: {download_dir}")
        return None

    # 2. Sort files by modification time to get the absolute newest one
    newest_file = max(csv_files, key=os.path.getmtime)
    print(f"🔍 Found newest file: {os.path.basename(newest_file)}")

    # 3. Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # 4. Construct the full destination path
    destination_path = os.path.join(target_dir, new_filename)

    try:
        # 5. Move and rename the file (overwrites if it already exists)
        shutil.move(newest_file, destination_path)
        print(f"✅ Successfully moved and renamed to: {destination_path}")
        return destination_path
    except Exception as e:
        print(f"❌ Error moving file: {e}")
        return None