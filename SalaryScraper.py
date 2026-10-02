import os
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc

# 1. CHANGE THESE PATHS FOR YOUR SYSTEM
CHROME_PROFILE_PATH = r"/home/ogbobby/.config/google-chrome/Default" # Update YOUR_NAME
PROFILE_DIRECTORY = "Default"  # Or "Profile 1", "Profile 2" depending on which one you use
CONTEST_DRAFT_URL = "https://www.draftkings.com/draft/contest/195905122" # Replace with your target draft page

def download_dk_salaries():
    options = uc.ChromeOptions()
    
    # Point Selenium to your local Chrome profile
    options.add_argument(f"--user-data-dir={CHROME_PROFILE_PATH}")
    options.add_argument(f"--profile-directory={PROFILE_DIRECTORY}")
    
    # 1. DEFINE YOUR EXACT TARGET PATH HERE
    # Example: '/home/ogbobby/Documents/git/NewNFLProjections2026/data'
    download_dir = "/home/ogbobby/Documents/git/NewNFLProjections2026/" 
    
    # Automatically create the folder if it doesn't exist yet
    os.makedirs(download_dir, exist_ok=True)
    
    # 2. FORCE CHROME TO USE THIS PATH
    options.add_experimental_option("prefs", {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": False,
        # This hidden preference forces Chrome to ignore your previous manual download history:
        "download.default_directory_behavior": "allow" 
    })
    
    options.add_argument("--disable-features=InsecureDownloadWarnings")

    print("Launching Chrome via undetected-chromedriver...")
    driver = uc.Chrome(options=options)
    
    try:
        # Navigate to the specific contest draft page where the export button lives
        print(f"Navigating to: {CONTEST_DRAFT_URL}")
        driver.get(CONTEST_DRAFT_URL)
        
        # Give the page ample time to load dynamically loaded React elements
        print("Waiting for page load and export button to appear...")
        wait = WebDriverWait(driver, 20)
        
        # DraftKings text on the button is typically "Export to CSV"
        # Using a partial text match is highly reliable
        export_button = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Export to CSV')]"))
        )
        
        print("Clicking export button...")
        export_button.click()
        
        # Wait a few seconds to let the file complete downloading before closing the browser
        print("Download triggered! Waiting for completion...")
        time.sleep(5)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Tip: Make sure all manual Chrome instances are fully closed before running this script.")
        
    finally:
        driver.quit()
        print("Browser closed.")

if __name__ == "__main__":
    download_dk_salaries()
