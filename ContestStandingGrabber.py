import os
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc


"""
This needs to be the automated process of logging in to dk and downloading the contest standings csv.
It will also need to have cleanup after files have been downloaded and moved.
"""