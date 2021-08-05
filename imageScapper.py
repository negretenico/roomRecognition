
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# import helper libraries
import time
import urllib.request
import shutil
import os
import requests
import struct
import imghdr
from selenium.webdriver.common.keys import Keys

class GoogleImageScraper():
    def __init__(self, webdriver_path ):
        # check parameter types
        self.webdriver_path = webdriver_path


    def find_and_save(self,search_key="Naruto", limit = 1,image_path=os.getcwd()):
        if (type(limit) != int):
            print("GoogleImageScraper Error: Number of images must be integer value.")
            return
        if not os.path.exists(image_path):
            print("GoogleImageScraper Notification: Image path not found. Creating a new folder.")
            os.makedirs(image_path)
       
       
        driver = webdriver.Chrome(self.webdriver_path)
        driver.get( "https://www.google.com/search?q=%s&source=lnms&tbm=isch&sa=X&ved=2ahUKEwie44_AnqLpAhUhBWMBHUFGD90Q_AUoAXoECBUQAw&biw=1920&bih=947" % (
            search_key))

        #Will keep scrolling down the webpage until it cannot scroll no more
        last_height = driver.execute_script('return document.body.scrollHeight')
        while True:
            driver.execute_script('window.scrollTo(0,document.body.scrollHeight)')
            time.sleep(1)
            new_height = driver.execute_script('return document.body.scrollHeight')
            try:
                driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div[1]/div[2]/div[2]/input').click()
                time.sleep(2)
            except:
                pass
            if new_height == last_height:
                break
            last_height = new_height
       
       #downloads images
        print(f"Began saving images for {search_key}")
        for i in range(1, limit+1):
            try:
                image_path = image_path+"\\"+str(i)+'.JPEG'
                original_size = driver.get_window_size()
                height = driver.execute_script("return document.body.parentNode.scrollHeight")
                driver.set_window_size(original_size['width'], height)
                driver.find_element_by_xpath('//*[@id="islrg"]/div[1]/div['+str(i-1)+']/a[1]/div[1]/img').screenshot(image_path)
                time.sleep(1)
            except:
                pass
        
        print("Completed.")
        driver.close()