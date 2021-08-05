from imageScapper import GoogleImageScraper
import os
dir = os.getcwd()
i = 0
webdriver_path = os.path.join(os.path.join(os.getcwd(),"webdriver"),"chromedriver.exe")
image_scapper = GoogleImageScraper(webdriver_path=webdriver_path)
LIMIT = 1000
with open(dir + "\\rooms.txt", encoding="utf8") as file:
    for room in file:
        room = room.replace("\n","")
        image_path = os.path.join(os.path.join(os.path.join(os.getcwd(),"data"),"train"),room)
        image_scapper.find_and_save(search_key=room,limit = LIMIT,image_path=image_path)