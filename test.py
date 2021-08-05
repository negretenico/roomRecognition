import os 
categories = ["BathRoom","BedRoom","Kitchen","LivingRoom"]
train_val = ["train","validation"]
source  = os.getcwd() +"\\data\\train"
target = os.getcwd()+"\\data\\validation"
"""
op portion creates train and validation split 
ottom portion reorders the directories
"""
# for cat in categories:
#     for item in os.listdir(os.path.join(source,cat))[:25]:
#         try:
#             os.rename(os.path.join(os.path.join(source,cat),item), os.path.join(os.path.join(target,cat),item))
#         except Exception as e:
#             print("Error Occured")
#             print(e)

# path = os.path.join(os.getcwd(),"data")
# for state in train_val:
#     for cat in categories:
#         print(f"{state, cat} has {len(os.listdir(os.path.join(os.path.join(path,state),cat)))} items")
#         for i,file in enumerate(os.listdir(os.path.join(os.path.join(path,state),cat))):
#             os.remove(os.path.join(os.path.join(os.path.join(path,state),cat),file))



path = os.path.join(os.path.join(os.getcwd(),"data"),"train")

driver_path = os.path.join(os.path.join(os.getcwd(),"webdriver"),"chromedriver.exe")