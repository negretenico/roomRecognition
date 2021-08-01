import os 
DIR  = os.getcwd() +"\\Images\\Basements"
for item in os.listdir(DIR):
    if item.startswith("u"):
        try:
            os.remove(os.path.join(DIR,item))
            print(f'{item}was removed')
        except:
            print("Error occured")