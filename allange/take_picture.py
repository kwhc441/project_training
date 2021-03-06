''''
Capture multiple Faces from multiple users to be stored on a DataBase (dataset directory)
	==> Faces will be stored on a directory: dataset/ (if does not exist, pls create one)
	==> Each face will have a unique numeric integer ID as 1, 2, 3, etc                       
Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    
Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18    
'''

import cv2
import os
import time
import datetime

def chk_mkdir(cusnum):
    face_id = str(cusnum)
    if not os.path.exists(face_id):
        os.mkdir(face_id)
        os.chdir(face_id)
    else:
        cusnum +=1
        chk_mkdir(cusnum)


#cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height


face_detector = cv2.CascadeClassifier('allange\\haarcascade_frontalface_default.xml')  # xmlファイルの場所をフルパスで指定



# For each person, enter one numeric face id
cusnum=1

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0
"""
opening_date = datetime.date.today().strftime('%y%m%d')
os.chdir("allange\\datasets")
if not os.path.exists(opening_date):
    os.mkdir(opening_date)
os.chdir(f"{opening_date}")
chk_mkdir(cusnum)
"""
while(True):
    
    ret, img = cam.read()
    time.sleep(1)
    #img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)#エラーの原因？
    opening_date = datetime.datetime.now().strftime('%y%m%d%H%M')
    

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        #cv2.imwrite(str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imwrite("allange\\datasets\\"+opening_date+str(00)+str(count) + ".jpg", gray[y:y+h, x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
"""
label = input("customer name>: ")
now_dir=os.getcwd()
os.chdir("..")
os.rename(now_dir,label)
"""

