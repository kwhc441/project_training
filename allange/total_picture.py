#撮影部
from PIL import Image
import numpy as np
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
        cusnum += 1
        chk_mkdir(cusnum)


#cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height


face_detector = cv2.CascadeClassifier('allange\\haarcascade_frontalface_default.xml')  # xmlファイルの場所をパスで指定


# For each person, enter one numeric face id
cusnum = 1

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
    faces = face_detector.detectMultiScale(gray, 1.3, 5)  # エラーの原因？
    opening_date = datetime.datetime.now().strftime('%y%m%d%H%M')

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1

        # Save the captured image into the datasets folder
        #cv2.imwrite(str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imwrite("allange\\datasets\\"+opening_date+str(00) +
                    str(count) + ".jpg", gray[y:y+h, x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30:  # Take 30 face sample and stop video
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
#訓練部
opening_date = datetime.date.today().strftime('%y%m%d')
# Path for face image database
# 日付ごと、対象ファイルごとに変えられるように工夫が必要
"""
cnt=1
cusname=input(f"cnt={cnt}>:")
path = f'allange\\datasets\\{opening_date}\\{cusname}'
cnt+=1
"""
path = 'allange\\datasets'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(
    "allange\\haarcascade_frontalface_default.xml")
imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
"""
for imagePath in imagePaths:
    id = os.path.split(imagePath)[-1].split(".")[0]
    print(id)
"""
# function to get the images and label data


def getImagesAndLabels(path):

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
        img_numpy = np.array(PIL_img, 'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[0])  # 数字は最初の要素なので[0]
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return faceSamples, ids


print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
# recognizer.save() worked on Mac, but not on Pi
recognizer.write('allange\\trainer\\trainer.yml')

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
#判断部

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('allange\\trainer\\trainer.yml')
cascadePath = "allange\\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None',"ttt","sss","aaa","kkk"]
cnt=1

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
#cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
"""
name = input("customer_name>:")
purpose = input("customer_purpose>:")
label = name+"\n"+purpose
names.append(label)
"""
name = input("customer_name>:")
purpose = input("customer_purpose>:")
label = name+"\n"+purpose
if cnt < len(names):
    names[cnt] = label
else:
    names.append(label)
while True:

    ret, img = cam.read()
    #img = cv2.flip(img, -1) # Flip vertically

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    #print(faces)

    for(x,y,w,h) in faces:

        
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        #print(id,confidence)
        #print(names)

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            #id = names[id]
            #confidence = "  {0}%".format(round(100 - confidence))
            try:
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            except:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

