import cv2
import json
from playsound import playsound
import threading
import pywhatkit as msg
import sys,os
sys.path.append(os.path.abspath(".\Knife_Detect"))
from Knife_Detect import yolo
net,labels,colors , layer_names = yolo.load_yolo()
FLAGS = yolo.parser()


def detector():
    #Capture video on webcam
    camera = cv2.VideoCapture(0)
    count = 0

    #Detect face and recognize 
    cascade = cv2.CascadeClassifier('.\Face_recognize\haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('.\\Face_recognize\\trainner.yml')

    #open users json file and retrieve names
    with open('.\\Face_recognize\\users.json') as jsonFile:
        users = json.load(jsonFile) 
    userList = []
    imgcount=0
    #loop over the frames 
    while 1:
        #read frames 
        ret, image = camera.read()

        #convert to gray and detect 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face = cascade.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5, minSize = (30,30))
        #height and width of the frame
        height, width = image.shape[:2]

        #infer real time using yolo
        if count == 0:
            image, boxes, confidences, classids, idxs = yolo.infer_image(net, layer_names, \
                                                                    height, width, image, colors, labels, FLAGS)

            count += 1
        else:
            image, boxes, confidences, classids, idxs = yolo.infer_image(net, layer_names, \
                                                                    height, width, image, colors, labels, FLAGS,
                                                                    boxes, confidences, classids, idxs, infer=False)
            count = (count + 1) % 6
        for (x,y,w,h) in face:
            cv2.rectangle(image, (x,y), (x+w, y+h), (100,0,100), 2)

            faceId, percentage = recognizer.predict(gray[y:y+h, x:x+w])

            if percentage < 50:
                userList.append(faceId)
                faceId = users[str(faceId)]['name']+' '+str(round(100-percentage,2))+'%'
                
            else:
                faceId = 'Unknown'
                

            cv2.putText(image, faceId, (x,y+h),cv2.FONT_HERSHEY_SIMPLEX, 1, (50,255,),2)

        # we need 5 screen and send 1 msg
        if classids : 
            cv2.imwrite("screenShot{}.jpg".format(imgcount), image)
            imgcount+=1
            if imgcount==1:
                thread = threading.Thread(target=playsound, args=("alarm3.wav",))
                thread.start()
                msg.sendwhatmsg_instantly('+201006573885', "We detect a knife with {}".format(faceId), 15, 2)

        cv2.imshow('Image',image)
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

        # msg.sendwhats_image('+201006573885', ".\Face_recognize\screenShot0.jpg" , "We detect a knife with ")
    camera.release()
    cv2.destroyAllWindows()



