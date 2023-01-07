import cv2
from trainner import trainner
import json
from random import randint
def create():
    camera = cv2.VideoCapture(0)

    cascade = cv2.CascadeClassifier('.\Face_recognize\haarcascade_frontalface_default.xml')

    name = input('Enter Name : ')
    while name.isdigit():
        print('enter only alpha')
        name = input('Enter Name : ')

    Id=randint(0,100)
    print(Id)
    itr = 1

    while 1:

        _, image = camera.read()

        grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        face = cascade.detectMultiScale(grayImage,
                                        scaleFactor = 1.5,
                                        minNeighbors = 5,
                                        minSize= (30,30))

        for (x,y,w,h) in face:
            
            cv2.rectangle(image, (x,y), (x+w,y+h), (100,100,0), 2)

            cv2.imwrite('./Face_recognize/dataset/User.' + str(Id) + '.' + str(itr) + '.jpg', grayImage[y:y+h+7, x:x+w+7])
        itr += 1

        if itr == 51:
            break
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        cv2.imshow('Frame',image)
    camera.release()
    cv2.destroyAllWindows()

    with open(".\\Face_recognize\\users.json") as json_file:
        data = json.load(json_file)

    username = {
                'name':name
            }
    data[Id] = username

    with open(".\\Face_recognize\\users.json", "w") as file:  
        json.dump(data, file,indent = 4)
        
    trainner()
