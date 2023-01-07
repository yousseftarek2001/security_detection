import sys
import os
import argparse
from multiprocessing import Process
#This function to set arguments
def main_parser():
   Flags = []
   parser = argparse.ArgumentParser()
   parser.add_argument('-r', '--create',type=bool,default=False,help='To run creator to take face photos')
   parser.add_argument('-d', '--detector',type=bool,default=False,help='To run Detector to recognize faces')
   parser.add_argument('-y', '--yolo',type=bool,default=False,help='To run yolo to detect knife')
   Flags, _ = parser.parse_known_args()

   return  Flags

Flags = main_parser()

#Check if arguments set to true then import Files 
if Flags.create or Flags.detector:
   sys.path.append(os.path.abspath(".\Face_recognize"))
   from Face_recognize import create,detector
if Flags.yolo:
   sys.path.append(os.path.abspath(".\Knife_Detect"))
   from Knife_Detect import yolo

####################
#####   Main  ######
####################
if __name__ == '__main__':
   #if not specify any command then print message
   if Flags.create is False and Flags.yolo is False and Flags.detector is False:
      print("Please specify which code you want to run : \n" +
            "-r OR --create => to start take photos and specify name and id \n"
            +"-d OR --detector => to open webcam and recognize face \n"
            + "-y OR --yolo => to Detect knife \n")


   #run create if true
   if Flags.create :
      create.create()
   #run detector if true
   if Flags.detector :
      p1 = Process(target=detector.detector())
      p1.start()
   #run yolo if true
   if Flags.yolo:
      p2 = Process(target=yolo.play_yolo())
      p2.start()


