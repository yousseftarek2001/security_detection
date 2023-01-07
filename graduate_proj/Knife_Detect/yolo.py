import numpy as np
import argparse
import cv2 as cv
import subprocess
import time


def play_yolo():
    FLAGS = parser()
    net,labels,colors , layer_names = load_yolo()
    # If both image and video files are given then open webcam
    if FLAGS.image_path is None and FLAGS.video_path is None:
            print('Neither path to an image or path to video provided')
            print('Starting Inference on Webcam')

        # Do inference with given image
    if FLAGS.image_path:
            # Read the image
            try:
                img = cv.imread(FLAGS.image_path)
                height, width = img.shape[:2]
            except:
                raise 'Image cannot be loaded!\n\
                                Please check the path provided!'

            finally:
                img, _, _, _, _ = infer_image(net, layer_names, height, width, img, colors, labels, FLAGS)
                show_image(img)

    elif FLAGS.video_path:
            # Read the video
            try:
                vid = cv.VideoCapture(FLAGS.video_path)
                height, width = None, None
                writer = None
            except:
                raise 'Video cannot be loaded!\n\
                                Please check the path provided!'

            finally:
                while True:
                    grabbed, frame = vid.read()

                    # Checking if the complete video is read
                    if not grabbed:
                        break

                    if width is None or height is None:
                        height, width = frame.shape[:2]

                    frame, _, _, _, _ = infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS)

                    if writer is None:
                        # Initialize the video writer
                        fourcc = cv.VideoWriter_fourcc(*"MJPG")
                        writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30,
                                                (frame.shape[1], frame.shape[0]), True)

                    writer.write(frame)

                print("[INFO] Cleaning up...")
                writer.release()
                vid.release()



    else:
            # Infer real-time on webcam
            count = 0
            vid = cv.VideoCapture(0)
            while True:
                _, frame = vid.read()
                height, width = frame.shape[:2]

                if count == 0:
                    frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
                                                                            height, width, frame, colors, labels, FLAGS)
                    count += 1
                else:
                    frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
                                                                            height, width, frame, colors, labels, FLAGS,
                                                                            boxes, confidences, classids, idxs, infer=False)
                    count = (count + 1) % 6

                cv.imshow('webcam', frame)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            vid.release()
            cv.destroyAllWindows()
def parser():
    FLAGS = []
    parser = argparse.ArgumentParser()
    # load weights file
    parser.add_argument('-w', '--weights',
                        type=str,
                        default='.\Knife_Detect\yolov3.weights',
                        help='Path to the file which contains the weights \
                                        for YOLOv3.')
    # load config file
    parser.add_argument('-cfg', '--config',
                        type=str,
                        default='.\Knife_Detect\yolov3.cfg',
                        help='Path to the configuration file for the YOLOv3 model.')
    # take the image path
    parser.add_argument('-i', '--image-path',
                        type=str,
                        help='The path to the image file')
    # take the video path
    parser.add_argument('-v', '--video-path',
                        type=str,
                        help='The path to the video file')
    # store the proccessed video to file
    parser.add_argument('-vo', '--video-output-path',
                        type=str,
                        default='.\Knife_Detect\output\output.avi',
                        help='The path of the output video file')
    # load labels file
    parser.add_argument('-l', '--labels',
                        type=str,
                        default='.\Knife_Detect\coco.names',
                        help='Path to the file having the \
    					labels in a new-line seperated way.')
    # set confidence
    parser.add_argument('-c', '--confidence',
                        type=float,
                        default=0.5,
                        help='The model will reject boundaries which has a \
    				probabiity less than the confidence value. \
    				default: 0.5')
    # set threshold
    parser.add_argument('-th', '--threshold',
                        type=float,
                        default=0.3,
                        help='The threshold to use when applying the \
    				Non-Max Suppresion')
    # download model if the model weights and configurations are not present on your local machine.
    parser.add_argument('--download-model',
                        type=bool,
                        default=False,
                        help='Set to True, if the model weights and configurations \
    				are not present on your local machine.')
    # show time YOLO took to infer image
    parser.add_argument('-t', '--show-time',
                        type=bool,
                        default=False,
                        help='Show the time taken to infer each image.')
    # store known arguments in FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS

def show_image(img):
    cv.imshow("Image", img)
    cv.waitKey(0)

def load_yolo():
    # Get the labels
    FLAGS = parser()
    labels = open(FLAGS.labels).read().strip().split('\n')

    # Intializing colors to represent each label uniquely
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Load the weights and configutation to form the pretrained YOLOv3 model
    net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

    # Get the output layer names of the model
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, labels, colors, layer_names

def draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels):
    # If there are any detections
    if len(idxs) > 0:
        for i in idxs.flatten():
            # Get the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            # Get the unique color for this class
            color = [int(c) for c in colors[classids[i]]]

            # Draw the bounding box rectangle and label on the image
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
            cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img

def generate_boxes_confidences_classids(output, height, width, tconf):
    boxes = []
    confidences = []
    classids =[]
    # loop over each of the layer outputs
    for out in output:
        # loop over each of the detections
        for detection in out:
            # Get the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]

            # Consider only the predictions that are above a certain confidence level
            if confidence > tconf:
                # TODO Check detection
                if classid == 43:
                    box = detection[0:4] * np.array([width, height, width, height])
                    centerX, centerY, bwidth, bheight = box.astype('int')

                    # Using the center x, y coordinates to derive the top
                    # and the left corner of the bounding box
                    x = int(centerX - (bwidth / 2))
                    y = int(centerY - (bheight / 2))
                    
                    # Append to list
                    boxes.append([x, y, int(bwidth), int(bheight)])
                    confidences.append(float(confidence)*100)
                    classids.append(classid)

    return boxes, confidences, classids

def infer_image(net, layer_names, height, width, img, colors, labels, FLAGS,
                boxes=None, confidences=None, classids=None, idxs=None, infer=True):
    if infer:
        # Contructing a blob from the input image
        blob = cv.dnn.blobFromImage(img, 1 / 255.0, (192, 192),
                                    swapRB=True, crop=False)

        # Perform a forward pass of the YOLO object detector
        net.setInput(blob)

        # Getting the outputs from the output layers
        start = time.time()
        outs = net.forward(layer_names)
        end = time.time()

        if FLAGS.show_time:
            print("[INFO] YOLOv3 took {:6f} seconds".format(end - start))

        # Download the YOLOv3 models if needed
        if FLAGS.download_model:
            subprocess.call(['./yolov3-coco/get_model.sh'])

        # Generate the boxes, confidences, and classIDs
        boxes, confidences, classids = generate_boxes_confidences_classids(outs, height, width, FLAGS.confidence)

        # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
        idxs = cv.dnn.NMSBoxes(boxes, confidences, FLAGS.confidence, FLAGS.threshold)

    if boxes is None or confidences is None or idxs is None or classids is None:
        raise '[ERROR] Required variables are set to None before drawing boxes on images.'
        # Draw labels and boxes on the image
    img = draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels)

    return img, boxes, confidences, classids, idxs


