# Utilities for object detector.

import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
from datetime import datetime
import cv2
from utils import label_map_util
from collections import defaultdict
from utils import alertcheck

detection_graph = tf.Graph()


TRAINED_MODEL_DIR = 'frozen_graphs'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
#PATH_TO_CKPT = TRAINED_MODEL_DIR + '/ssd5_optimized_inference_graph.pb'
PATH_TO_CKPT = TRAINED_MODEL_DIR + '/person_frozen_inference_graph.pb'
PATH_TO_CKPT2 = TRAINED_MODEL_DIR + '/mask_frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = TRAINED_MODEL_DIR + '/Glove_label_map.pbtxt'

NUM_CLASSES = 2
# load label map using utils provided by tensorflow object detection api
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

a=b=0

# Load a frozen infrerence graph into memory
def load_inference_graph():

    # load frozen tensorflow model into memory
    
    print("> ====== Loading frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Inference graph loaded.")
    return detection_graph, sess


def load_inference_graph2():
    # load frozen tensorflow model into memory

    print("> ====== Loading frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT2, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Inference graph loaded.")
    return detection_graph, sess
detection_graph2, sess2 = load_inference_graph2()


def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, classes, im_width, im_height, image_np):
    # Determined using a piece of paper of known length, code can be found in distance to camera
    focalLength = 875
    # The average width of a human hand (inches) http://www.theaveragebody.com/average_hand_size.php
    # added an inch since thumb is not included
    avg_width = 4.0
    # To more easily differetiate distances and detected bboxes

    global a,b
    hand_cnt=0
    color = None
    color0 = (0,255,0)
    color1 = (255,0,0)
    person=0
    per_coord=[]
    for i in range(num_hands_detect):

        if (scores[i] > score_thresh):
            person+=1
            # print('person',person,'\ni',i)
            #no_of_times_hands_detected+=1
            #b=b+1
            #b=1
            #print(b)
            if classes[i] == 1:
                id = 'person'
                #b=1
            color = color0
            # if i == 0:
            #     color = color0
            # else:
            #     color = color1

            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,boxes[i][0] * im_height, boxes[i][2] * im_height)
            # print('left',left,'\nright',right,'\ntop',top,'\nbottom',bottom)
            per_coord.append([int(left),int(right),int(top),int(bottom)])
            # print('per_coord', per_coord)


            #0-top,1-left,2-bottom,3-right
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))

            #cv2.line(img=image_np, pt1=(30, 0), pt2=p1, color=(0, 255, 0),thickness=2, lineType=8, shift=0)
            #cv2.line(img=image_np, pt1=(0, 640), pt2=p2, color=(0, 255, 0), thickness=2, lineType=8, shift=0)

            dist = distance_to_camera(avg_width, focalLength, int(right-left))

            if dist:
                hand_cnt=hand_cnt+1
            #cv2.rectangle(image_np, p1, p2, color , 3, 1)
            crop_img=image_np[int(top):int(bottom),int(left):int(right)]
            #cv2.imshow("Detection",crop_img)
            cv2.imwrite(r"C:\Users\Vignesh\PycharmProjects\social_distancing_with_mask\test_images\1{}.jpg".format(i),crop_img)

            boxes2, scores2, classes2 = detect_objects(crop_img, detection_graph2, sess2)
            cr_im_height, cr_im_width = crop_img.shape[:2]
            ith_person=i
            draw_box_on_image2(num_hands_detect, score_thresh, scores2, boxes2, classes2,
                               cr_im_width, cr_im_height, crop_img,ith_person)

            cv2.putText(image_np, 'person '+str(i), (int(left), int(top)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (255,0,0), 2)

            cv2.putText(image_np, 'confidence: '+str("{0:.2f}".format(scores[i])),
                        (int(left),int(top)-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # cv2.putText(image_np, 'distance from camera: '+str("{0:.2f}".format(dist)+' inches'),
            #             (int(im_width*0.65),int(im_height*0.9+30*i)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 2)

           # a=alertcheck.drawboxtosafeline(image_np,p1,p2,Line_Position2,Orientation)
        if hand_cnt==0 :
            b=0
            #print(" no hand")
        else:
            b=1
            #print(" hand")
    #print('per_coord',per_coord)

    # for sorting the person
    sort_per_coord=per_coord
    for i in range(len(sort_per_coord)):
        for j in range(len(sort_per_coord) - 1):
            #print('i ', i, '\nj ', j)
            if sort_per_coord[j][0] > sort_per_coord[j + 1][0]:
                temp = sort_per_coord[j]
                sort_per_coord[j] = sort_per_coord[j + 1]
                sort_per_coord[j + 1] = temp
    # print('sort_per_coord',sort_per_coord)

    #for drawing the line between boxes
    for i in range(len(sort_per_coord) - 1):
        point1=(sort_per_coord[i][1],int((sort_per_coord[i][2]+sort_per_coord[i][3])/2))
        point2=(sort_per_coord[i+1][0],int((sort_per_coord[i+1][2]+sort_per_coord[i+1][3])/2))
        cv2.line(img=image_np, pt1=point1, pt2=point2, color=(255, 0, 0),thickness=2, lineType=8, shift=0)
    # for the color of the box around the person
    for j in range(num_hands_detect):
        if (scores[j] > score_thresh):
            p11=(per_coord[j][0],per_coord[j][2])
            p22=(per_coord[j][1],per_coord[j][3])
            #cv2.rectangle(image_np, p11, p22, color1, 3, 1)
            # print('j num_hands_detect',num_hands_detect)
            # print('k len_per_coord',len(per_coord))
            for k in range(len(per_coord)):

                if per_coord[j][1] in range(per_coord[k][0],per_coord[k][1]):
                    cv2.rectangle(image_np, p11, p22, color1, 3, 1)
                    break
                elif per_coord[j][0] in range(per_coord[k][0] + 1, per_coord[k][1]):
                    cv2.rectangle(image_np, p11, p22, color1, 3, 1)
                    break
                else:
                    cv2.rectangle(image_np, p11, p22, color0, 3, 1)
    return a,b


def draw_box_on_image2(num_hands_detect, score_thresh, scores, boxes, classes, im_width, im_height, image_np,ith_person):
    # Determined using a piece of paper of known length, code can be found in distance to camera
    focalLength = 875
    # The average width of a human hand (inches) http://www.theaveragebody.com/average_hand_size.php
    # added an inch since thumb is not included
    avg_width = 4.0
    # To more easily differetiate distances and detected bboxes

    global a, b
    hand_cnt = 0
    color = None
    color0 = (0, 0, 255)
    color1 = (0, 0, 255)
    for i in range(num_hands_detect):

        if (scores[i] > score_thresh):

            # no_of_times_hands_detected+=1
            # b=b+1
            # b=1
            # print(b)
            if classes[i] == 1:
                id = 'Mask'
                # b=1

            if classes[i] == 2:
                id = 'No_mask'

            if classes[i] == 3:
                id = 'Mask_incorrect'
                 # To compensate bbox size change
                # b=1
            color = color0
            # if i == 0:
            #     color = color0
            # else:
            #     color = color1


            (left, right, top, bottom) = (
            boxes[i][1] * im_width, boxes[i][3] * im_width, boxes[i][0] * im_height, boxes[i][2] * im_height)
            # 0-top,1-left,2-bottom,3-right
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))

            # cv2.line(img=image_np, pt1=(30, 0), pt2=p1, color=(0, 255, 0),thickness=2, lineType=8, shift=0)
            # cv2.line(img=image_np, pt1=(0, 640), pt2=p2, color=(0, 255, 0), thickness=2, lineType=8, shift=0)

            dist = distance_to_camera(avg_width, focalLength, int(right - left))

            if dist:
                hand_cnt = hand_cnt + 1
            cv2.rectangle(image_np, p1, p2, color, 3, 1)

            cv2.putText(image_np, 'person ' + str(ith_person) + ': ' + id, (int(left), int(top) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

            cv2.putText(image_np, 'confidence: ' + str("{0:.2f}".format(scores[i])),
                        (int(left), int(top) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # cv2.putText(image_np, 'distance from camera: ' + str("{0:.2f}".format(dist) + ' inches'),
            #             (int(im_width * 0.65), int(im_height * 0.9 + 30 * i)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 2)

        # a=alertcheck.drawboxtosafeline(image_np,p1,p2,Line_Position2,Orientation)
        if hand_cnt == 0:
            b = 0
            # print(" no hand")
        else:
            b = 1
            # print(" hand")

    return a, b


# Show fps value on image.
def draw_text_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
# compute and return the distance from the hand to the camera using triangle similarity
def distance_to_camera(knownWidth, focalLength, pixelWidth):
    return (knownWidth * focalLength) / pixelWidth

# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)


    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
#    print('\nboxes', boxes, '\nscores', scores, '\nnum', num)
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)
