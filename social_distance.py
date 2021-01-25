import cv2
import argparse
import datetime
from imutils.video import VideoStream
from utils import detector_utils as detector_utils
import pandas as pd
from datetime import date
import xlrd
from xlwt import Workbook
from xlutils.copy import copy 
import numpy as np

lst1=[]
lst2=[]
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
args = vars(ap.parse_args())

detection_graph, sess = detector_utils.load_inference_graph()
#detection_graph2, sess2 = detector_utils.load_inference_graph2()

def save_data(no_of_time_hand_detected, no_of_time_hand_crossed):

    try:   
        today = date.today()
        today=str(today)
        #loc = (r'C:\Users\rahul.tripathi\Desktop\result.xls') 
      
        rb = xlrd.open_workbook('result.xls')
        sheet = rb.sheet_by_index(0) 
        sheet.cell_value(0, 0) 
      
         
        #print(sheet.nrows)
        q=sheet.cell_value(sheet.nrows-1,1)
        
        rb = xlrd.open_workbook('result.xls') 
        #rb = xlrd.open_workbook(loc) 
        wb=copy(rb)
        w_sheet=wb.get_sheet(0)
        
        if q==today:
            w=sheet.cell_value(sheet.nrows-1,2)
            e=sheet.cell_value(sheet.nrows-1,3)
            w_sheet.write(sheet.nrows-1,2,w+no_of_time_hand_detected)
            w_sheet.write(sheet.nrows-1,3,e+no_of_time_hand_crossed)
            wb.save('result.xls')      
        else:
            w_sheet.write(sheet.nrows,0,sheet.nrows)
            w_sheet.write(sheet.nrows,1,today)
            w_sheet.write(sheet.nrows,2,no_of_time_hand_detected)
            w_sheet.write(sheet.nrows,3,no_of_time_hand_crossed)
            wb.save('result.xls')
    except FileNotFoundError:
        today = date.today()
        today=str(today)
         

        # Workbook is created 
        wb = Workbook() 

        # add_sheet is used to create sheet. 
        sheet = wb.add_sheet('Sheet 1') 

        sheet.write(0, 0, 'Sl.No')
        sheet.write(0, 1, 'Date') 
        sheet.write(0, 2, 'Number of times hand detected') 
        sheet.write(0, 3, 'Number of times hand crossed') 
        m=1
        sheet.write(1, 0, m)
        sheet.write(1, 1, today) 
        sheet.write(1, 2, no_of_time_hand_detected) 
        sheet.write(1, 3, no_of_time_hand_crossed) 

        wb.save('result.xls')
        
if __name__ == '__main__':
    # Detection confidence threshold to draw bounding box
    score_thresh = 0.80
    
    # vs = cv2.VideoCapture('rtsp://192.168.1.64')
    # vp=r'C:\Users\Vignesh\PycharmProjects\social_distancing_with_mask\vid4.mp4'
    # vs = cv2.VideoCapture(vp)

    # uncomment 1 of 2 for live video
    # vs = VideoStream(0).start()

    # uncomment 1 of 2 for recorded video
    vs=cv2.VideoCapture(r'C:\Users\Vignesh\PycharmProjects\social_distancing_with_mask\vid6.mp4')
    #Oriendtation of machine    
    #Orientation= 'bt'
	#input("Enter the orientation of hand progression ~ lr,rl,bt,tb :")

    #For Machine
    #Line_Perc1=float(input("Enter the percent of screen the line of machine :"))
    Line_Perc1=float(15)

    #For Safety
    #Line_Perc2=float(input("Enter the percent of screen for the line of safety :"))
    Line_Perc2=float(30)

    # max number of hands we want to detect/track
    num_hands_detect = 10

    # Used to calculate fps
    start_time = datetime.datetime.now()
    num_frames = 0

    im_height, im_width = (None, None)
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    def count_no_of_times(lst):
        x=y=cnt=0
        for i in lst:
            x=y
            y=i
            if x==0 and y==1: 
                cnt=cnt+1
        return cnt 
    try:
        while True:
            #uncomment 2 of 2 for recorded video
            ret, frame = vs.read()

            #uncomment 2 of 2 for live video
            # frame = vs.read()

            # for resizing the frame
            # frame=cv2.resize(frame,(200,200))


            #frame=r'C:\Users\Vignesh\PycharmProjects\social_distancing_with_mask\test_images\1.jpg'
            frame = np.array(frame)
            # print('frame.shape',frame.shape)
            # print('frame',frame)

            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
            if im_height == None:
                im_height, im_width = frame.shape[:2]

            # Convert image to rgb since opencv loads images in bgr, if not accuracy will decrease
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            except:
                print("Error converting to RGB")
            #cv2.line(img=frame, pt1=(0, Line_Position1), pt2=(frame.shape[1], Line_Position1), color=(255, 0, 0), thickness=2, lineType=8, shift=0)
            
            #cv2.line(img=frame, pt1=(0, Line_Position2), pt2=(frame.shape[1], Line_Position2), color=(255, 0, 0), thickness=2, lineType=8, shift=0)
            
            # Run image through tensorflow graph

            boxes, scores, classes = detector_utils.detect_objects(frame, detection_graph, sess)
            #boxes2, scores2, classes2 = detector_utils.detect_objects(frame, detection_graph2, sess2)
            # print('boxes',boxes,'\nscores',scores,'\nclasses',classes)
            #print('boxes',boxes2,'\nscores',scores2,'\nclasses',classes2)

            #Line_Position2=orien_lines.drawsafelines(frame,Orientation,Line_Perc1,Line_Perc2)
            # Draw bounding boxeses and text
            a,b=detector_utils.draw_box_on_image(
                num_hands_detect, score_thresh, scores, boxes, classes, im_width, im_height, frame)
            #a2,b2=detector_utils.draw_box_on_image2(
             #   num_hands_detect, score_thresh, scores2, boxes2, classes2, im_width, im_height, frame,Line_Position2,Orientation)
            lst1.append(a)
            lst2.append(b)
            no_of_time_hand_detected=no_of_time_hand_crossed=0
            # Calculate Frames per second (FPS)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() -
                            start_time).total_seconds()
            fps = num_frames / elapsed_time

            if args['display']:
            
                # Display FPS on frame
                detector_utils.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), frame)
                cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows() 
                    vs.stop()
                    break
        
        no_of_time_hand_detected=count_no_of_times(lst2)
        #no_of_time_hand_detected=b
        no_of_time_hand_crossed=count_no_of_times(lst1)
        #print(no_of_time_hand_detected)
        #print(no_of_time_hand_crossed)
        save_data(no_of_time_hand_detected, no_of_time_hand_crossed) 
        print("Average FPS: ", str("{0:.2f}".format(fps)))
        
    except KeyboardInterrupt: 
        no_of_time_hand_detected=count_no_of_times(lst2)
        no_of_time_hand_crossed=count_no_of_times(lst1)
        today = date.today()
        save_data(no_of_time_hand_detected, no_of_time_hand_crossed)
        print("Average FPS: ", str("{0:.2f}".format(fps)))