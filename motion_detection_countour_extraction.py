# To see explaination visit:- https://www.youtube.com/watch?v=MkcUgPhOlP8
import cv2
import numpy as np

cap = cv2.VideoCapture('videos/street_video.mp4')
#frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))

#frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

#fourcc = cv2.VideoWriter_fourcc('X','V','I','D')

#out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280,720))

ret, frame1 = cap.read()
ret, frame2 = cap.read()
#print(frame1.shape)
while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2) # Find difference between successive frames
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) # color the diff image Gray
    blur = cv2.GaussianBlur(gray, (5,5), 0) # apply a gaussian blur to extract contour
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY) # For Contour extraction
    dilated = cv2.dilate(thresh, None, iterations=3) # Image Processing technique for contour extraction
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Find all the contours present in frame 1

    for contour in contours: # iterate over all the contours in a frame1
        (x, y, w, h) = cv2.boundingRect(contour) # find the bounding points of a contour in a frame

        if cv2.contourArea(contour) < 900: # apply area threshold to detect People among all contours
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2) # draw a green rectangle over valid contours
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3) # put information in frame if "movement" occurs
    #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    #image = cv2.resize(frame1, (1280,720))
    #out.write(image)
    cv2.imshow("feed", frame1)
    frame1 = frame2 # propage A frame, overlapping slidding window technique
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27: # press escape key for 40ms to exit the video
        break

cv2.destroyAllWindows()
cap.release()
#out.release()