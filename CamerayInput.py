import cv2
from IPA import process_frame

cap = cv2.VideoCapture(0) # create a video capture object

while True:
    ret, frame = cap.read() # read a frame
    if ret: # check if the frame is valid
        processed = process_frame(frame) # process the frame using the function
        cv2.imshow("Processed", processed) # show the processed frame
        if cv2.waitKey(1) & 0xFF == ord('q'): # exit on q
            break
    else: # if the frame is not valid, break the loop
        break

cap.release() # release the video capture object
cv2.destroyAllWindows() # destroy all windows

