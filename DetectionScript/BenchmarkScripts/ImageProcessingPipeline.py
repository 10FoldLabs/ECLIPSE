import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import EclipseFunctions.PalmDetection as pd
import EclipseFunctions.ImageProcessing as ip



# create a video capture object
cap = cv2.VideoCapture(0) 
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# create a MediaPipe Hands object
hands = mp.solutions.hands.Hands() 

# load super sampling model
sr = cv2.dnn_superres.DnnSuperResImpl.create()
sr.readModel("FSRCNN_x4.pb")
sr.setModel("fsrcnn",4)


# used to record the time at which we processed current frame 
new_frame_time = 0
prev_frame_time = 0
# Our operations on the frame come here 
font = cv2.FONT_HERSHEY_SIMPLEX 
fps = 0
start_time = cv2.getTickCount()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    font = cv2.FONT_HERSHEY_SIMPLEX 

    # convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # process the image using the hands object
    results = hands.process(image) 
    # convert the image back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 

    if results.multi_hand_landmarks: 
        # for each hand get the landmarks as a list of tuples
        for hand_landmarks in results.multi_hand_landmarks: # 
            
            landmarks = [(int(lm.x * image.shape[1]), int(lm.y * image.shape[0])) for lm in hand_landmarks.landmark] 
            landmarksNP = [(int(lm.x * image.shape[1]), int(lm.y * image.shape[0])) for lm in hand_landmarks.landmark]
            landmarksNP = np.array(landmarks, dtype=np.int32)
            splayed = pd.is_palm_splayed(landmarks) # check if the palm is splayed open
            if splayed: # if the palm is splayed open create window for the cropped image

                # get the bounding box coordinates and dimensions from the hand landmarks
                x, y, w, h = cv2.boundingRect(landmarksNP)
                print(f'x = {x :^8} |  {y :^8}')
 
                # draw a rectangle around the bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cut box from original image
                if x > 0 and y > 0:
                    cropped = image[y:y+h, x:x+w]
                    processed = ip.process_frame(image, landmarksNP, sr)

                    cv2.imshow("Processed", processed)

    # FPS DISPLAY (NOT FINISHED)
    # new_frame_time = time.time() 
    # fps = 1/(new_frame_time-prev_frame_time) 
    # prev_frame_time = new_frame_time 
    # fps = int(fps) 
    # fps = str(fps)


    # show the original image
    cv2.imshow('ECLIPSE', image) 
    fps += 1
    if cv2.waitKey(5) & 0xFF == 27: # exit on ESC
        break

end_time = cv2.getTickCount()
elapsed_time = (end_time - start_time) / cv2.getTickFrequency()

average_fps = fps / elapsed_time

print(f"Average FPS: {average_fps:.2f}")

cap.release() # release the video capture object

# destroy all windows
cv2.destroyAllWindows()