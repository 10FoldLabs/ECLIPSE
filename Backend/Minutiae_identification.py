import cv2
import mediapipe as mp
import numpy as np


mpHands = mp.solutions.hands # create a MediaPipe Hands object
hands = mpHands.Hands() # create a hands object
mpDraw = mp.solutions.drawing_utils # create a drawing object
model = cv2.dnn.readNetFromTensorflow('\EDSRx2.pb')
cap = cv2.VideoCapture(0) # create a video capture object

def process_frame(frame):
# create a named window for the cropped image
    # create a named window for the cropped image
    cv2.namedWindow("Cropped", cv2.WINDOW_AUTOSIZE)
    # get the bounding box coordinates and dimensions from the hand landmarks
    x, y, w, h = cv2.boundingRect(landmarksNP)
    # draw a rectangle around the bounding box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cut the bounding box from the original image
    cropped = image[y:y+h, x:x+w]
    # apply image processing to the cropped image
    # convert to grayscale
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    # apply histogram equalization
    cropped = cv2.equalizeHist(cropped)
    # apply Canny edge detector
    cropped = cv2.Canny(cropped, 90, 100)
    # apply opening operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    cropped = cv2.morphologyEx(cropped, cv2.MORPH_OPEN, kernel)
    # apply closing operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    cropped = cv2.morphologyEx(cropped, cv2.MORPH_CLOSE, kernel)
    # apply Gaussian blur to the cropped image
    cropped = cv2.GaussianBlur(cropped, (5, 5), 0)

    # show the cropped image
    return cropped

def angle_between(p1, p2, p3):
    # calculate the angle between three points as before
    v1 = np.array(p1) - np.array(p2) # vector from p2 to p1
    v2 = np.array(p3) - np.array(p2) # vector from p2 to p3
    dot = np.dot(v1, v2) # dot product of v1 and v2
    norm = np.linalg.norm(v1) * np.linalg.norm(v2) # product of norms of v1 and v2
    cos = dot / norm # cosine of the angle
    angle = np.degrees(np.arccos(cos)) # angle in degrees
    return angle

def is_palm_splayed(landmarks):
    # check if the palm is splayed open as before
    # landmarks is a list of 21 (x, y) coordinates of the hand
    # the order of the landmarks is defined by MediaPipe Hands
    # https://google.github.io/mediapipe/solutions/hands.html

    # define the threshold angles for each finger
    # you can change these values according to your preference
    thumb_angle = 30 # angle between thumb, index finger, and wrist
    index_angle = 10 # angle between index finger, middle finger, and palm
    middle_angle = 10 # angle between middle finger, ring finger, and palm
    ring_angle = 10 # angle between ring finger, pinky finger, and palm
    pinky_angle = 10 # angle between pinky finger, palm, and wrist

    # calculate the angles for each finger
    thumb = angle_between(landmarks[4], landmarks[8], landmarks[0])
    index = angle_between(landmarks[8], landmarks[12], landmarks[9])
    middle = angle_between(landmarks[12], landmarks[16], landmarks[13])
    ring = angle_between(landmarks[16], landmarks[20], landmarks[17])
    pinky = angle_between(landmarks[20], landmarks[18], landmarks[0])

    # check if all the angles are greater than the thresholds
    if thumb > thumb_angle and index > index_angle and middle > middle_angle and ring > ring_angle and pinky > pinky_angle:
        return True # palm is splayed open
    else:
        return False # palm is not splayed open

fps = 0
start_time = cv2.getTickCount()

while cap.isOpened():
    success, image = cap.read() # read a frame from the camera
    if not success: # if the frame is not valid, break the loop
        break
    
    blob = cv2.dnn.blobFromImage(image, scalefactor=2.0, size=(0, 0), mean=(0, 0, 0), swapRB=True, crop=False)
    model.setInput(blob)
    upscaled_frame = model.forward()[0]
    
    image = cv2.cvtColor(upscaled_frame, cv2.COLOR_BGR2RGB) # convert the image to RGB
    results = hands.process(image) # process the image using the hands object
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # convert the image back to BGR
    CroppedWindow = False
    if results.multi_hand_landmarks: # if there are any hand landmarks detected
        for hand_landmarks in results.multi_hand_landmarks: # for each hand
            landmarks = [(int(lm.x * image.shape[1]), int(lm.y * image.shape[0])) for lm in hand_landmarks.landmark] # get the landmarks as a list of tuples
            landmarksNP = [(int(lm.x * image.shape[1]), int(lm.y * image.shape[0])) for lm in hand_landmarks.landmark]
            landmarksNP = np.array(landmarks, dtype=np.int32)
            splayed = is_palm_splayed(landmarks) # check if the palm is splayed open
            if splayed: # if the palm is splayed open
                # create a named window for the cropped image
                cv2.namedWindow("Cropped", cv2.WINDOW_AUTOSIZE)
                # get the bounding box coordinates and dimensions from the hand landmarks
                x, y, w, h = cv2.boundingRect(landmarksNP)
                w = w + 30

                print(f'x = {x :^8} |  {y :^8}')
 
                # draw a rectangle around the bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cut the bounding box from the original image
                if x > 0 and y > 0:
                    cropped = image[y:y+h, x:x+w]
                    processed = process_frame(cropped)
                # you can use the cropped variable as a new image stream to run through another function later
                # for example, you can process the cropped image using the process_frame function
                
                
                # show the cropped image and the processed image
                cv2.imshow("Cropped", cropped)
                cv2.imshow("Processed", processed)

    cv2.imshow('MediaPipe Hands', image) # show the original image
    if cv2.waitKey(5) & 0xFF == 27: # exit on ESC
        break


end_time = cv2.getTickCount()
elapsed_time = (end_time - start_time) / cv2.getTickFrequency()
average_fps = fps / elapsed_time
print(f"Average FPS: {average_fps:.2f}")

cap.release() # release the video capture object
cv2.destroyWindow("Cropped") # destroy the window for the cropped image
cv2.destroyAllWindows() # destroy all windows
