import numpy as np
import cv2
import mediapipe as mp
def angle_between(p1, p2, p3):
    # calculate the angle between three points
    v1 = np.array(p1) - np.array(p2) # vector from p2 to p1
    v2 = np.array(p3) - np.array(p2) # vector from p2 to p3
    dot = np.dot(v1, v2) # dot product of v1 and v2
    norm = np.linalg.norm(v1) * np.linalg.norm(v2) # product of norms of v1 and v2
    cos = dot / norm # cosine of the angle
    angle = np.degrees(np.arccos(cos)) # angle in degrees
    return angle

def is_palm_splayed(landmarks):
    # check if the palm is splayed open
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



mpHands = mp.solutions.hands # create a MediaPipe Hands object
hands = mpHands.Hands() # create a hands object
mpDraw = mp.solutions.drawing_utils # create a drawing object

cap = cv2.VideoCapture(0) # create a video capture object

while cap.isOpened():
    success, image = cap.read() # read a frame from the camera
    if not success: # if the frame is not valid, break the loop
        break
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert the image to RGB
    results = hands.process(image) # process the image using the hands object
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # convert the image back to BGR
    if results.multi_hand_landmarks: # if there are any hand landmarks detected
        for hand_landmarks in results.multi_hand_landmarks: # for each hand
            mpDraw.draw_landmarks(image, hand_landmarks, mpHands.HAND_CONNECTIONS) # draw the landmarks and connections
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark] # get the landmarks as a list of tuples
            splayed = is_palm_splayed(landmarks) # check if the palm is splayed open
            if splayed: # if the palm is splayed open
                cv2.putText(image, "Splayed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # put text "Splayed" on the image
            else: # if the palm is not splayed open
                cv2.putText(image, "Not Splayed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # put text "Not Splayed" on the image
    cv2.imshow('MediaPipe Hands', image) # show the image
    if cv2.waitKey(5) & 0xFF == 27: # exit on ESC
        break

cap.release() # release the video capture object
cv2.destroyAllWindows() # destroy all windows
