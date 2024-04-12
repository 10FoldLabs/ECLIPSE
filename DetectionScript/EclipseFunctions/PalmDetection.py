import numpy as np

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
