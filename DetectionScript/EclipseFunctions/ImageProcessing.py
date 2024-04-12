import cv2
import numpy as np

def enhance_frame(frame, model:cv2.dnn_superres.DnnSuperResImpl):
    upscaled_frame = model.upsample(frame)
    return upscaled_frame

def filter_ridges(ridge_orientation): # (EXPERIMENTAL)
    # Compute ridge frequency using the absolute values of gradients 
    ridge_frequency = np.sqrt(np.square(cv2.Sobel(ridge_orientation, cv2.CV_64F, 1, 0, ksize=5)) + np.square(cv2.Sobel(ridge_orientation, cv2.CV_64F, 0, 1, ksize=5)))
    
    # Apply threshold to get binary ridge mask
    _, ridge_mask = cv2.threshold(ridge_frequency, 90, 255, cv2.THRESH_BINARY)
    
    return ridge_mask

def boostcontrast(img):
    # converting to LAB color space
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel (try different values [limit,grid size])
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge L, a and b channels and convert
    limg = cv2.merge((cl,a,b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img


def process_frame(frame, landmarksNP, model:cv2.dnn_superres.DnnSuperResImpl):
    # create window for the cropped image
    cv2.namedWindow("Cropped", cv2.WINDOW_AUTOSIZE)
    
    # draw a rectangle around the hand landmarks
    x, y, w, h = cv2.boundingRect(landmarksNP)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cut the bounding box from the original image
    cropped = frame[y:y+h, x:x+w]

    # apply image processing to the cropped image
    cropped = enhance_frame(cropped, model)
    cropped = boostcontrast(cropped)

    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    cropped = cv2.equalizeHist(cropped)
    cropped = cv2.bitwise_not(cropped)

    # show the cropped image
    return cropped