import cv2
import numpy as np

def preprocess_palm_image(palm_image):
    # Convert to grayscale
    gray = cv2.cvtColor(palm_image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to binarize the image
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

def skeletonize(image):
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)

    _, img = cv2.threshold(image, 128, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skel

def detect_minutiae(skeleton):
    minutiae_points = []

    for i in range(1, skeleton.shape[0] - 1):
        for j in range(1, skeleton.shape[1] - 1):
            if skeleton[i, j] == 255:
                neighbor_pixels = [skeleton[i-1, j-1], skeleton[i-1, j], skeleton[i-1, j+1],
                                    skeleton[i, j-1], skeleton[i, j+1],
                                    skeleton[i+1, j-1], skeleton[i+1, j], skeleton[i+1, j+1]]

                if sum(neighbor_pixels) == 255:
                    minutiae_points.append((j, i))  # Note the reversal of coordinates

    return minutiae_points

# Example usage with video stream
cap = cv2.VideoCapture(0)  # You may need to adjust the video source index

while True:
    ret, frame = cap.read()

    # Preprocess the palm image
    palm_binary = preprocess_palm_image(frame)

    # Skeletonize the binary image
    palm_skeleton = skeletonize(palm_binary)

    # Detect minutiae in the skeletonized image
    minutiae_points = detect_minutiae(palm_skeleton)

    # Draw minutiae points on the original frame
    for point in minutiae_points:
        cv2.circle(frame, point, 3, (0, 255, 0), -1)

    cv2.imshow('Palm Minutiae Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()

