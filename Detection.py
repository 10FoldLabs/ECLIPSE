import cv2
import numpy as np

def preprocess_palm_image(palm_image):
    # Convert to grayscale
    gray = cv2.cvtColor(palm_image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Sobel operator to compute gradients
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    
    # Compute ridge orientation
    ridge_orientation = np.arctan2(sobely, sobelx)
    
    return ridge_orientation

def filter_ridges(ridge_orientation):
    # Compute ridge frequency using the absolute values of gradients
    ridge_frequency = np.sqrt(np.square(cv2.Sobel(ridge_orientation, cv2.CV_64F, 1, 0, ksize=5)) +
                              np.square(cv2.Sobel(ridge_orientation, cv2.CV_64F, 0, 1, ksize=5)))
    
    # Apply threshold to get binary ridge mask
    _, ridge_mask = cv2.threshold(ridge_frequency, 50, 255, cv2.THRESH_BINARY)
    
    return ridge_mask

def skeletonize(image):
    _, img = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    skel = np.zeros_like(img, np.uint8)
    
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        done = cv2.countNonZero(img) == 0

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
    ridge_orientation = preprocess_palm_image(frame)

    # Filter ridges using ridge orientation
    ridge_mask = filter_ridges(ridge_orientation)

    # Detect minutiae in the skeletonized image
    minutiae_points = detect_minutiae(ridge_mask)

    # Draw minutiae points on the original frame
    for point in minutiae_points:
        cv2.circle(frame, point, 3, (0, 255, 0), -1)

    cv2.imshow('Palm Minutiae Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
