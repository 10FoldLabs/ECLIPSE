import cv2

# create a video capture object
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # convert the image back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 


    # show the original image
    cv2.imshow('ECLIPSE', image) 
    if cv2.waitKey(5) & 0xFF == 27: # exit on ESC
        break

cap.release() # release the video capture object


# destroy all windows
cv2.destroyAllWindows()