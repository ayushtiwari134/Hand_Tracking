"""imports"""
import cv2  # opencv-python library.
import mediapipe as mp  # google mediapipe library for hand tracking and more.
import time  # library to display the frame rate on the screen.

"""code"""
# use this as a canvas to capture the video.
cap = cv2.VideoCapture(0)

# create a hand-detection model object.
mpHand = mp.solutions.hands

# create a hand object which detects the hands.
hands = mpHand.Hands()

# mediapipe's inbuilt function to draw the traced connections.
mpDraw = mp.solutions.drawing_utils

# variables to calculate the frame rate displayed on the screen
pTime = 0
cTime = 0

"""
We use an infinite while loop here to continuously track the moving hands. 
If no while loop is used, only one image is captured by the camera and so it will be of no use. 
"""
while True:

    # read the image that is being captured by the camera.
    success, img = cap.read()

    # convert the captured image per second into RGB format for the mediapipe model to understand.
    # function - cv2.cvtColor(image that you want to convert, format to convert it into).
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # transfer the detected hands on the screen (maximum 2 by default) into an array called results.
    results = hands.process(imgRGB)

    # to check if hands are detected or not we use multi_hand_landmarks in results list.
    if results.multi_hand_landmarks:

        # we iterate over each 'hand' present in the results list to draw the connections b/w traced points on the
        # palms and detect the location of each single landmark.
        for hand in results.multi_hand_landmarks:
            # to find the position of every single landmark in the list provided.
            for id, lm in enumerate(hand.landmark):
                # to find the height width and depth of image.
                h, w, c = img.shape
                # to convert the coordinates of the landmarks into pixels to draw on the image(img).
                xc, yc = int((lm.x * w)), int((lm.y * h))

                # to detect any landmarks and draw circles on it to point it out
                if id == 4:
                    # function - cv2.circle(image where you want to display, coordinates, radius, color, filled or not)
                    cv2.circle(img, (xc, yc), 15, (255, 10, 255), cv2.FILLED)

            """
            this function draws the connections and dots on the hands that are detected by the camera and displays 
            them over the images of the hands.
            """
            mpDraw.draw_landmarks(img, hand, mpHand.HAND_CONNECTIONS)

    # to calculate the fps
    cTime = time.time()
    fps = str(int(1 / (cTime - pTime)))
    pTime = cTime

    # to display the frame rate on the top left of the screen
    # function - cv2.putText (image you want to display on, text you want to display,
    # coordinates in pixels, font, font-size, color, font-weight)
    cv2.putText(img, str("fps:" + fps), (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 10, 255), 2)

    # after the tracking and tracing is completed, we display the image and the traced connections on the screen.
    # function - cv2.imshow(name of file, file you want to display)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
