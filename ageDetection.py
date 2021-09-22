import numpy as np
import argparse
import time
import cv2

def main():
    # camera variables
    camera = cv2.VideoCapture(0)
    t, testFrame = camera.read()
    (height, width) = testFrame.shape[:2]

    # time variables
    previousTime = time.time()
    pTime = 0

    # face cascades
    faceCascadePath = "haarcascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(faceCascadePath)

    while True:
        ret, frame = camera.read()
        originalFrame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        faces = faceCascade.detectMultiScale(image = blurred,
                                             scaleFactor = 1.1,
                                             minNeighbors = 5,
                                             minSize = (30, 30))

        # calculating fps and elapsed time
        currentTime = time.time()
        fps = 1 / (currentTime - pTime)
        pTime = currentTime
        cv2.putText(frame, f"FPS: {str(int(fps))}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)


        if len(faces) != 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            elapsedTime = currentTime - previousTime
            if int(elapsedTime) == 3:
                time.sleep(3)
                image = originalFrame[x:x+w, y:y+h]
                break

        if len(faces) == 0:
            elapsedTime = 0
            previousTime = time.time()

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    camera.release()

    cv2.imshow("image", image)
    time.sleep(2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()