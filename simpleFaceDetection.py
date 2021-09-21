import time
import cv2

def main():
    # camera variables
    camera = cv2.VideoCapture(0)
    ret, testFrame = camera.read()
    (height, width) = testFrame.shape[:2]

    # haarcascades variables
    frontalFaceCascadePath = "haarcascades/haarcascade_frontalface_default.xml"
    frontalFaceCascade = cv2.CascadeClassifier(frontalFaceCascadePath)

    # time variables
    currentTime = 0
    previousTime = 0

    while True:
        t, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        faces = frontalFaceCascade.detectMultiScale(image = blurred,
                                                    scaleFactor = 1.1,
                                                    minNeighbors = 5,
                                                    minSize = (30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # calculating the FPS
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(frame, f"FPS: {str(int(fps))}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.putText(frame, f"Number of faces detected: {str(int(len(faces)))}", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),3)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
