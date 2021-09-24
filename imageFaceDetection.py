import argparse
import time
import cv2

# added the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True)
ap.add_argument("-o", "--output", required = False)
args = vars(ap.parse_args())

# processing the image
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# cascades
frontalFaceCascadePath = "haarcascades/haarcascade_frontalface_default.xml"
frontalFaceCascade = cv2.CascadeClassifier(frontalFaceCascadePath)

# detecting the faces
faces = frontalFaceCascade.detectMultiScale(image = image,
                                            scaleFactor = 1.1,
                                            minNeighbors = 5,
                                            minSize = (30, 30))

if len(faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    print(f"i've detected {len(faces)} faces")
else:
    print("i did not detect any face")

cv2.putText(image, "https://github.com/VladTomici14", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
cv2.imshow("detected faces", image)
cv2.imwrite(args["output"], image)

cv2.waitKey(0)
cv2.destroyAllWindows()