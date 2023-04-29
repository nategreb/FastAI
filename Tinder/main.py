import cv2
import mss
import numpy
import numpy as np


def select_region(img):
    r = cv2.selectROI('Select the area', img)
    return r


def capture_video(top=0, left=0, width=1920, height=1080):
    # Start capturing video
    with mss.mss() as sct:
        monitor = {"top": top, "left": left, "width": width, "height": height}

        try:
            while True:
                # Get the screenshot
                img = numpy.array(sct.grab(monitor))

                # Convert the screenshot to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Detect faces
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                # Draw a rectangle around the faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Display the screenshot
                cv2.imshow("Face Detection", img)

                # Break the loop if the "q" key is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            # If Ctrl+C is pressed, exit the loop and release resources
            print("Interrupted by user. Exiting...")
        finally:
            # Destroy all windows
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Load the cascade classifier
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    with mss.mss() as sct:
        img = np.array(sct.grab({"top": 0, "left": 0, "width": 1920, "height": 1080}))

    # select_region of screen to screenshot
    region = select_region(img)

    capture_video(top=int(region[1]), left=int(region[0]), width=int(region[2]), height=int(region[3]))
