from time import sleep

import dlib
import cv2
import mss
import numpy as np

from inference import FacialBeautyPredictor


def select_region(img):
    r = cv2.selectROI('Select the area', img)
    # close window
    cv2.destroyWindow('Select the area')
    return r


def capture_video(top=0, left=0, width=1920, height=1080):
    # Initialize the Dlib face detector
    detector = dlib.get_frontal_face_detector()

    # Start capturing video
    with mss.mss() as sct:
        monitor = {"top": top, "left": left, "width": width, "height": height}
        while True:
            screenshot = np.array(sct.grab(monitor))

            # Convert the screenshot to grayscale
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

            # Detect faces using Dlib
            faces = detector(gray, 1)

            # Draw a rectangle around the faces
            for face in faces:
                x, y, width, height = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(screenshot, (x, y), (x + width, y + height), (255, 0, 0), 2)

                # Extract the face and resize it to the size expected by the gender classifier
                cropped_face = screenshot[y:y + height, x:x + width]
                rating = round(FBP.infer_arr(cropped_face)['beauty'], 2)
                print(rating)

            cv2.imshow('Detected Faces', screenshot)

            # Break the loop if the "q" key is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # wait 3 sec
            sleep(3)


if __name__ == "__main__":
    BEAUTY_CLASSIFICATION_THRESH = 3

    # Load the cascade classifier
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # beauty classifier - Define the model architecture (same as the original)
    FBP = FacialBeautyPredictor(pretrained_model_path='pytorch-models/ComboNet_SCUTFBP5500.pth')

    try:
        with mss.mss() as sct:
            img = np.array(sct.grab({"top": 0, "left": 0, "width": 1920, "height": 1080}))
        # select_region of screen to screenshot
        region = select_region(img)
        capture_video(top=int(region[1]), left=int(region[0]), width=int(region[2]), height=int(region[3]))
    except KeyboardInterrupt:
        # If Ctrl+C is pressed, exit the loop and release resources
        print("Interrupted by user. Exiting...")
    finally:
        # Destroy all windows
        cv2.destroyAllWindows()
