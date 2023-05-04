from time import sleep

import dlib
import cv2
import mss
import numpy as np
import pyautogui
import keyboard
from inference import FacialBeautyPredictor


def select_region(img):
    r = cv2.selectROI('Select the area', img)
    # close window
    cv2.destroyWindow('Select the area')
    return r


def like():
    pyautogui.moveTo(profile_position[0], profile_position[1])
    pyautogui.mouseDown()
    pyautogui.moveTo(profile_position[0] + x_offset, profile_position[1], duration=0.75)
    pyautogui.mouseUp()


def dislike():
    pyautogui.moveTo(profile_position[0], profile_position[1])
    pyautogui.mouseDown()
    pyautogui.moveTo(profile_position[0] - x_offset, profile_position[1], duration=0.75)
    pyautogui.mouseUp()


def next_profile_pic():
    pyautogui.click(profile_position[0] + 60, profile_position[1])


def capture_video(top=0, left=0, width=1920, height=1080):
    # Initialize the Dlib face detector
    detector = dlib.get_frontal_face_detector()

    MAX_NO_FACE = 3

    # Start capturing video
    with mss.mss() as sct:
        monitor = {"top": top, "left": left, "width": width, "height": height}
        while True:
            sleep(3)

            screenshot = np.array(sct.grab(monitor))

            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

            # Detect faces using Dlib
            faces = detector(screenshot, 1)

            print(f'faces detect: {len(faces)}')

            if not faces:
                if MAX_NO_FACE == 0:
                    dislike()
                    MAX_NO_FACE = 3
                else:
                    MAX_NO_FACE -= 1
                    next_profile_pic()
            elif len(faces) > 1:
                dislike()
            else:
                # 1 face
                face = faces[0]
                x, y, width, height = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(screenshot, (x, y), (x + width, y + height), (255, 0, 0), 2)

                # Extract the face and resize it to the size expected by the gender classifier
                cropped_face = screenshot[y:y + height, x:x + width]
                rating = round(FBP.infer_arr(cropped_face)['beauty'], 2)

                # Display the text above the bounding box
                cv2.putText(screenshot, str(rating), (x + 30, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                cv2.imshow('Rating', screenshot)
                # trying to move the window exactly to the left of ROI box selected
                #cv2.moveWindow('Rating', left-3*width, top)

                cv2.waitKey(3000)


                if rating > BEAUTY_CLASSIFICATION_THRESHOLD:
                    like()
                else:
                    dislike()

            # Break the loop if the "q" key is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    BEAUTY_CLASSIFICATION_THRESHOLD = 2.5

    # beauty classifier - Define the model architecture (same as the original)
    FBP = FacialBeautyPredictor(pretrained_model_path='pytorch-models/ComboNet_SCUTFBP5500.pth')

    x_offset = 800

    #Ask the user to manually move the mouse to the profile location

    print("Move the mouse cursor to the profile location within 5 seconds")
    #sleep(5)
    profile_position = pyautogui.position()
    print(f'profile position: {profile_position}')


    try:
        with mss.mss() as sct:
            img = np.array(sct.grab({"top": 0, "left": 0, "width": 3449, "height": 1440}))
        # select_region of screen to screenshot
        region = select_region(img)
        capture_video(top=int(region[1]), left=int(region[0]), width=int(region[2]), height=int(region[3]))
    except KeyboardInterrupt:
        # If Ctrl+C is pressed, exit the loop and release resources
        print("Interrupted by user. Exiting...")
    finally:
        # Destroy all windows
        cv2.destroyAllWindows()
