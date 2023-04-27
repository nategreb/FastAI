# Get a reference to webcam
import cv2


def main():
    # first camera or webcam
    video_capture = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not video_capture.isOpened():
        raise Exception("Error: Camera could not be opened.")

    try:
        while video_capture.isOpened():
            # capture each frame of the video
            ret, frame = video_capture.read()

            if ret:
                # display the frame
                cv2.imshow('Camera Feed', frame)

                # Press Q on keyboard to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                break
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")

    finally:
        # release the video capture object at the end
        video_capture.release()
        # close all the frames
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
