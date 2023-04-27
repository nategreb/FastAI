# Get a reference to webcam
import cv2
from mtcnn import MTCNN


def main():
    # first camera or webcam
    video_capture = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not video_capture.isOpened():
        raise Exception("Error: Camera could not be opened.")

    detector = MTCNN()

    try:
        while video_capture.isOpened():
            # capture each frame of the video
            ret, frame = video_capture.read()

            # if the frame is read successfully, process and display it
            if ret:
                # detect faces using MTCNN model
                faces = detector.detect_faces(frame)

                # Draw bounding boxes around detected faces
                for face in faces:
                    x, y, width, height = face['box']
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)

                # display the frame
                cv2.imshow('Camera Feed', frame)

                # Press Q on keyboard to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                print("Error: frame couldn\'t be read")
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
