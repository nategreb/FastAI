import cv2
import dlib


def main():
    # Create a VideoCapture object to capture the video from the camera
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    # Initialize the Dlib face detector
    detector = dlib.get_frontal_face_detector()

    try:
        # Loop to display the camera feed with face bounding boxes
        while True:
            # Capture each frame
            ret, frame = cap.read()

            # If the frame is read successfully, process and display it
            if ret:
                # Detect faces using Dlib
                faces = detector(frame, 1)

                # Draw bounding boxes around detected faces
                for face in faces:
                    x, y, width, height = face.left(), face.top(), face.width(), face.height()
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)

                # Display the processed frame
                cv2.imshow('Camera Feed', frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Error: Frame could not be read.")
                break
    except KeyboardInterrupt:
        # If Ctrl+C is pressed, exit the loop and release resources
        print("Interrupted by user. Exiting...")
    finally:
        # Release the VideoCapture object and close all windows
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
