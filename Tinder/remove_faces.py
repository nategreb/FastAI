import cv2
import dlib

input_path = '../../tinder-demo.mp4'
output_path = '../../tinder-edited-demo.mp4'

# initialize dlibs face detector
detector = dlib.get_frontal_face_detector()

# To capture video from a file
cap = cv2.VideoCapture(input_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    # Read the frame
    ret, frame = cap.read()

    # If frame is read correctly ret is True
    if not ret:
        break

    colored = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Detect the faces
    faces = detector(colored)

    # Draw the rectangle around each face and black them out
    for rect in faces:
        x = rect.left()
        y = rect.top()
        w = rect.width()
        h = rect.height()
        frame[y:y+h, x:x+w] = [0, 0, 0]

    # Write the frame into the file 'output_video.mp4'
    out.write(frame)

# Release the VideoWriter and VideoCapture objects
cap.release()
out.release()
