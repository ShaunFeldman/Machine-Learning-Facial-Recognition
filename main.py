import threading
import cv2
from deepface import DeepFace

# Open the default camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False

# Load the reference image
reference_img = cv2.imread("reference1.png")

# Define text properties
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.2
THICKNESS = 3

def check_face(frame):
    # Runs in a separate thread to compare the current camera frame
    # against the reference image. Updates the global face_match variable.

    global face_match
    try:
        result = DeepFace.verify(frame, reference_img.copy())  
        face_match = result["verified"]
    except Exception as e:
        # If DeepFace fails (e.g no face detected), treat it as no match
        face_match = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Flip the frame horizontally so it feels like a front-facing camera
    frame = cv2.flip(frame, 1)

    # Every 60 frames, spawn a thread to check if there's a face match
    if counter % 60 == 0:
        threading.Thread(target=check_face, args=(frame.copy(),)).start()
    counter += 1

    # Decide what text to draw based on face_match
    if face_match:
        text = "MATCH"
        color = (0, 255, 0)    # green
    else:
        text = "NO MATCH"
        color = (0, 0, 255)    # red

    # Calculate text size (so I can draw a background rectangle)
    (text_width, text_height), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
    x, y = 20, 450

    # Draw a filled white rectangle behind the text
    cv2.rectangle(
        frame,
        (x, y - text_height - 10),
        (x + text_width + 10, y + baseline),
        (255, 255, 255),  # white fill
        -1
    )

    # put the text over that rectangle
    cv2.putText(frame, text, (x, y), FONT, FONT_SCALE, color, THICKNESS)

    #Show the frame in a window
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to exit program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
