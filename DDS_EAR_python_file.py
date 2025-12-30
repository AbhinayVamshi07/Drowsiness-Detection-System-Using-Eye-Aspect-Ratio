import cv2
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import winsound  # For beep sound

# Constants
EYE_AR_THRESH = 0.25  # Eye aspect ratio threshold
EYE_AR_CONSEC_FRAMES = 20  # Number of consecutive frames to trigger drowsiness

# Initialize variables
COUNTER = 0
ALARM_ON = False

# Function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load facial landmarks predictor and initialize dlib's face detector
print("Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Extract indices for eyes from the facial landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Start video stream
print("Starting video stream...")
cap = cv2.VideoCapture(0)  # Webcam feed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    rects = detector(gray, 0)  # Detect faces

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)  # Convert to NumPy array

        # Extract eye regions
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Average EAR for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # Draw contours around the eyes (showing the eye region)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check if EAR is below the threshold
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    print("Drowsiness Detected!")
                    # Beep sound
                    winsound.Beep(1000, 500)  # Frequency=1000Hz, Duration=500ms

                # Draw a red box around the face
                (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box for face

                # Display drowsiness warning
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            ALARM_ON = False

        # Display EAR value on the frame
        cv2.putText(frame, f"Eyes: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Drowsiness Detection", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
