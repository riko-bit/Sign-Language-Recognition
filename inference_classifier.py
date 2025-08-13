import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load trained model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model.p:", e)
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
else:
    print("Camera started successfully.")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels A–Z
labels_dict = {i: chr(65 + i) for i in range(26)}  # 0: A, 1: B, ..., 25: Z

# Accuracy counters
total_attempts = 0
correct_predictions = 0

# Ask user which letter is being shown
expected_char = input("📝 Enter the gesture you are showing (A–Z): ").upper()

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Camera read failed.")
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

        if data_aux:
            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

                print(f"🔤 Predicted: {predicted_character} | Expected: {expected_char}")

                total_attempts += 1
                if predicted_character == expected_char:
                    correct_predictions += 1

            except Exception as e:
                print("Prediction error:", e)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('n'):
        expected_char = input("📝 Enter new gesture (A–Z): ").upper()

# Accuracy summary
if total_attempts > 0:
    accuracy = (correct_predictions / total_attempts) * 100
    print(f"\nFinal Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_attempts})")
else:
    print("\nNo predictions made.")

cap.release()
cv2.destroyAllWindows()