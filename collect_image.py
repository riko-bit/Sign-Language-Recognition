import os
import cv2

# Set custom path for saving images
DATA_DIR = '/Users/abhinav./Desktop/SignLanRecognition'
# Create main directory if not exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
# Number of classes and images per class
number_of_classes = 50
dataset_size = 100
# Open default camera
cap = cv2.VideoCapture(0)
# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()
# Loop through classes
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Wait for user to press 'q' to start
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        cv2.putText(frame, 'Ready? Press "Q" ! :) | ESC = Exit',
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif key == 27:  # ESC key
            print("Exited.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Capture dataset_size images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            continue

        cv2.imshow('frame', frame)
        key = cv2.waitKey(25) & 0xFF
        if key == 27:
            print("Exited.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        image_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(image_path, frame)
        counter += 1

print("Data collection complete.")
cap.release()
cv2.destroyAllWindows()