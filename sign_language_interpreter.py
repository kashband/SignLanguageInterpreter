import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Model setup
model = Sequential([
    Conv2D(8, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
    Conv2D(16, kernel_size=(3, 3), activation='relu'),
    Dropout(0.5),
    MaxPooling2D(pool_size=(4, 4), strides=(4, 4)),
    Flatten(),
    Dense(128, activation='tanh'),
    Dense(25, activation='softmax')
])

# Load the weights
model_path = './model.h5'
model.load_weights(model_path)
print("Weights loaded successfully into the model.")

# Label dictionary
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
    6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
    12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z'
}

# Video capture setup
capture = cv2.VideoCapture(0)

# Create a Hands instance
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

while True:
    _, frame = capture.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Drawing landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Extract landmarks for bounding box
            x_arr = [landmark.x for landmark in hand_landmarks.landmark]
            y_arr = [landmark.y for landmark in hand_landmarks.landmark]
            x1, y1 = int(min(x_arr) * W) - 75, int(min(y_arr) * H) - 75
            x2, y2 = int(max(x_arr) * W) + 75, int(max(y_arr) * H) + 75

            # Crop and process the image
            hand_image = frame[max(y1, 0):min(y2, H), max(x1, 0):min(x2, W)]
            hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)
            hand_image = cv2.resize(hand_image, (28, 28))
            hand_image = np.expand_dims(hand_image, axis=-1) / 255.0
            hand_image = np.expand_dims(hand_image, axis=0)

            # Prediction
            prediction = model.predict(hand_image)
            predicted_index = np.argmax(prediction)
            predicted_character = labels_dict.get(predicted_index, '?')

            # Testing confidence display
            confidence = np.max(prediction)
            confidence_percent = confidence * 100

            # Display the bounding box and prediction
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{predicted_character} {confidence_percent:.2f}%', (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 2)

    # Display the result
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Cleanup
capture.release()
cv2.destroyAllWindows()
