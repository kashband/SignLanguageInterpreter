import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Determines whether returning edited image (for display purposes),
# or just the necessary hand landmark data.
# Set to True, if you would like to see the data being used.
# DISPLAY = True
DISPLAY = False

def preprocessing(image_path):
    # Input: Image
    # Output: Image with landmarks

    # Setup mp stuff
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2, # 2 if gestures
        min_detection_confidence=0.5, # 0.3
        )

    # Read image and convert to RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process image and identify landmarks
    results = hands.process(image) # landmarks for each image

    if DISPLAY:
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        else:
            print("err")

        return results, image
    else:
        return results

if __name__ == "__main__":
    # image = preprocessing()
    # plt.figure()
    # plt.imshow(image)
    # plt.show()
    pass