import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def preprocessing():
    # Input: Image
    # Output: Smaller image with less noise

    # Setup mp stuff

    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2, # 2 if gestures
        min_detection_confidence=0.5, # 0.3
        )

    # Get images
    images = []
    images_dir = './images'
    for filename in os.listdir(images_dir):
        img = cv2.imread(os.path.join('images', filename))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)
    
    for image in images:
        results = hands.process(image) # landmarks for each image
        if not results.multi_hand_landmarks:
            print("err")
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    return images

if __name__ == "__main__":
    # print("Hello World!")
    images = preprocessing()
    plt.figure()
    plt.imshow(images[0])
    plt.show()