import os
import matplotlib.pyplot as plt
from pathlib import Path
from cv import preprocessing


# Places all similar coordinates into one array. (I.e., all x's together, all y's together, ...)
#
# Inputs: List of landmarks of a particular image, of format [{x: , y: , z: }].
# Outputs: Individual lists of landmarks.
def get_coords_as_lists(landmarks):
    x = [landmark.x for landmark in landmarks]
    y = [landmark.y for landmark in landmarks]
    z = [landmark.z for landmark in landmarks]
    return x, y, z


# Generates train and test CSV. Converts images to more useful
# format.
#
# Inputs: None.
# Outputs: None (implied population of CSV files).
def generate_csv():
    # Change path to your data source.
    PATH = str(Path.home() / 'Downloads')
    TRAIN = '/archive/asl_alphabet_train/asl_alphabet_train'
    TEST = '/archive/asl_alphabet_test/asl_alphabet_test'

    # Generate train CSV.
    with open('train.csv', 'w') as file:
        _CATEGORIES = os.listdir(PATH + TRAIN)
        for _CATEGORY in _CATEGORIES:
            _IMAGES = os.listdir(PATH + TRAIN + '/' + _CATEGORY)
            for _IMAGE in _IMAGES:
                results = preprocessing(PATH + TRAIN + '/' + _CATEGORY + '/' + _IMAGE)
                if results.multi_hand_landmarks:
                    landmarks = results.multi_hand_landmarks[0].landmark
                    x, y, z = get_coords_as_lists(landmarks)
                    if len(x) == len(y) == len(z):
                        file.write(str(x) + ',' + str(y) + ',' + str(z) + ',' + _CATEGORY + '\n')
    
    # Generate test CSV.
    with open('test.csv', 'w') as file:
            _IMAGES = os.listdir(PATH + TEST)
            for _IMAGE in _IMAGES:
                results = preprocessing(PATH + TEST + '/' + _IMAGE)
                if results.multi_hand_landmarks:
                    landmarks = results.multi_hand_landmarks[0].landmark
                    x, y, z = get_coords_as_lists(landmarks)
                    if len(x) == len(y) == len(z):
                        file.write(str(x) + ',' + str(y) + ',' + str(z) + ',' + _IMAGE[:_IMAGE.find('_')] + '\n')


if __name__ == '__main__':
    generate_csv()