import sys
import os
import cv
import matplotlib.pyplot as plt


# Validate image extension.
# Inputs: File extension.
# Output: Boolean. True, if extension in list of valid image extensions.
_EXTENSIONS = ["PNG", "JPG", "JPEG"]
def is_valid_file_extension(_EXTENSION):
    return _EXTENSION in _EXTENSIONS


def main():
    # Ensure that executable has all required parameters.
    assert len(sys.argv) == 2, "Insufficient argument count. Args. expected: 2: <.py executable> <path to image folder>."

    # Obtain path to folder containing images.
    _PATH = sys.argv[1]

    # Obtain files in folder.
    assert os.path.exists(_PATH), "Path does not exist."
    _FILES = os.listdir(_PATH)

    processed_images = []

    with open('./dump.txt', 'w') as _DUMP:
        for _FILE in _FILES:
            file_split = _FILE.split('.')
            if len(file_split) == 2 and is_valid_file_extension(file_split[1].upper()):
                processed_images.append(cv.preprocessing(_PATH + '/' + _FILE))
            else:
                _DUMP.write('File \"' + _FILE + '\" has invalid file extension. (Valid file extensions include PNG, JPG/JPEG)\n')

    # Pass procssed images to model
    plt.figure()
    plt.imshow(processed_images[0])
    plt.show()

if __name__ == '__main__':
    main()

