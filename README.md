# SignLanguageInterpreter
Sign Language Interpreter project utilizing Computer Vision

run `sign_language_interpreter.py`, the primary file and uses the saved `model.h5` and `model.json` tensorflow model, this is the real-time interpreter that utilizes a webcam and will require imported libraries

`50epochs.h5`, `model.h5` : saved tensorflow models

`cv.py`, `data.py`, `main.py` : was used for computer vision preprocessing (feature detection and matching)

`kagglemodel.ipynb` : main tensorflow model notebook, using Kaggle dataset

`model.ipynb` : preliminary model, unused

`video.py` : testing real-time video capture

`sign_mnist_train.csv`, `sign_mnist_test.csv` : image dataset in csv form, not used for final model's training

The `images/` folder contains a handful of images from the dataset for quick testing. The dataset is gitignored due to it's sheer size with approx. 3000 images per character.
