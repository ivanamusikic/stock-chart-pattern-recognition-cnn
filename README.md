# Stock Chart Pattern Recognition with CNN

An educational convolutional neural network (CNN) implemented in Python with TensorFlow/Keras to classify images of financial price charts as **Double Top** or **Double Bottom**. Developed independently as the CNN component of a university Neural Networks course project.

## Project overview

- **Task:** binary image classification (not financial-price forecasting or a trading strategy).
- **Dataset:** 340 labeled JPEG images, equally divided between Double Top (170) and Double Bottom (170).
- **Input:** RGB images resized to 128 × 128 and normalized to [0, 1].
- **Data split:** 80% training / 20% validation, using a fixed split seed (123).
- **Architecture:** Conv2D (32, 64, 128 filters, 3 × 3) + MaxPooling2D after each layer, Flatten, Dense(128, ReLU), Dropout(0.5), and a two-unit Softmax output.
- **Training:** SGD (learning rate 0.01, momentum 0.9), sparse categorical cross-entropy, 10 epochs.

## Reported results

The course report records **67 correct predictions out of 68 validation images (98.5%, rounded to 99%)**. This result is from the original course run and **has not been independently reproduced for this GitHub package**. The report noted that validation loss rose after approximately epoch 4, a potential sign of overfitting.

| Actual class | Predicted Double Bottom | Predicted Double Top |
|---|---:|---:|
| Double Bottom | 38 | 1 |
| Double Top | 0 | 29 |

**Limitations:** The original implementation uses a training/validation split and does not hold out a separate test set. Small sample size and potentially similar chart images may cause optimistic validation results. These scores do not imply real-world stock-price prediction capability.

## Files

- `main.py`: original university implementation (preserved as submitted).
- `Patterns.csv`: labels and image metadata, with relative paths to original chart images.
- `requirements.txt`: Python dependencies.
- `Patterns/`: 340 dataset images (redistributed with source attribution, per dataset permissions confirmed by the project author).
- `.gitignore`: ignores generated outputs and local development artifacts.

## Running locally

1. Create a Python virtual environment and install the packages in `requirements.txt`.
2. The dataset images are included in `Patterns/`. Keep that directory next to `main.py` and `Patterns.csv`.
3. From the project root, run `python main.py`.

The script creates the class-organized `data/` folder, trains the model, and displays plots and a classification report. It is an interactive research script, not a deployed application.

**Dataset source:** [Stock Chart Patterns on Kaggle, uploaded by mustaphaelbakai](https://www.kaggle.com/datasets/mustaphaelbakai/stock-chart-patterns). Dataset credit goes to its original creator/uploader; the classification model and CNN analysis are separate work. Confirm any relevant dataset license terms before further redistribution.
