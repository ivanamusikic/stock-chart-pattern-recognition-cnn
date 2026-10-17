import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Load image labels from the CSV file
df = pd.read_csv("Patterns.csv")

print(df.head())
# Check the number of samples per class
print(df["ClassName"].value_counts())

# Visualize the class distribution
plt.figure(figsize=(6,4))
sns.countplot(x="ClassName", data=df)
plt.title("Number of samples per class")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

output_folder = "data"
images_folder = "Patterns"

# Organize images into class-specific directories
for index, row in df.iterrows():
    class_name = row["ClassName"]
    image_name = os.path.basename(row["Path"])

    class_folder = os.path.join(output_folder, class_name)
    os.makedirs(class_folder, exist_ok=True)# Create the directory if it does not exist

    src = os.path.join(images_folder, image_name)
    dst = os.path.join(class_folder, image_name)

    if os.path.exists(src):
        shutil.copy(src, dst)


train_ds = tf.keras.utils.image_dataset_from_directory(
    "data",
    validation_split=0.2, # 80% training, 20% validation
    subset="training",
    seed=123,
    image_size=(128,128),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "data",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128,128),
    batch_size=32
)

class_names = train_ds.class_names
print("Classes:", class_names)

# Scale pixel values from [0, 255] to [0, 1]
normalization_layer = layers.Rescaling(1./255)

# Apply normalization to images while keeping labels unchanged

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Dropout helps reduce overfitting
    layers.Dense(len(class_names), activation='softmax')
])
print("Summary: ")
model.summary()

model.compile(
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    loss='sparse_categorical_crossentropy', # Labels are integer-encoded rather than one-hot encoded
    metrics=['accuracy']
)





history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# Compare training and validation loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()

y_true = []
y_pred = []

for images, labels in val_ds:
    predictions = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.show()

print(classification_report(y_true, y_pred, target_names=class_names))

# Collect correctly and incorrectly classified validation samples
correct_images = []
correct_labels = []
correct_preds = []

wrong_images = []
wrong_labels = []
wrong_preds = []

for images, labels in val_ds:
    predictions = model.predict(images)
    preds = np.argmax(predictions, axis=1)

    for i in range(len(labels)):
        if labels[i] == preds[i]:
            correct_images.append(images[i])
            correct_labels.append(labels[i])
            correct_preds.append(preds[i])
        else:
            wrong_images.append(images[i])
            wrong_labels.append(labels[i])
            wrong_preds.append(preds[i])

plt.figure(figsize=(10,5))

for i in range(min(5, len(correct_images))):
    plt.subplot(1,5,i+1)
    plt.imshow(correct_images[i])
    plt.title(f"True: {class_names[correct_labels[i]]}\nPred: {class_names[correct_preds[i]]}")
    plt.axis("off")

plt.suptitle("Correctly classified images")
plt.show()

plt.figure(figsize=(10,5))

for i in range(min(5, len(wrong_images))):
    plt.subplot(1,5,i+1)
    plt.imshow(wrong_images[i])
    plt.title(f"True: {class_names[wrong_labels[i]]}\nPred: {class_names[wrong_preds[i]]}")
    plt.axis("off")

plt.suptitle("Misclassified images")
plt.show()