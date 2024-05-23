import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

# Load the datasets
labels_df = pd.read_csv('example/input/classification_labels.csv')
ancestors_df = pd.read_csv('example/input/name_to_ancestors.csv')

# Define the directory containing the images
image_dir = 'example/input/images'  # Adjust this path as needed

# Construct full image paths
labels_df['image_path'] = labels_df['basename'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))

# Preprocessing function
def preprocess_images(image_paths):
    images = []
    for path in image_paths:
        image = cv2.imread(path)
        if image is not None:
            image = cv2.resize(image, (128, 128))  # Resize to a fixed size
            images.append(image)
        else:
            print(f"Warning: Unable to read image at {path}")
    images = np.array(images)
    images = images / 255.0  # Normalize the images
    return images

# Load and preprocess the images
image_paths = labels_df['image_path'].values
images = preprocess_images(image_paths)

# Encode the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels_df['deepest_name'])
labels = to_categorical(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(labels_df['deepest_name'])), activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save predictions to CSV
predictions = model.predict(X_test)
predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
output_df = pd.DataFrame({'image_path': image_paths[:len(X_test)], 'predicted_species': predicted_labels})
output_df.to_csv('example/output/predictions.csv', index=False)