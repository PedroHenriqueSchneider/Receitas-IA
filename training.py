import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# Load and preprocess data
data_dir = 'food_data'
image_exts = ['.jpg', '.png', '.jpeg', '.bmp']
data = []
labels = []

for label_name in os.listdir(data_dir):
    label_path = os.path.join(data_dir, label_name)
    if os.path.isdir(label_path):
        label = label_name
        for img_file in os.listdir(label_path):
            if any(ext in img_file for ext in image_exts):
                img_path = os.path.join(label_path, img_file)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (150, 150))
                img = img / 255.0
                data.append(img)
                labels.append(label)

data = np.array(data)
labels = np.array(labels)

# Perform label encoding and one-hot encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
labels_encoded = onehot_encoder.fit_transform(integer_encoded)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

# Build and train CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Prediction on a single test image
test_image_path = 'frango.jpeg'  # Replace with the actual image path
img = image.load_img(test_image_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = model.predict(img_array)
class_index = np.argmax(prediction)
predicted_class = label_encoder.classes_[class_index]

print(f"Predicted class: {predicted_class}")
