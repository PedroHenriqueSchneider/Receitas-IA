import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def preprocess_data(data_dir):
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

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    labels_encoded = onehot_encoder.fit_transform(integer_encoded)

    X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoder

def train_and_save_model(X_train, y_train, X_test, y_test, num_classes, save_path):
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
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

    model.save(save_path)

if __name__ == "__main__":
    data_dir = 'food_data'
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(data_dir)

    save_path = 'trained_model.h5'
    train_and_save_model(X_train, y_train, X_test, y_test, num_classes=len(label_encoder.classes_), save_path=save_path)
