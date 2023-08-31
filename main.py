from train_model import preprocess_data, train_and_save_model
from predict import predict_class
import numpy as np
from tensorflow.keras.models import load_model

def main():
    data_dir = 'food_data'
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(data_dir)

    save_path = 'trained_model.h5'
    train_and_save_model(X_train, y_train, X_test, y_test, num_classes=len(label_encoder.classes_), save_path=save_path)

    # Salvar as classes mapeadas pelo LabelEncoder
    np.save('label_encoder_classes.npy', label_encoder.classes_)

    model = load_model(save_path)

    # Prediction on a single test image
    test_image_path = 'frango.jpeg'  # Replace with the actual image path
    predicted_class = predict_class(model, label_encoder, test_image_path)
    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    main()
