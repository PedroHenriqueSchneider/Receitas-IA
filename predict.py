import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder

def predict_class(model, label_encoder, image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    predicted_class = label_encoder.classes_[class_index]

    return predicted_class
if __name__ == "__main__":
    model_path = 'trained_model.h5'
    model = load_model(model_path)

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('label_encoder_classes.npy')

    # Prediction on a single test image
    test_image_path = 'frango2.jpeg'  # Replace with the actual image path
    predicted_class = predict_class(model, label_encoder, test_image_path)
    print(f"Predicted class: {predicted_class}")
