import tensorflow as tf
import numpy as np
from PIL import Image

def load_pneumonia_model(path):
    return tf.keras.models.load_model(path)

def preprocess(img):
    img = img.resize((224, 224)).convert("RGB")
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0).astype(np.float32)

def predict_image(model, img_array):
    prediction = model.predict(img_array)[0][0]
    if prediction > 0.5:
        return "PNEUMONIA", prediction
    else:
        return "NORMAL", 1 - prediction