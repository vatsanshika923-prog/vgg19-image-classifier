import numpy as np
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions

# Load pretrained model
model = VGG19(weights="imagenet")

def preprocess(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def predict(img_path):
    img = preprocess(img_path)
    preds = model.predict(img)
    return decode_predictions(preds, top=3)[0]