import os

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

from PIL import Image
import pickle
import numpy as np


def get_extract_model():
    model = VGG16(weights='imagenet')
    return Model(inputs=model.input, outputs=model.get_layer('fc1').output)

def image_preprocess(img):
    img = img.convert('RGB')
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def extract_vector(model, img_path):
    print("Proccesing Image: ", img_path)
    img = Image.open(img_path)
    img_tensor = image_preprocess(img)

    vector = model.predict(img_tensor)[0]
    vector = vector / np.linalg.norm(vector)
    return vector

data_folder = "dataset"

model = get_extract_model()
vectors = []
paths = []

for img_path in os.listdir(data_folder):
    img_path_full = os.path.join(data_folder, img_path)
    img_vector = extract_vector(model, img_path_full)

    vectors.append(img_vector)
    paths.append(img_path_full)


vector_file = "vectors.pkl"
path_file = "paths.pkl"

pickle.dump(vectors, open(vector_file, "wb"))
pickle.dump(paths, open(path_file, "wb"))



