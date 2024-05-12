import math
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


model = get_extract_model()
search_img = "test/test01.jpg"

search_vector = extract_vector(model, search_img)

vector_file = "vectors.pkl"
path_file = "paths.pkl"

vectors = pickle.load(open(vector_file, "rb"))
paths = pickle.load(open(path_file, "rb"))

distance = np.linalg.norm(vectors - search_vector, axis=1)
K = 16
ids = np.argsort(distance)[:K]

nearest_image  = [(paths[id], distance[id]) for id in ids]

import matplotlib.pyplot as plt

axes = []
grid_size = int(math.sqrt(K))
fig = plt.figure(figsize=(10, 5))

for id in range(K):
    draw_image = nearest_image[id]
    axes.append(fig.add_subplot(grid_size, grid_size, id + 1))

    axes[-1].set_title(draw_image[1])
    plt.imshow(Image.open(draw_image[0]))


fig.tight_layout()
plt.show()