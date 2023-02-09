# TensorFlow y tf.keras
import tensorflow as tf
from tensorflow import keras

# Librerias de ayuda
import random
import numpy as np
import streamlit as st
import os
import time
import PIL
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
from sklearn.decomposition import PCA
from scipy.spatial import distance
st.title('Tensorflow/Fashion Mnist soft tunning')

with open('features.npy','rb') as f:
    features = np.load(f)

images_path = 'dataset'
image_extensions = ['.jpg', '.png', '.jpeg']   # case-insensitive (upper/lower doesn't matter)
max_num_images = 10000

images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if os.path.splitext(f)[1].lower() in image_extensions]
if max_num_images < len(images):
    images = [images[i] for i in sorted(random.sample(xrange(len(images)), max_num_images))]

print("keeping %d images to analyze" % len(images))

def load_image(path):
    img = PIL.Image.open(path).resize((28,28))
    x = np.asarray(img)[:,:,0]
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x



pca = PCA(n_components=300)
pca.fit(features)
pca_features = pca.transform(features)
def get_closest_images(query_image_idx, num_results=5):
    distances = [ distance.cosine(pca_features[query_image_idx], feat) for feat in pca_features ]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results+1]
    return idx_closest

def get_concatenated_images(indexes, thumb_height):
    thumbs = []
    for idx in indexes:
        img = PIL.Image.open(images[idx])
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image

if st.button('Query aleatorio'):
    query_image_idx = int(len(images) * random.random())
    # do a query on a random image

    idx_closest = get_closest_images(query_image_idx)
    query_image = get_concatenated_images([query_image_idx], 300)
    results_image = get_concatenated_images(idx_closest, 200)

    # display the query image
    im=images[query_image_idx]
    st.image(im,"Query")

    # display the resulting images
    st.image(results_image,'Resultado')