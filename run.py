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
from google.oauth2 import service_account
from google.cloud import storage
import requests
from io import BytesIO
# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = storage.Client(credentials=credentials)

st.title('Tensorflow/Fashion Mnist soft tunning')

with open('features.npy','rb') as f:
    features = np.load(f)

bucket = client.bucket('atx_banana_republic')
images_path = bucket.list_blobs(prefix='Banana_Republic/Mens')

#images_path = '/home/findmine/Desktop/envatxel/pyimagesearch-ropa/Banana_Republic/Mens'
image_extensions = ['.jpg', '.png', '.jpeg']   # case-insensitive (upper/lower doesn't matter)
max_num_images = 10000

images = list(images_path)
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
    result_imx = [f'https://storage.cloud.google.com/atx_banana_republic/{images[idx].name}' for idx in indexes]
    st.image(result_imx,width=300)


if st.button('Query aleatorio'):
    query_image_idx = int(len(images) * random.random())
    # do a query on a random image
    im=f'https://storage.cloud.google.com/atx_banana_republic/{images[query_image_idx].name}'
    st.image(im,"Query")    

    idx_closest = get_closest_images(query_image_idx)
    results_image = get_concatenated_images(idx_closest, 200)
    