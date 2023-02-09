import tensorflow as tf
from tensorflow import keras
from keras.models import Model
# Librerias de ayuda
import numpy as np
import os
import PIL
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
from sklearn.decomposition import PCA

model = keras.models.load_model('model.h5')
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("flatten").output)

images_path = '/home/findmine/Desktop/envatxel/pyimagesearch-ropa/Banana_Republic/Mens'
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


features = []
for i, image_path in enumerate(images):
    img, x = load_image(image_path);
    feat = feat_extractor.predict(x)[0]
    features.append(feat)

print('finished extracting features for %d images' % len(images))

features = np.array(features)
np.save('features.npy',features)