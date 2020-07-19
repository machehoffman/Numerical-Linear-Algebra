import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from datetime import datetime
from skimage.io import imread
from skimage.transform import resize
import csv
import numpy as np
import os
import ksvd
# raw_train, metadata = tfds.load('DeepWeeds',  with_info=True,as_supervised=True)
# get_label_name = metadata.features['label'].int2str
IMG_DIRECTORY = "./images/"
LABEL_DIRECTORY = "./labels/"
OUTPUT_DIRECTORY = "./outputs/"
# Global variables
RAW_IMG_SIZE = (256, 256)
IMG_SIZE = (224, 224)
INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)
CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8]
CLASS_NAMES = ['Chinee Apple',
               'Lantana',
               'Parkinsonia',
               'Parthenium',
               'Prickly Acacia',
               'Rubber Vine',
               'Siam Weed',
               'Snake Weed',
               'Negatives']


def inference(model,dataframe):
  # Load DeepWeeds dataframe
  image_count = dataframe.shape[0]
  filenames = dataframe.Filename
  labels = dataframe.Label

  feature_vect_list = []
  labels_vect_list = []
  for i in range(image_count):
    # Load image
    img = imread(IMG_DIRECTORY + filenames[i])
    # Resize to 224x224
    img = resize(img, (224, 224))
    # Map to batch
    img = np.expand_dims(img, axis=0)
    # Scale from int to float
    img = img * 1. / 255
    # Predict label
    feature_vect = conv_base.predict(img)
    feature_vect_np = np.array(feature_vect)
    feature_vect_list.append(feature_vect_np.flatten())
    labels_vect = np.array(np.int(labels[i]))
    labels_vect_list.append(labels_vect)
  feature_vect_list_np = np.array(feature_vect_list)
  labels_vect_list_np = np.array(labels_vect_list)

  return feature_vect_list_np,labels_vect_list_np


if __name__ == '__main__':

    conv_base = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    conv_base.summary()
    dataframe = pd.read_csv(LABEL_DIRECTORY + "labels.csv")
    labels = dataframe.Label

    feature_file = 'feature_np.npy'
    lable_file ='labels_vect_list_np.npy'
    if not os.path.isfile(feature_file):
        feature_np,labels_vect_list_np = inference(conv_base,dataframe)
        np.save('feature_np.npy', feature_np)
        np.save('labels_vect_list_np.npy', labels_vect_list_np)
    else:
        feature_np = np.load(feature_file)
        labels_vect_list_np = np.load(lable_file)
    for i in range(len(CLASSES)):
        tmp_indices = np.argwhere(labels_vect_list_np==i)

        sliced_feturmatrix=feature_np[np.squeeze(tmp_indices),:]
        aksvd = ksvd.ApproximateKSVD(n_components=512)
        dictionary = aksvd.fit(sliced_feturmatrix).components_ # dose not work!!!
        gamma = aksvd.transform(sliced_feturmatrix)
        a=1