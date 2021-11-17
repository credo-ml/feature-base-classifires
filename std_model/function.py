import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
import glob
from numpy import expand_dims
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import cv2
from sklearn import svm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import os
from joblib import dump, load
import mahotas
import numpy as np
from link_paths import path_hit_images

def read_img():

    dots4 = []
    lines4 = []
    worms4 = []

    dots5 = []
    lines5 = []
    worms5 = []

    artefacts = []


    for img in glob.glob(path_hit_images + "hits_votes_4_Dots/*.png"):
        n = cv2.imread(img)
        dots4.append(n)

    for img in glob.glob(path_hit_images + "hits_votes_5_Dots/*.png"):
        n = cv2.imread(img)
        dots5.append(n)

    dots = dots4 + dots5
    target_dots = [0 for _ in dots]
    target_dots4 = [0 for _ in dots4]

    for img in glob.glob(path_hit_images + "hits_votes_4_Lines/*.png"):
        n = cv2.imread(img)
        lines4.append(n)

    for img in glob.glob(path_hit_images + "hits_votes_5_Lines/*.png"):
        n = cv2.imread(img)
        lines5.append(n)

    lines = lines4 + lines5
    target_lines = [1 for _ in lines]
    target_lines4 = [1 for _ in lines4]

    for img in glob.glob(path_hit_images + "hits_votes_4_Worms/*.png"):
        n = cv2.imread(img)
        worms4.append(n)

    for img in glob.glob(path_hit_images + "hits_votes_5_Worms/*.png"):
        n = cv2.imread(img)
        worms5.append(n)

    worms = worms4 + worms5
    target_worms = [2 for _ in worms]
    target_worms4 = [2 for _ in worms4]

    for img in glob.glob(path_hit_images + "artefacts/*.png"):
        n = cv2.imread(img)
        artefacts.append(n)
    target_artefacts = [3 for _ in artefacts]

    images=dots4+lines4+worms4+artefacts
    targets=target_dots4+target_lines4+target_worms4+target_artefacts

    # images=dots+lines+worms+artefacts
    # targets=target_dots+target_lines+target_worms+target_artefacts

    print(len(images),len(targets))
    print(images[0].shape)
    print(len(dots), len(lines), len(worms), len(artefacts))

    return images,targets


def dataAugmentation2(images, size):
    stack = []
    j = 0
    for img in images:
        j += 1
        data = img_to_array(img, dtype='uint32')

        # print(data.shape)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # print(samples.shape)

        params = {
            # 'zoom_range': [1.0,1.],
            # 'width_shift_range': [-5,5],
            # 'height_shift_range': [-5,5],
            'rotation_range': 20
        }

        datagen = ImageDataGenerator(**params)

        # prepare iterator
        it = datagen.flow(samples, batch_size=1, seed=40)
        for i in range(size):
            batch = it.next()
            batch = batch.astype('uint32')
            stack.append(batch[0].copy())


    return stack

def computeZM(images, targets):
  rows , cols, _ =images[0].shape
  radius=np.sqrt((rows/2)**2+(cols/2.)**2)
  features_train =[]
  for img in images:
    img = img.astype('int32')
    blackwhite=img[:,:,0]+img[:,:,1]+img[:,:,2]
    threshold = blackwhite.mean() + blackwhite.std() * 5
    threshold = threshold if threshold < 100 else 100
    mask = np.where(blackwhite > threshold, 1, 0)
    blackwhite = blackwhite * mask
    zm = mahotas.features.zernike_moments(blackwhite, radius, degree=8, cm=(rows/2., cols/2.))
    zm = np.hstack([zm, blackwhite.mean()])
    features_train.append(zm)
  # print('akuku ',blackwhite.dtype)
  feature_array=np.array(features_train)
  label_array=np.array(targets)
  X_std=feature_array
  return (X_std, label_array)


# KRZ
def augment_train_data(X_train, y_train, augconf=[0, 4, 12, 0]):
    X_train_result = []
    y_train_result = []
    # pobierz kolejno obrazy wg labelek
    for label in set(y_train):
        imgs = []
        for i in range(len(X_train)):
            if y_train[i] == label:
                imgs.append(X_train[i])
        # i je zaugemntuj
        aimgs = dataAugmentation2(imgs, augconf[label])

        X_train_result += imgs + aimgs
        # stosownie rozszerzając listę labelek
        y_train_result += [label for _ in range(len(imgs + aimgs))]
        # print(label, len(imgs), len(aimgs), len(X_train_result), len(y_train_result))
    return X_train_result, y_train_result