import cv2
import glob
import mahotas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

from link_paths import path_hit_images,path_compile_model


def read_img():
    dots = []
    lines = []
    worms = []
    artefacts = []

    for img in glob.glob(path_hit_images+"hits_votes_4_Dots/*.png"):
        n = cv2.imread(img)
        dots.append(n)
    target_dots = [0 for _ in dots]

    for img in glob.glob(path_hit_images+"hits_votes_4_Lines/*.png"):
        n = cv2.imread(img)
        lines.append(n)
    target_lines = [1 for _ in lines]

    for img in glob.glob(path_hit_images+"hits_votes_4_Worms/*.png"):
        n = cv2.imread(img)
        worms.append(n)
    target_worms = [2 for _ in worms]

    for img in glob.glob(path_hit_images+"artefacts/*.png"):
        n = cv2.imread(img)
        artefacts.append(n)
    target_artefacts = [3 for _ in artefacts]

    images = dots+lines+worms+artefacts

    #####################################################################
    target_signals_binary = [0 for _ in (dots+lines+worms)]
    target_artefacts_binary = [1 for _ in artefacts]
    targets = target_signals_binary+target_artefacts_binary

    print(len(images),len(targets))
    print(images[0].shape)
    print(len(dots), len(lines), len(worms), len(artefacts))

    return images, targets


def preprocess_data(data, wavelets=(2,), verbose=True):
    images, targets = data

    features = []
    bl_images = []
    th_images = []

    for img in images:

        img = img.astype('int32')

        blackwhite = img[:, :, 0]+img[:, :, 1]+img[:, :, 2]
        bl_images.append(blackwhite.copy())

        threshold = blackwhite.mean() + blackwhite.std() * 5
        threshold = threshold if threshold < 100 else 100

        mask = np.where(blackwhite > threshold, 1, 0)
        blackwhite = blackwhite * mask

        th_images.append(blackwhite.copy())

        # Transform using Dx Wavelets to obtain transformed images

        img_x, img_y, img_z = img.shape
        # print("blackwhite: ",blackwhite.shape)
        if img_x == img_y:
            # we want img 60x60
            layers = {
                'raw': img.reshape(img_x, img_y, 3),
                0: blackwhite.reshape(img_x, img_y, 1),
                2: mahotas.daubechies(blackwhite, 'D2').reshape(img_x, img_y, 1),
                4: mahotas.daubechies(blackwhite, 'D4').reshape(img_x, img_y, 1),
                6: mahotas.daubechies(blackwhite, 'D6').reshape(img_x, img_y, 1),
                8: mahotas.daubechies(blackwhite, 'D8').reshape(img_x, img_y, 1),
                10: mahotas.daubechies(blackwhite, 'D10').reshape(img_x, img_y, 1),
                12: mahotas.daubechies(blackwhite, 'D12').reshape(img_x, img_y, 1),
                14: mahotas.daubechies(blackwhite, 'D14').reshape(img_x, img_y, 1),
                16: mahotas.daubechies(blackwhite, 'D16').reshape(img_x, img_y, 1),
                18: mahotas.daubechies(blackwhite, 'D18').reshape(img_x, img_y, 1),
                20: mahotas.daubechies(blackwhite, 'D20').reshape(img_x, img_y, 1)
            }

            # print(layers)
            # tt = np.concatenate((t02, t04, t06, t08), axis=2)
            out = np.concatenate(tuple(map(layers.__getitem__, wavelets)), axis=2)

            features.append(out)
        # else:
        #     print(img_x, img_y, img_z)  # size img != 60,60,3

    feature_array, label_array = np.array(features), np.array(targets)

    if verbose:
        print(feature_array.shape)
        print(label_array.shape)

    return feature_array, label_array, th_images, bl_images


def compile_and_fit_model(model, x_train, y_train, x_test, y_test, batch_size, n_epochs, our_model, verbose=True):
    # compile the model
    model.compile(
        # optimizer=tf.keras.optimizers.Adam(),
        optimizer=tf.keras.optimizers.RMSprop(),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    model.save(path_compile_model+our_model[1]+'_init.model_'+our_model[0]+'.h5')

    # define callbacks
    callbacks = [ModelCheckpoint(filepath=path_compile_model+'best_'+our_model[1]+'_model_'+our_model[0]+'.h5',
                                 monitor='val_accuracy', save_best_only=True)]

    # fit the model
    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=n_epochs,
                        verbose=verbose,
                        callbacks=callbacks,
                        validation_data=(x_test, y_test))

    return model, history


def build_cnn_model(activation, input_shape, type_size="small", verbose=True,):
    model = Sequential()

    if type_size == "big":
        # BIG MODEL
        # 2 Convolution layer with Max polling
        model.add(Conv2D(32, 5, activation=activation, padding='same', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.2))
        model.add(Conv2D(64, 5, activation=activation, padding='same', kernel_initializer="he_normal"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.2))
        model.add(Conv2D(128, 5, activation=activation, padding='same', kernel_initializer="he_normal"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.2))
        model.add(Conv2D(128, 5, activation=activation, padding='same', kernel_initializer="he_normal"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())

    else:
        # SMALL MODEL
        # 2 Convolution layer with Max polling
        model.add(Conv2D(16, 3, activation=activation, padding='same', input_shape=input_shape))
        model.add(Conv2D(16, 5, activation=activation, padding='same', kernel_initializer="he_normal"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.2))
        model.add(Conv2D(16, 3, activation=activation, padding='same', kernel_initializer="he_normal"))
        model.add(Conv2D(16, 3, activation=activation, padding='same', kernel_initializer="he_normal"))
        model.add(Conv2D(16, 5, activation=activation, padding='same', kernel_initializer="he_normal"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.2))
        model.add(Conv2D(16, 3, activation=activation, padding='same', kernel_initializer="he_normal"))
        model.add(Conv2D(16, 3, activation=activation, padding='same', kernel_initializer="he_normal"))
        model.add(Conv2D(16, 5, activation=activation, padding='same', kernel_initializer="he_normal"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.2))
        model.add(Conv2D(16, 3, activation=activation, padding='same', kernel_initializer="he_normal"))
        model.add(Conv2D(16, 3, activation=activation, padding='same', kernel_initializer="he_normal"))
        model.add(Conv2D(16, 5, activation=activation, padding='same', kernel_initializer="he_normal"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())

    # 3 Full connected layer
    model.add(Dense(300, activation = activation, kernel_initializer="he_normal",
                    kernel_regularizer=tf.keras.regularizers.l1(0.01),
                    activity_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(Dense(50, activation = activation, kernel_initializer="he_normal",
                    kernel_regularizer=tf.keras.regularizers.l1(0.01),
                    activity_regularizer=tf.keras.regularizers.l2(0.01)))
    #############################################################################################
    # number of  classes
    model.add(Dense(2, activation='softmax'))
    #############################################################################################
    # summarize the model
    if verbose:
        print(model.summary())
    return model


def create_confusion_matrix(y_pred, y_test, our_model):
    cm = confusion_matrix(y_test, y_pred)
    # Normalise
    cmn = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(5, 5))
    target_names = [0, 1]
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    # plt.show(block=False)

    plt.savefig(path_compile_model+our_model[1]+'_model_'+our_model[0]+'.png')


def create_plot(cnn_history, our_model):

    acc = cnn_history.history['accuracy']
    loss = cnn_history.history['loss']

    val_acc = cnn_history.history['val_accuracy']
    val_loss = cnn_history.history['val_loss']

    epochs = range(len(acc))

    plt.style.use('dark_background')
    plt.figure()

    plt.plot(epochs, acc, 'r', label='Accuracy of training')
    plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.title('loss')
    plt.legend()

    plt.savefig(path_compile_model+our_model[1]+'_model_'+our_model[0]+'.png')
