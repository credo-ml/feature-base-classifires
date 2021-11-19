import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from function import read_img, create_confusion_matrix, build_cnn_model, create_plot, preprocess_data, compile_and_fit_model
from link_paths import path_save_model, path_compile_model

# Level 2 means to ignore warning and information, and print only the error.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


def create_model(feature_array, label_array, our_model):
    x_std = feature_array
    y = label_array
    x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=0.2, random_state=0)

    # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3])
    # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3])

    train_cat = to_categorical(y_train)
    test_cat = to_categorical(y_test)

    # shape of the input images
    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    print(input_shape, x_train.shape)

    # leaky ReLU
    # lrelu = tf.keras.layers.LeakyReLU(alpha=0.3)

    # create cnn model
    cnn_model = build_cnn_model("relu", input_shape, our_model[1])
    # cnn_model = build_cnn_model(lrelu, input_shape)
    # train cnn model
    print(x_train.shape, 'aaa')
    trained_cnn_model, cnn_history = compile_and_fit_model(cnn_model, x_train, train_cat, x_test, test_cat, 64, 50, our_model)
    # print score
    loss, acc = trained_cnn_model.evaluate(x_test, test_cat)
    print("acc={:.2f} loss={:.2f}".format(acc, loss))

    # PLOT
    create_plot(cnn_history,our_model)

    # confusion_matrix
    # make predictions for test data
    # y_pred = trained_cnn_model.predict_classes(X_test)
    y_pred = np.argmax(trained_cnn_model.predict(x_test), axis=1)
    # determine the total accuracy
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    # print score
    loss, acc = trained_cnn_model.evaluate(x_test, test_cat)
    print("acc={:.2f} loss={:.2f}".format(acc, loss))
    create_confusion_matrix(y_pred, y_test, our_model)

    # save
    trained_cnn_model.load_weights(path_compile_model+'best_'+our_model[1]+'_model_'+our_model[0]+'.h5')

    test_loss, test_acc = trained_cnn_model.evaluate(x_test, test_cat)
    print('test_loss', test_loss, 'test_acc', test_acc)

    trained_cnn_model.summary()
    trained_cnn_model.save(path_save_model+'CNN_'+our_model[1]+'_'+our_model[0]+'.h5')


def main():
    cnn_model_name = "raw"

    images, targets = read_img()
    feature_array, label_array = np.array(images), np.array(targets)
    size_model = ["small", "big"]

    # RAW MODEL
    for our_size_model in size_model:
        our_model = [cnn_model_name, our_size_model]
        create_model(feature_array, label_array, our_model)


if __name__ == '__main__':
    main()