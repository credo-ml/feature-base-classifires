import os
import cv2
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


home = os.path.abspath(os.path.dirname(__file__))

PATH_HIT_IMAGES = os.path.join(
    home, "hit-images"
)  # path to directory containing rated (divided into classes) detection images
PATH_SAVE_MODEL = os.path.join(home, "model")  # path where we will save trained models


def loadData(path_hit_images, verbose=True):
    dots = []
    lines = []  # track
    worms = []
    artefacts = []

    for img in glob.glob(os.path.join(path_hit_images, "hits_votes_4_Dots", "*.png")):
        n = cv2.imread(img)
        dots.append(n)
    target_dots = [0 for _ in dots]

    for img in glob.glob(os.path.join(path_hit_images, "hits_votes_4_Lines", "*.png")):
        n = cv2.imread(img)
        lines.append(n)
    target_lines = [1 for _ in lines]

    for img in glob.glob(os.path.join(path_hit_images, "hits_votes_4_Worms", "*.png")):
        n = cv2.imread(img)
        worms.append(n)
    target_worms = [2 for _ in worms]

    for img in glob.glob(os.path.join(path_hit_images, "artefacts", "*.png")):
        n = cv2.imread(img)
        artefacts.append(n)
    target_artefacts = [3 for _ in artefacts]

    images = dots + lines + worms + artefacts

    target_dots_binary = [0 for _ in dots]
    target_lines_binary = [1 for _ in lines]
    target_worms_binary = [2 for _ in worms]
    target_artefacts_binary = [3 for _ in artefacts]

    targets = (
        target_dots_binary
        + target_lines_binary
        + target_worms_binary
        + target_artefacts_binary
    )

    if verbose:
        print(len(images), len(targets))
        print(images[0].shape)
        print(len(dots), len(lines), len(worms), len(artefacts))

    return images, targets


def preprocessData(data, verbose=True):
    images, targets = data

    features = []
    for img in images:
        img = img.astype("int32")

        blackwhite = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
        # bl_images.append(blackwhite.copy())

        threshold = blackwhite.mean() + blackwhite.std() * 5
        threshold = threshold if threshold < 100 else 100

        mask = np.where(blackwhite > threshold, 1, 0)
        blackwhite = blackwhite * mask

        # feature #1
        num_pixels_over_th = np.sum(mask)

        # feature #2
        total_luminosity_over_th = np.sum(blackwhite)

        out = (num_pixels_over_th, total_luminosity_over_th)
        features.append(out)

    feature_array, label_array = np.array(features), np.array(targets)

    if verbose:
        print(feature_array.shape)
        print(label_array.shape)

    return feature_array, label_array


class BaseTrigger(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        #         compute minimal luminosity for artefacts and maximal luminosity for signals
        #         compute minimal pix_count for artefacts and maximal pix_count for signals
        self.min_pixcount_arte_ = X[:, 0].max()
        self.min_lum_arte_ = X[:, 1].max()
        self.max_pixcount_sig_ = X[:, 0].min()
        self.max_lum_sig_ = X[:, 1].min()
        print(
            "{} {} {} {}\n".format(
                self.min_pixcount_arte_,
                self.min_lum_arte_,
                self.max_pixcount_sig_,
                self.max_lum_sig_,
            )
        )
        for features, label in zip(X, y):
            pix_count = features[
                0,
            ]
            lum = features[
                1,
            ]
            if label == 0:  # signal
                if pix_count > self.max_pixcount_sig_:
                    self.max_pixcount_sig_ = pix_count
                if lum > self.max_lum_sig_:
                    self.max_lum_sig_ = lum
            else:
                if pix_count < self.min_pixcount_arte_:
                    self.min_pixcount_arte_ = pix_count
                if lum < self.min_lum_arte_:
                    self.min_lum_arte_ = lum

        print(
            "{} {} {} {}".format(
                self.min_pixcount_arte_,
                self.min_lum_arte_,
                self.max_pixcount_sig_,
                self.max_lum_sig_,
            )
        )
        self.border_lum_ = (self.min_lum_arte_ + self.max_lum_sig_) / 2
        self.border_pixcount_ = (self.min_pixcount_arte_ + self.max_pixcount_sig_) / 2
        return self

    def predict(self, X):
        Y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            pix_count = X[i, 0]
            # pix_count=0
            lum = X[i, 1]
            if (pix_count / self.border_pixcount_) ** 2 + (
                lum / self.border_lum_
            ) ** 2 <= 1:
                Y[i] = 0
            else:
                Y[i] = 1
        return Y


def prepare_baseline(feature_array, label_array):
    X_train, X_test, y_train, y_test = train_test_split(
        feature_array, label_array, test_size=0.33, random_state=1
    )

    baseline_list = ["baseline_knn", "baseline_rf"]

    for name2 in baseline_list:
        if name2 == "baseline_knn":
            bt = KNeighborsClassifier(n_neighbors=7)
            bt.fit(X_train, y_train)

        elif name2 == "baseline_rf":
            bt = RandomForestClassifier(max_depth=2, random_state=0)
            bt.fit(X_train, y_train)
        else:  # baseline
            bt = BaseTrigger().fit(X_train, y_train)

        cm, cm_std, acc_mean, acc_std = kf_validation(
            feature_array, label_array, bt, rounds=5, verbose=True
        )
        if not os.path.exists(PATH_SAVE_MODEL):
            os.makedirs(PATH_SAVE_MODEL, exist_ok=True)
        filename = os.path.join(PATH_SAVE_MODEL, f"{name2}.pkl")
        dump(bt, filename)


def kf_validation(X_std, y, _clf_, rounds=1, verbose=False):

    scores = []
    CM = np.zeros_like(np.eye(4))
    cm_seq = []

    for _ in range(rounds):

        kf = KFold(n_splits=5, shuffle=True)
        kf.get_n_splits(X_std)

        for train_index, test_index in kf.split(X_std):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X_std[train_index], X_std[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = _clf_
            clf.fit(X_train, y_train)

            # y_pred = np.argmax(clf.predict(X_test), axis=1)
            y_pred = clf.predict(X_test)
            score = 100 * accuracy_score(y_test, y_pred)
            scores.append(score)
            # print('precision: {:.2f}%'.format(score))

            cm = confusion_matrix(y_test, y_pred)
            cm_seq.append(cm.copy())

    print("\navg. precision: {:.2f}%".format(sum(scores) / len(scores)))

    # cm = CM
    cm = reduce(np.add, cm_seq)
    # print(cm)

    # cm = CM
    # Normalise
    cmn = 100 * cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Std devs
    cm_seq = np.array(cm_seq)
    # cmn_std = 100 * np.divide(cm_seq.std(axis=2), cm_seq.sum(axis=2))
    cumulative = []
    for item in cm_seq:
        c_ = 100 * item.astype("float") / item.sum(axis=1)[:, np.newaxis]
        cumulative.append(c_)

    cmn_std = np.array(cumulative).std(axis=0)

    if verbose:
        fig, ax = plt.subplots(figsize=(6, 5))
        target_names = [0, 1, 2, 3]
        sns.heatmap(
            cmn_std,
            annot=True,
            fmt=".2f",
            xticklabels=target_names,
            yticklabels=target_names,
        )
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.show(block=False)

        scores_mean = np.array(scores).mean()
        scores_std = np.array(scores).std()

    return cmn, cmn_std, scores_mean, scores_std


def main():
    images, targets = loadData(PATH_HIT_IMAGES)
    feature_array, label_array = preprocessData(data=(images, targets))
    prepare_baseline(feature_array, label_array)


if __name__ == "__main__":
    main()
