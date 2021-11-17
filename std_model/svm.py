import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import os
from joblib import dump, load
from function import augment_train_data, computeZM, read_img
from link_paths import path_save_model


def create_model(images, targets):
    rs=np.random.randint(100)
    images_train4, images_test4, targets_train4, targets_test4  = train_test_split(images, targets, test_size=0.2, stratify=targets,random_state=rs)
    images_train4, targets_train4 = augment_train_data(images_train4, targets_train4, augconf = [0, 4, 12, 0])
    X_train4, y_train4 = computeZM(images_train4, targets_train4)
    X_test4, y_test4 = computeZM(images_test4, targets_test4)

    name,scaler = 'StandardScaler',StandardScaler()

    scaler = scaler.fit(X_train4, y_train4)
    X_train4 = scaler.transform(X_train4)
    X_test4 = scaler.transform(X_test4)

    C = 700.0
    svm1 = svm.SVC(kernel='rbf', gamma=0.08, C=C, probability=True)

    clf = svm1.fit(X_train4, y_train4)
    clf1 = clf
    y_pred = clf1.predict(X_test4)
    print('Dokładność: {:.2f}%'.format(100 * accuracy_score(y_test4, y_pred)))

    C = confusion_matrix(y_test4, y_pred)
    print(C)

    cm = C
    # Normalise
    cmn = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(5, 5))
    # target_names = ['dots', 'lines', 'worms', 'artefacts']
    target_names = ['spots', 'tracks', 'worms', 'artefacts']
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names,
                yticklabels=target_names, annot_kws={"size": 16}, cbar=False, ax=ax)
    ax.set_yticklabels(target_names, fontsize="14", va="center")
    ax.set_xticklabels(target_names, fontsize="14")
    #plt.savefig('cm_svm.png')


    pkl_filename = "svm.pkl"


    filename = os.path.join(path_save_model, pkl_filename)
    filename2 = os.path.join(path_save_model, 'svm_scaler.pkl')
    dump(clf, filename)
    dump(scaler, filename2)

def main():
    images,targets = read_img()
    create_model(images, targets)


if __name__ == '__main__':
    main()