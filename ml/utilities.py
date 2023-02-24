import os
import shutil

import cv2
import numpy
import numpy as np
import scipy.io
import sklearn
import matplotlib.pyplot as plt
from random import randrange
from scipy.interpolate import interp1d, make_interp_spline, make_lsq_spline, BarycentricInterpolator


def vis_gaussian(path, title):
    np_array = np.load(path)
    figure, ax = plt.subplots(2, 5, figsize=(40, 10))
    figure.tight_layout()
    for i in range(0, 5):
        index = randrange(1496)
        '''第一标签'''
        ax[0][i].title.set_text('Label1_' + title)
        ax[0][i].plot(np_array[i, 0, :], label="Original")
        ax[0][i].legend()
        ax[0][i].plot(np_array[i, 1, :], label="Predicted")
        ax[0][i].legend()

        '''第二标签'''
        ax[1][i].title.set_text('Label2_' + title)
        ax[1][i].plot(np_array[i, 2, :], label="Original")
        ax[1][i].legend()
        ax[1][i].plot(np_array[i, 3, :], label="Predicted")
        ax[1][i].legend()
    plt.show()


def load_npy(path):
    np_array = np.load(path)
    figure, ax = plt.subplots(2, 5, figsize=(40, 10))
    figure.tight_layout()
    for i in range(0, 5):
        index = randrange(1496)
        '''第一标签'''
        ax[0][i].title.set_text('Label1 at #' + str(index))
        # ax[0][i].plot(np_array[index, 0, :], label="Original")
        # ax[0][i].legend()
        ax[0][i].plot(np_array[index, 1, :], label="Predicted")
        ax[0][i].legend()
        ax[0][i].plot(np_array[i, 4, :], label="Predicted_interp")
        ax[0][i].legend()

        '''第二标签'''
        ax[1][i].title.set_text('Label2 at #' + str(index))
        # ax[1][i].plot(np_array[index, 2, :], label="Original")
        # ax[1][i].legend()
        ax[1][i].plot(np_array[index, 3, :], label="Predicted")
        ax[1][i].legend()
        ax[1][i].plot(np_array[i, 5, :], label="Predicted_interp")
        ax[1][i].legend()
    plt.show()


def load_images(folder):
    # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    # img = cv2.imread("data/original/Templates/Mw_1.80_Angle_0_Length_0.01.jpg")  # Read image
    # crop_img = img[50:580, 115:780]
    # resize_img = cv2.resize(crop_img, (400, 400))  # Resize image
    # cv2.imshow("output", resize_img)  # Show image
    # cv2.waitKey(0)
    labels = scipy.io.loadmat('data/original/label_guass.mat')['label']
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        crop_img = img[50:580, 115:780]
        resize_img = cv2.resize(crop_img, (32, 32))  # Resize image
        if img is not None:
            images.append(resize_img)
    mat = numpy.array(images)

    array1_shuffled, array2_shuffled = sklearn.utils.shuffle(mat, labels)
    train_data, test_data = array1_shuffled[:5984, ...], array1_shuffled[5984:, ...]
    train_labels, test_labels = array2_shuffled[:5984, ...], array2_shuffled[5984:, ...]

    np.save('data/cleaned_temp_32/train_data.npy', train_data)
    np.save('data/cleaned_temp_32/test_data.npy', test_data)
    np.save('data/cleaned_temp_32/train_labels.npy', train_labels)
    np.save('data/cleaned_temp_32/test_labels.npy', test_labels)
    print('finished')


def load_images_v2():
    folder = 'data/original/Templates'
    # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    # img = cv2.imread("data/original/Templates/Mw_1.80_Angle_0_Length_0.01.jpg")  # Read image
    # crop_img = img[50:580, 115:780]
    # resize_img = cv2.resize(crop_img, (400, 400))  # Resize image
    # cv2.imshow("output", resize_img)  # Show image
    # cv2.waitKey(0)
    labels = scipy.io.loadmat('data/original/label_guass.mat')['label']
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        crop_img = img[50:580, 115:780]
        resize_img = cv2.resize(crop_img, (224, 224))  # Resize image
        if img is not None:
            images.append(resize_img)
    mat = numpy.array(images)

    array1_shuffled, array2_shuffled = sklearn.utils.shuffle(mat, labels)
    train_data, test_data = array1_shuffled[:5984, ...], array1_shuffled[5984:, ...]
    train_labels, test_labels = array2_shuffled[:5984, ...], array2_shuffled[5984:, ...]

    np.save('data/cleaned_temp_224/train_data.npy', train_data)
    np.save('data/cleaned_temp_224/test_data.npy', test_data)
    np.save('data/cleaned_temp_224/train_labels.npy', train_labels)
    np.save('data/cleaned_temp_224/test_labels.npy', test_labels)
    print('finished')


def smooth_line(npy_path):
    np_array = np.load(npy_path)
    x = np.arange(0, 256, 1)
    figure, ax = plt.subplots(2, 5, figsize=(40, 10))
    figure.tight_layout()
    result_max_label1 = []
    result_max_label2 = []
    for i in range(5):
        index = randrange(1495)
        y1 = np_array[index, 1, :]
        y2 = np_array[index, 3, :]
        new_x = np.linspace(x.min(), x.max(), 50)
        spl1 = make_interp_spline(x, y1, k=1)  # type: BSpline
        spl2 = make_interp_spline(x, y2, k=1)  # type: BSpline
        new_y1 = spl1(new_x)
        new_y2 = spl2(new_x)
        f2_1 = interp1d(new_x, new_y1, kind='cubic')
        f2_2 = interp1d(new_x, new_y2, kind='cubic')
        xnew1 = np.linspace(x.min(), x.max(), 30, endpoint=True)
        ynew1 = f2_1(xnew1)
        xnew2 = np.linspace(x.min(), x.max(), 30, endpoint=True)
        ynew2 = f2_2(xnew2)

        '''第一标签'''
        ax[0][i].title.set_text('Label1 at #' + str(index))
        ax[0][i].plot(np_array[index, 0, :], label="Original")
        ax[0][i].legend()
        # ax[0][i].plot(np_array[index, 1, :], label="Predicted")
        # ax[0][i].legend()
        ax[0][i].plot(xnew1, ynew1, label="Predicted_interp")
        ax[0][i].legend()
        result_max_label1.append(np.max(ynew1))
        '''第二标签'''
        ax[1][i].title.set_text('Label2 at #' + str(index))
        ax[1][i].plot(np_array[index, 2, :], label="Original")
        ax[1][i].legend()
        # ax[1][i].plot(np_array[index, 3, :], label="Predicted")
        # ax[1][i].legend()
        ax[1][i].plot(xnew2, ynew2, label="Predicted_interp")
        ax[1][i].legend()
        result_max_label2.append(np.max(ynew2))
    plt.show()
    combined = np.stack((result_max_label1, result_max_label2), axis=0)
    # np.save('results/r2_interp_npy_max.npy', combined)


def average_interp(npy_path):
    np_array = np.load(npy_path)
    result_list = []
    for i in range(0, len(np_array)):
        y1 = np_array[i, 1, :]
        y2 = np_array[i, 3, :]
        tmp1 = np.copy(y1)
        tmp2 = np.copy(y2)
        for j in range(0, len(y1)):
            division1 = sum(y1[j: j + 8]) / 8.
            division2 = sum(y2[j: j + 20]) / 20.
            tmp1[j] = division1
            tmp2[j] = division2
        new_row = np.concatenate((np_array[i], np.stack((tmp1, tmp2), axis=0)), axis=0)
        result_list.append(new_row)
    fin = np.array(result_list)
    np.save('results/r2_interp', fin)
    '''
    ------r2.npy data structure------
    row0: original data corresponding with label1
    row1: predicted data corresponding with label1
    row2: original data corresponding with label2
    row3: predicted data corresponding with label2
    row4: predicted data after interpolate corresponding with label1
    row5: predicted data after interpolate corresponding with label2
    '''
    # x = np.arange(0, 256, 1)
    # plt.plot(x, fin[0, 3, :])
    # plt.plot(x, fin[0, 5, :])
    # plt.show()
    # print()


def evaluation(npy_path):
    np_array = np.load(npy_path)
    p = np.load('results/r2_interp_npy_max.npy')
    result_label1 = []
    result_label2 = []
    for i in range(0, len(np_array)):
        y = np_array[i, :, :]
        o1 = y[0, :]
        o2 = y[2, :]
        p1_max = p[0, i]
        p2_max = p[1, i]
        o1_max = np.max(o1)
        o2_max = np.max(o2)
        index_of_maximum_o1 = np.where(o1 == o1_max)
        index_of_maximum_o2 = np.where(o2 == o2_max)
        o1_range = o1[int(index_of_maximum_o1[0]) - 16 : int(index_of_maximum_o1[0]) + 15]
        o2_range = o2[int(index_of_maximum_o2[0]) - 16 : int(index_of_maximum_o2[0]) + 15]
        if p1_max >= o1_range[0] and p1_max >= o1_range[30]:
            result_label1.append('1')
            print('label1 validation success')
        else:
            result_label1.append('0')
            # print('label1 validation failed')

        if p2_max >= o2_range[0] and p2_max >= o2_range[30]:
            result_label2.append('1')
            # print('label2 validation success')
        else:
            result_label2.append('0')
            # print('label2 validation failed')
    num1 = result_label1.count('1')
    num2 = result_label2.count('1')
    print(str(num1) + '/' + '1496' + '\n' +
          str(num2) + '/' + '1496')


def dataReader(index, path):
    #######################
    ####### 读取数据集 ######
    #######################
    trainData = np.load('data/' + path + '/train_data.npy')
    testData = np.load('data/' + path + '/test_data.npy')

    trainLabel = np.load('data/' + path + '/train_labels.npy')
    trainLabel_oneRow = trainLabel[:, index, :]
    testLabel = np.load('data/' + path + '/test_labels.npy')
    testLabel_oneRow = testLabel[:, index, :]

    train_data = trainData.astype('float32') / 255.
    test_data = testData.astype('float32') / 255.
    train_label = trainLabel_oneRow.astype('float32') / 255.
    test_label = testLabel_oneRow.astype('float32') / 255.
    return train_data, test_data, train_label, test_label

def tiny_imagenet_val_cleaner():
    f = open("data/tiny-imagenet-200/val/val_annotations.txt", "r")
    val = f.read()

    # val_folder = re.findall('\\bn\w+\\b', val)
    # val_folder = list(dict.fromkeys(val_folder))
    # for item in val_folder:
    #     folder_path = 'data/tiny-imagenet-200/val/' + item
    #     os.mkdir(folder_path)

    cleaned = val.splitlines()
    res = {}
    for item in cleaned:
        line_data = item.split('\t')
        res.update({line_data[0] : line_data[1]})

    for key, value in res.items():
        from_var = 'data/tiny-imagenet-200/val/images/' + key
        to_var = "data/tiny-imagenet-200/val/" + value + "/"
        shutil.move(src=from_var, dst=to_var)
    print('finished')


if __name__ == '__main__':
    tiny_imagenet_val_cleaner()

