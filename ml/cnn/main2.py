import numpy as np
import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import layers, models
import utilities as utils


def Train_CNN_Label1():
    train_data, test_data, train_label, test_label =  utils.dataReader(0, 'cleaned_temp_32')

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256))
    # model.add(layers.Reshape((1, 256)))
    model.summary()

    # 编译模型
    model.compile(optimizer='adam', loss=losses.MeanSquaredError())
    model.fit(train_data, train_label, epochs=10, validation_data=(test_data, test_label))
    model.save('saved_models/CNN_temp_v1_L1')
    print('finished')


def Train_CNN_Label2():
    train_data, test_data, train_label, test_label =  utils.dataReader(1, 'cleaned_temp_128')

    model = models.Sequential()
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation='relu'))
    # model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(256))
    # model.add(layers.Reshape((1, 256)))
    model.summary()

    # 编译模型
    model.compile(optimizer='adam', loss=losses.MeanSquaredError())
    model.fit(train_data, train_label, epochs=10, validation_data=(test_data, test_label))
    model.save('saved_models/CNN_temp_v1_L2')
    print('finished')


def Test_Model():
    # test_data = testData_1stRow[..., tf.newaxis]
    # test_label = testLabel_1stRow[..., tf.newaxis]

    # 加载之前保存的模型
    print('start to load the model:')

    model = tf.keras.models.load_model('saved_models/CNN_temp_v1_L1')
    model.summary()
    train_data, test_data, train_label, test_label = utils.dataReader(0, 'cleaned_temp_32')
    results = model.predict(test_data)
    combined1 = np.stack((test_label, results), axis=1)

    model = tf.keras.models.load_model('saved_models/CNN_temp_v1_L2')
    model.summary()
    train_data, test_data, train_label, test_label = utils.dataReader(1, 'cleaned_temp_128')
    results = model.predict(test_data)
    combined2 = np.stack((test_label, results), axis=1)
    fin = np.concatenate((combined1, combined2), axis=1)
    np.save('results/r2', fin)
    '''
    ------r2.npy data structure------
    row1: original data corresponding with label1
    row2: predicted data corresponding with label1
    row3: original data corresponding with label2
    row4: predicted data corresponding with label2
    '''
    utils.vis_gaussian(fin, 'CNN')
    print('prediction finished!')


