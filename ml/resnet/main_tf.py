from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf
import utilities as utils
from tensorflow.keras import losses
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from random import randrange

def Train_DRN_Label():
    train_data, test_data, train_label, test_label = utils.dataReader(0, 'cleaned_temp_128')
    # model = ResNet50(weights='imagenet')
    # img_path = 'data/fragrans.jpg'
    # img = image.load_img(img_path, target_size=(224, 224))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    # input_tensor = Input(shape=(224, 224, 3))
    base_model = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_tensor=None,
                                                       input_shape=(128,128,3), pooling='max')
    x = base_model.output
    x = Dense(1024, activation='LeakyReLU')(x)
    predictions = Dense(256, activation='LeakyReLU')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.summary()

    # 编译模型
    model.compile(optimizer='adam', loss=losses.MeanSquaredError())
    model.fit(train_data, train_label, epochs=11, validation_data=(test_data, test_label))
    model.save('saved_models/Resnet_v1')
    print('finished')


def Test_Model():
    model = tf.keras.models.load_model('saved_models/Resnet_v1')
    model.summary()
    train_data, test_data, train_label, test_label = utils.dataReader(0, 'cleaned_temp_128')
    results = model.predict(test_data)
    combined1 = np.stack((test_label, results), axis=1)

    train_data, test_data, train_label, test_label = utils.dataReader(1, 'cleaned_temp_128')
    results = model.predict(test_data)
    combined2 = np.stack((test_label, results), axis=1)

    fin = np.concatenate((combined1, combined2), axis=1)
    np.save('results/r1_resnet', fin)
    print('prediction finished!')

    # figure, ax = plt.subplots(2, 5, figsize=(40, 10))
    # figure.tight_layout()
    # for i in range(0, 5):
    #     index = randrange(1496)
    #     '''第一标签'''
    #     ax[0][i].title.set_text('Label1')
    #     ax[0][i].plot(combined1[index, 0, :], label="Original")
    #     ax[0][i].legend()
    #     ax[0][i].plot(combined1[index, 1, :], label="Predicted")
    #     ax[0][i].legend()
    # plt.show()
    # print()


