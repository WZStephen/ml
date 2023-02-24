# import csv
# import os
# import mat73
# import numpy
# import scipy.io
# import matplotlib.pyplot as plt
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import losses
# from tensorflow.keras.models import Model
# from tensorflow.keras import layers, models
# from sklearn.decomposition import PCA
# import pandas as pd
# from utilities import vis_gaussian
#
# #######################
# ####### 读取数据集 ######
# #######################
# # trainData_plots = mat73.loadmat('data/cleaned_res_plots/shuffled_res_plots_80.mat')['shuffled_res_plots_80']
# # testData_plots = scipy.io.loadmat('data/cleaned_res_plots/shuffled_res_plots_20.mat')['shuffled_res_plots_20']
# # trainData_pca = np.load('data/cleaned_res_plots/pca_res_plots_80.npy')
# # testData_pca = np.load('data/cleaned_res_plots/pca_res_plots_20.npy')
#
#
# # trainLabel = scipy.io.loadmat('data/cleaned_res_plots/shuffled_gauss_labels_80.mat')[ 'shuffled_gauss_labels_80']
# # testLabel = scipy.io.loadmat('data/cleaned_res_plots/shuffled_gauss_labels_20.mat')['shuffled_gauss_labels_20']
# # trainLabel_1stRow = scipy.io.loadmat('data/cleaned_res_plots/shuffled_gauss_labels_80_1stRow.mat')['shuffled_gauss_labels_80_1stRow']
# # testLabel_1stRow = scipy.io.loadmat('data/cleaned_res_plots/shuffled_gauss_labels_20_1stRow.mat')['shuffled_gauss_labels_20_1stRow']
#
# trainData = np.load('data/cleaned_temp_32/train_data.npy')
# testData = np.load('data/cleaned_temp_32/test_data.npy')
# trainLabel = np.load('data/cleaned_temp_32/train_labels.npy')
# testLabel = np.load('data/cleaned_temp_32/test_labels.npy')
#
# def Train_CNN():
#     train_data = trainData_plots[..., tf.newaxis]
#     test_data = testData_plots[..., tf.newaxis]
#     train_label = trainLabel_1stRow[..., tf.newaxis]
#     test_label = testLabel_1stRow[..., tf.newaxis]
#
#     train_data = train_data.astype('float32') / 255.
#     test_data = test_data.astype('float32') / 255.
#     train_label = train_label.astype('float32') / 255.
#     test_label = test_label.astype('float32') / 255.
#
#     model = models.Sequential()
#     model.add(layers.Conv2D(32, (7, 7), activation=LeakyReLU(alpha=0.1), strides=2, padding="same",input_shape=(400, 400, 1)))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(16, (5, 5), activation=LeakyReLU(alpha=0.1), strides=2, padding="same"))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(1, (3, 3), activation=LeakyReLU(alpha=0.1), strides=2, padding="same"))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(256, activation='sigmoid'))
#     model.add(layers.Reshape((1, 256)))
#     model.summary()
#
#     # 编译模型
#     model.compile(optimizer='adam', loss=losses.MeanSquaredError())
#     model.fit(train_data, train_label, epochs=10, validation_data=(test_data, test_label))
#     model.save('saved_models/CNN_v2')
#
#
# def Train_AutoCNN():
#     train_data = trainData_pca[..., tf.newaxis]
#     test_data = testData_pca[..., tf.newaxis]
#     train_label = trainLabel[..., tf.newaxis]
#     test_label = testLabel[..., tf.newaxis]
#
#     train_data = train_data.astype('float32') / 255.
#     test_data = test_data.astype('float32') / 255.
#     train_label = train_label.astype('float32') / 255.
#     test_label = test_label.astype('float32') / 255.
#
#     autoencoder = Autoencoder(128)
#     autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
#     autoencoder.fit(train_data, train_label, epochs=10, shuffle=True, validation_data=(test_data, test_label))
#     autoencoder.summary()
#
#     encoded_imgs = autoencoder.encoder(test_data).numpy()
#     decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
#
#     # n = 10
#     # plt.figure(figsize=(20, 4))
#     # for i in range(n):
#     #     # display original + noise
#     #     ax = plt.subplot(2, 256, i + 1)
#     #     plt.title("original")
#     #     plt.imshow(tf.squeeze(test_data[i]))
#     #     ax.get_xaxis()
#     #     ax.get_yaxis()
#     #
#     #     # display reconstruction
#     #     bx = plt.subplot(2, n, i + n + 1)
#     #     plt.title("Predicted")
#     #     plt.imshow(tf.squeeze(decoded_imgs[i]))
#     # plt.show()
#     print()
#
#
# def Test_Model():
#     test_data = testData_plots[..., tf.newaxis]
#     test_label = testLabel_1stRow[..., tf.newaxis]
#
#     # 加载之前保存的模型
#     model = tf.keras.models.load_model('saved_models/CNN_v2')
#     model.summary()
#     res = model.predict(test_data)
#     print()
#
#     # 测试
#     encoded_imgs = model.encoder(testData_pca)  # .numpy()
#     decoded_imgs = model.decoder(encoded_imgs)  # .numpy()
#
#     # np.concatenate((decoded_imgs[:][0], info3[:]), axis=1)
#
#     results = []
#     for i in range(0, len(decoded_imgs) - 1):
#         decoded_img = decoded_imgs[i]
#         label = testLabel[i]
#         res = np.concatenate((decoded_img, label))
#         results.append(res)
#
#     utilities.results_writer('results/test_results.csv', results)
#     t1, t2, v1, v2 = [], [], [], []
#     for i in range(0, len(results) - 1):
#         test = results[i].astype(float)
#         t1 = t1.append(results[i][0])
#         t2 = t1.append(results[i][1])
#         v1 = v1.append(results[i][2])
#         v2 = v2.append(results[i][3])
#     tmp = []
#     for row in results:
#         tmp.append(row[0])
#     plt.plot(results)
#     plt.show()
#     print()
#
#
# def pca():
#     arr = trainData_pca.reshape(5984, 160000).transpose()
#     pca_160000 = PCA(n_components=512)
#     pca_results = pca_160000.fit(arr)
#     components = pca_results.components_
#     swapped = components.transpose()
#     res = swapped.reshape((5984, 2, 256))
#     numpy.save("data/cleaned_res_plots/pca_res_plots_80", res)
#
#
# class Autoencoder(Model):
#     def __init__(self, latent_dim):
#         super(Autoencoder, self).__init__()
#         self.latent_dim = latent_dim
#         self.encoder = tf.keras.Sequential([
#             layers.Input(shape=(2, 256, 1)),
#             layers.Conv2D(16, (1, 3), activation='relu', padding='same', strides=2),
#             layers.Conv2D(8, (1, 3), activation='relu', padding='same', strides=2),
#         ])
#         self.decoder = tf.keras.Sequential([
#             layers.Conv2DTranspose(8, 3, strides=2, activation='relu', padding='same'),
#             layers.Conv2DTranspose(16, 3, strides=2, activation='relu', padding='same'),
#             layers.Conv2D(1, (3, 1), activation='sigmoid'),
#
#         ])
#
#     def call(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded
#
#
# if __name__ == "__main__":
#     Train_CNN()
#     vis_gaussian(testData)
#     # Train_AutoCNN()
#     # Test_Model()
