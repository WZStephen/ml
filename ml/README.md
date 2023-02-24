# Finite Fault Gaussian Prediction Models
## Preparation: 
  * Dataset: 
    * Download location: https://pan.baidu.com/s/1J6PYi9KuMSLzfiAwAt8-5w Download code: 1234 
## Dataset Structure
  * number of data: 7480 
  * Info2: **震级、破裂角度**、破裂长度、宽度
  * res_plots: 400 x 400 pixels of plots
## Datasets
  * <ins>Format:</ins> [# of samples] x [pixels] x [pixels]
  * 7480 x 600x 800 Simulated Finite-Fault Rupture Data(Unshuffled)
  * Training set for label 1: 5984 x 32 x 32 (Cropped)
  * Training set for label 2: 5984 x 128 x 128 (Cropped)
  * Testing set for label 1: 1496 x 32 x 32  (Cropped)
  * Testing set for label 2: 1496 x 128 x 128 (Cropped)
## Methods:
  * ✔ CNN(卷积神经网络) 
  * Autoencoder(自动编码器)
  * AutoCNN(卷积神经网络 + CNN)
  * Deep Reisidual Network
## Results
  * CNN
    * Treat the peak point of predicted sample as a correct prediction if the it located at range of -10 to +10 from corresponding validation sample. 
      * Label1: 1102/1496 (73.66%)
      * Label2：394/1496 (69.59%)
    * Treat the peak point of predicted sample as a correct prediction if the it located at range of -15 to +15 from corresponding validation sample. (Table 4.)
      * Label1: 1496/1496 (100.00%)
      * Label2: 1223/1496 (81.75%)

    
# Deep Residual Network
* Tried with the finite failure model, but got an unexpected results.
* Will be trying more models in the future.


# Quiver
* Add quiver cache IO optimizer, the progress is on the way. However, I have to move to another repository to work with my colleagues.
* WILL BE UPDATED ONCE IT'S FINISHED.

# BERT
* TBD