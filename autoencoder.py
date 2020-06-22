import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Flatten, Dense,Input
#from keras.datasets import mnist

x = pd.read_csv('/kaggle/input/mnist-in-csv/mnist_train.csv')
y=pd.read_csv('/kaggle/input/mnist-in-csv/mnist_test.csv')

(x_train,y_train) = (x.iloc[:,1:].values,x.iloc[:,0].values)
(x_test,y_test) = (y.iloc[:,1:].values,y.iloc[:,0].values)
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

x_train=np.array(x_train,'float32')
y_train=np.array(y_train,'float32')
x_test=np.array(x_test,'float32')
y_test=np.array(y_test,'float32')

x_train = x_train.reshape(60000, 28,28,1) / 255
x_test = x_test.reshape(10000, 28,28,1) / 255

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

x_train_noisy = x_train_noisy.reshape(x_train_noisy.shape[0],28,28,1)
x_test_noisy = x_test_noisy.reshape(x_test_noisy.shape[0],28,28,1)
plt.imshow(x_test_noisy[1].reshape(28, 28))

input_img = Input(shape=(28,28,1))
enc_1 = Conv2D(32,(3,3),activation='relu', padding='same')(input_img)
mxpool = MaxPooling2D((2,2), padding='same')(enc_1)
enc_2 = Conv2D(32,(3,3), activation='relu', padding='same')(mxpool)
encoder = MaxPooling2D((2,2), padding='same')(enc_2)

dec_1 = Conv2D(32,(3,3),activation='relu', padding='same')(encoder)
upsamp1 = UpSampling2D((2,2))(dec_1)
dec_2 = Conv2D(32,(3,3),activation='relu', padding='same')(upsamp1)
upsamp2 = UpSampling2D((2,2))(dec_2)
decoder = Conv2D(1,(3,3), activation='sigmoid', padding='same')(upsamp2)

autoencoder = Model(input_img,decoder)

autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy')
autoencoder.summary()
autoencoder.fit(x_train_noisy,x_train,epochs=20,verbose=1,batch_size=256,validation_data=(x_test_noisy,x_test))

score = autoencoder.evaluate(x_test_noisy,x_test,verbose=1)
print(score)

x_pred = x_test_noisy[10].reshape(1,28,28,1)
plt.imshow(x_pred.reshape(28,28))

x_out = autoencoder.predict(x_pred)
plt.imshow(x_out.reshape(28,28))

# from PIL import Image
# img = Image.fromarray(x_out.reshape(28,28))
# img.show()