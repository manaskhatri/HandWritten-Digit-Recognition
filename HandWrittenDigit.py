import pandas as pd
import numpy as np
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout,BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
import matplotlib.pyplot as plt
dt= pd.read_csv('train.csv')
dt1=pd.read_csv('test.csv')

X_prediction=dt1.iloc[:,:].values
X_prediction=np.array(X_prediction,'float32')
X_prediction = X_prediction.reshape(X_prediction.shape[0],28,28,1)


X_train = dt.iloc[:,1:].values
Y_train=dt.iloc[:,0].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X_train,Y_train,test_size=0.25,random_state=0)

X_train=np.array(X_train,'float32')
Y_train=np.array(Y_train,'float32')
X_test=np.array(X_test,'float32')
Y_test=np.array(Y_test,'float32')

Y_train= np_utils.to_categorical(Y_train,10)
Y_test= np_utils.to_categorical(Y_test,10)

X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)

model=Sequential()

model.add(Convolution2D(128,3,3,activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Convolution2D(64,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='sigmoid'))
model.summary()
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train,Y_train,batch_size=128,epochs=10,verbose=2,validation_data=(X_test,Y_test),shuffle=True)

fer1_json = model.to_json()
with open("fer1.json", "w") as json_file:
    json_file.write(fer1_json)
model.save_weights("fer1.h5")

score = model.evaluate(X_test, Y_test, verbose=1)
print('\n''Test accuracy:', score[1])
n=10
mask = range(10,20)
X_valid = X_test[mask]
y_pred = model.predict_classes(X_valid)
print(y_pred)
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_valid[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
plt.close()

X_valid=X_prediction
y_pred = model.predict_classes(X_valid,batch_size=32)
print(y_pred)

x=np.array(y_pred)
y=[]
for i in range(1,28001):
    y.append(i)

dict = {'ImageId': x,'Label':y}
dp = pd.DataFrame(dict)
dp.to_csv('xyz.csv', header=False, index=False)