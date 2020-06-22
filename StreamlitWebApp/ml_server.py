from flask import Flask, request
import json
import tensorflow as tf
import random
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model('model.h5')

_, (x_test,_) = tf.keras.datasets.mnist.load_data()

x_test = x_test.reshape(x_test.shape[0],28,28,1)
x_test=np.array(x_test,'float32')
x_test = x_test/255

noise_factor = 0.5
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
x_test_noisy = x_test_noisy.reshape(x_test_noisy.shape[0],28,28,1)

def get_prediction():
    index = np.random.choice(x_test_noisy.shape[0])
    image = x_test_noisy[index,:,:]
    image_arr = np.reshape(image, (1,28,28,1))
    return model.predict(image_arr),image


@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='POST':
        pred,image = get_prediction()
        final_pred = [p.tolist() for p in pred]
        print(final_pred)
        return json.dumps({
            'prediction':final_pred,
            'image':image.tolist()
            })
    return 'welcome to the model server'

if __name__=='__main__':
    app.run()
