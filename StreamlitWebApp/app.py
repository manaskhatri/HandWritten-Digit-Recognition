
import streamlit as st
import json
import requests
import matplotlib.pyplot as plt
import numpy as np

URI = 'http://127.0.0.1:5000'
st.title('MNIST Image Denoising')
st.markdown('## Input image')

if st.button('Get random prediction'):
    response = requests.post(URI,data={})
    response = json.loads(response.text)
    preds = response.get('prediction')
    image = response.get('image')
    image = np.reshape(image,(28,28))
    st.sidebar.image(image,width=150)
    st.image(np.array(preds),width=150)
    
