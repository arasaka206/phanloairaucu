
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from io import BytesIO

def load_model():
	model = tf.keras.models.load_model('model2.h5')
	return model
def predict_class(image, model):
# 	image = tf.cast(image, tf.float32)
        image = np.resize(image, (224,224))
# 	image_1 = image
        image = np.dstack((image,image,image))
# 	image_2 = image
# 	cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = np.expand_dims(image, axis = 0)         
        prediction = model.predict(image)
        return prediction

def preprocessing_uploader(file, model):
    bytes_data = file.read()
    inputShape = (224, 224)
    image = Image.open(BytesIO(bytes_data))
    image = image.convert("RGB")
    image = image.resize(inputShape)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image) 
    return prediction
model = load_model()

st.set_option('deprecation.showfileUploaderEncoding', False)
st.cache(allow_output_mutation=True)

st.title('Ứng dụng dự đoán tình trạng thuốc trừ sâu ở rau xanh')

file = st.file_uploader("Bạn vui lòng nhập ảnh để phân loại ở đây")
if file is None:
        st.text('Đang chờ tải lên....')

else:
        slot = st.empty()
        slot.text('Hệ thống đang thực thi chẩn đoán....')
        	
        pred = preprocessing_uploader(file, model)
        test_image = Image.open(file)
        st.image(test_image, caption="Ảnh đầu vào", width = 400)
        class_names = ['chuaphun', 'phun5ngay']

        result = class_names[np.argmax(pred)]
	
        
        if result == "phun5ngay":
            statement = str('Kết quả chẩn đoán: **Rau đã phun thuốc trừ sâu trong vòng dưới 5 ngày**')
            st.error(statement)
	    slot.text(pred)
        if result == "chuaphun":
            statement = str('Kết quả chẩn đoán: **Rau chưa được phun thuốc trừ sâu**')
            st.success(statement)
	    slot.text(pred)
	
	slot.success('Hoàn thành chẩn đoán!')
