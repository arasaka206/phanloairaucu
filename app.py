
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
	model = tf.keras.models.load_model('dautien.h5')
	model1 = tf.keras.models.load_model('xalach.h5')
	model2 = tf.keras.models.load_model('raumuong.h5')
	model3 = tf.keras.models.load_model('caibe.h5')
	model4 = tf.keras.models.load_model('Bapcai.h5')
	model5 = tf.keras.models.load_model('mongtoi.h5')
	return model, model1, model2, model3, model4, model5
def tach_kenh_mau(buc_hinh, kenh):
    #image = Image.open(buc_hinh)
    image = buc_hinh
    channels = list(image.split())

    if kenh == 'R':
        channel_image = Image.merge('RGB', (channels[0], Image.new('L', image.size, 0), Image.new('L', image.size, 0)))
    elif kenh == 'G':
        channel_image = Image.merge('RGB', (Image.new('L', image.size, 0), channels[1], Image.new('L', image.size, 0)))
    elif kenh == 'B':
        channel_image = Image.merge('RGB', (Image.new('L', image.size, 0), Image.new('L', image.size, 0), channels[2]))
    else:
        raise ValueError("Invalid channel. Choose 'R', 'G', or 'B'.")

    return channel_image
def preprocessing_uploader(file, model):
    inputShape = (224, 224)
    bytes_data = Image.open(file)
    image = tach_kenh_mau(bytes_data,'R')
    image = image.convert("RGB")
    image = image.resize(inputShape)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image) 
    return prediction
model, model1, model2, model3, model4, model5 = load_model()

st.set_option('deprecation.showfileUploaderEncoding', False)
st.cache(allow_output_mutation=True)

st.title('Ứng dụng dự đoán tình trạng thuốc trừ sâu ở rau xanh')

file = st.file_uploader("Bạn vui lòng nhập ảnh để phân loại ở đây")
if file is None:
        st.text('Đang chờ tải lên....')

else:
	slot = st.empty()
	slot.text('Hệ thống đang thực thi chẩn đoán....')
        	
	
	test_image = Image.open(file)
	pred = preprocessing_uploader(file, model)
	st.image(test_image, caption="Ảnh đầu vào", width = 400)
	class_names = ['xalach', 'raumuong','caibe', 'bapcai','mongtoi']

	result = class_names[np.argmax(pred)]
	st.text(pred)
#	st.text(result)
        
	if result == "xalach":
		pred = preprocessing_uploader(file, model1)
		class_names = ['daphun','chuaphun']
		if result == "daphun":
		    statement = str('Kết quả chẩn đoán: **Rau đã phun thuốc trừ sâu trong vòng dưới 5 ngày**')
		    st.success(statement)
		else:
		    statement = str('Kết quả chẩn đoán: **Rau chưa được phun thuốc trừ sâu**')
		    st.error(statement)
	if result == "raumuong":
		pred = preprocessing_uploader(file, model2)
		class_names = ['daphun','chuaphun']
		if result == "daphun":
		    statement = str('Kết quả chẩn đoán: **Rau đã phun thuốc trừ sâu trong vòng dưới 5 ngày**')
		    st.success(statement)
		else:
		    statement = str('Kết quả chẩn đoán: **Rau chưa được phun thuốc trừ sâu**')
		    st.error(statement)
	if result == "caibe":
		pred = preprocessing_uploader(file, model3)
		class_names = ['daphun','chuaphun']
		if result == "daphun":
		    statement = str('Kết quả chẩn đoán: **Rau đã phun thuốc trừ sâu trong vòng dưới 5 ngày**')
		    st.success(statement)
		else:
		    statement = str('Kết quả chẩn đoán: **Rau chưa được phun thuốc trừ sâu**')
		    st.error(statement)
	if result == "bapcai":
		pred = preprocessing_uploader(file, model4)
		class_names = ['daphun','chuaphun']
		if result == "daphun":
		    statement = str('Kết quả chẩn đoán: **Rau đã phun thuốc trừ sâu trong vòng dưới 5 ngày**')
		    st.success(statement)
		else:
		    statement = str('Kết quả chẩn đoán: **Rau chưa được phun thuốc trừ sâu**')
		    st.error(statement)
	if result == "mongtoi":
		pred = preprocessing_uploader(file, model5)
		class_names = ['daphun','chuaphun']
		if result == "daphun":
		    statement = str('Kết quả chẩn đoán: **Rau đã phun thuốc trừ sâu trong vòng dưới 5 ngày**')
		    st.success(statement)
		else:
		    statement = str('Kết quả chẩn đoán: **Rau chưa được phun thuốc trừ sâu**')
		    st.error(statement)
	#slot.success('Hoàn thành chẩn đoán!')
