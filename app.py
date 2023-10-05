
import streamlit as st
from PIL import Image, ImageOps
import tensorflow 
import numpy as np
import pandas as pd
import keras
import cv2

st.set_option('deprecation.showfileUploaderEncoding', False)
st.cache(allow_output_mutation=True)

st.title('Ứng dụng dự đoán dư lượng thuốc trừ sâu')

file = st.file_uploader("Bạn vui lòng nhập ảnh để phân loại ở đây")
if file is None:
        st.text('Đang chờ tải lên....')

else:
        slot = st.empty()
        slot.text('Hệ thống đang thực thi chẩn đoán....')
        
        # pred = preprocessing_uploader(file, model)
        test_image = Image.open(file)
        st.image(test_image, caption="Ảnh đầu vào", width = 400)
        class_names = ['chuaphun', 'phun5ngay']

        # result = class_names[np.argmax(pred)]
        
        # if str(result) == 'chuaphun:
        #     statement = str('Chẩn đoán của mô hình học máy: **Rau chưa được phun thuốc trừ sâu**')
        #     st.success(statement)
        # elif str(result) == 'phun5ngay':
        #     statement = str('Chẩn đoán của mô hình học máy: **Rau đã phun thuốc trừ sâu trong vòng dưới 5 ngày**')
        #     st.error(statement)
        slot.success('Hoàn tất!')
