import gdown
import tensorflow as tf  # ใช้ TensorFlow สำหรับการโหลดโมเดล
import streamlit as st
from PIL import Image
import numpy as np

# ดาวน์โหลดไฟล์โมเดลจาก Google Drive
url = 'https://drive.google.com/uc?export=download&id=1abSk9bY-f5eiVqqCpIcfrNmWA2tfpe2S'
output = 'cnn_model.h5'  # เปลี่ยนเป็น .h5 หากคุณบันทึกเป็นไฟล์ Keras

gdown.download(url, output, quiet=False)

# โหลดโมเดลด้วย TensorFlow/Keras
try:
    model = tf.keras.models.load_model(output)  # ใช้ TensorFlow load model
    st.write("Model loaded successfully.")
except Exception as e:
    st.write(f"Error loading model: {e}")

# กำหนดขนาดของภาพที่โมเดลคาดหวัง
IMG_SIZE = (150, 150)

# ฟังก์ชันสำหรับพยากรณ์ผลลัพธ์
def predict(image):
    image = image.resize(IMG_SIZE)  # Resize ภาพ
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)  # Normalize และเพิ่มมิติ
    prediction = model.predict(image_array)
    return "Accident" if prediction[0][0] > 0.5 else "Non_Accident"

# สร้างอินเทอร์เฟซ Streamlit
st.title("Road Accident Detection System")
st.write("อัปโหลดภาพเพื่อให้ระบบพยากรณ์")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("กำลังพยากรณ์...")

    result = predict(image)
    st.write(f"ผลลัพธ์: {result}")
