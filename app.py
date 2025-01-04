import streamlit as st
from PIL import Image
import numpy as np
import joblib
import gzip

# โหลดโมเดลจากไฟล์ที่บีบอัด
with gzip.open('cnn_model.pkl.gz', 'rb') as f:
    model = joblib.load(f)

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
