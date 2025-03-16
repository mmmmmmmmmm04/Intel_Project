import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib  # ใช้โหลดตัว scaler ที่เทรนไว้

# โหลดโมเดลที่เทรนไว้
@st.cache_resource
def load_trained_model():
    try:
        model = load_model('models/mlp_model.h5')  # ตรวจสอบว่าไฟล์โมเดลอยู่ใน path นี้จริง
        scaler = joblib.load('models/scaler.pkl')  # โหลดตัวปรับสเกลที่ใช้ตอนเทรนโมเดล
        return model, scaler
    except Exception as e:
        st.error(f"ไม่สามารถโหลดโมเดลหรือ scaler ได้: {e}")
        return None, None

model, scaler = load_trained_model()

# หากโหลดโมเดลไม่สำเร็จ ให้หยุดการทำงาน
if model is None or scaler is None:
    st.stop()

# ฟังก์ชันทำนายผลเบาหวาน
def predict_diabetes(gender, age, hypertension, heart_disease, bmi, hbA1c, blood_glucose):
    input_data = np.array([[gender, age, hypertension, heart_disease, bmi, hbA1c, blood_glucose]])
    input_data_scaled = scaler.transform(input_data)  # ใช้ scaler ที่โหลดมา
    prediction = model.predict(input_data_scaled)[0, 0]  # ดึงค่าเดียวจาก array
    
    return "ท่านมีโอกาสเป็นโรคเบาหวาน" if prediction >= 0.5 else "ท่านไม่มีโอกาสเป็นโรคเบาหวาน"

# ส่วนของ UI
st.title("ทำนายการเกิดโรคเบาหวาน")
st.write("กรอกข้อมูลต่อไปนี้เพื่อทำนายว่าอาจเป็นโรคเบาหวานหรือไม่:")

# รับอินพุตจากผู้ใช้
gender = st.selectbox("เพศ", ["ชาย", "หญิง"])
age = st.number_input("อายุ", min_value=1, max_value=120, value=30, step=1)
hypertension = st.selectbox("มีความดันโลหิตสูงหรือไม่", ["ใช่", "ไม่ใช่"])
heart_disease = st.selectbox("มีโรคหัวใจหรือไม่", ["ใช่", "ไม่ใช่"])
bmi = st.number_input("ดัชนีมวลกาย (BMI)", min_value=10.0, max_value=50.0, value=24.99, step=0.1)
hbA1c = st.number_input("ระดับ HbA1c (%)", min_value=4.0, max_value=15.0, value=6.0, step=0.1)
blood_glucose = st.number_input("ระดับน้ำตาลในเลือด (mg/dL)", min_value=50, max_value=500, value=120, step=1)

# แปลงค่าที่เป็นข้อความให้เป็นตัวเลข
gender = 1 if gender == "หญิง" else 0
hypertension = 1 if hypertension == "ใช่" else 0
heart_disease = 1 if heart_disease == "ใช่" else 0

# ปุ่มทำนาย
if st.button("ทำนาย"):
    result = predict_diabetes(gender, age, hypertension, heart_disease, bmi, hbA1c, blood_glucose)
    st.success(result)