import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="แอปทำนายการผิดนัดชำระเงินกู้", layout="wide")

# โหลดโมเดลที่เทรนไว้แล้ว
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load('models/rf_model.pkl')
        xgb_model = joblib.load('models/xgb_model.pkl')
        return rf_model, xgb_model
    except Exception as e:
        st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")
        return None, None

rf_model, xgb_model = load_models()

# ฟังก์ชันสำหรับการทำนาย
def predict(input_data, model_choice):
    # แปลงข้อมูลเป็น DataFrame
    input_df = pd.DataFrame([input_data])
    
    # ดึงรายชื่อคอลัมน์จากโมเดล
    if model_choice == "Random Forest":
        if hasattr(rf_model, 'feature_names_in_'):
            feature_names = rf_model.feature_names_in_
        else:
            # ถ้าไม่มี feature_names_in_ ให้ดูจากโมเดล XGBoost แทน
            if hasattr(xgb_model, 'feature_names_in_'):
                feature_names = xgb_model.feature_names_in_
            else:
                # ถ้าไม่มีทั้งคู่ ให้แสดงข้อผิดพลาด
                st.error("ไม่สามารถดึง feature names จากโมเดลได้")
                return None, None
    else:  # XGBoost
        if hasattr(xgb_model, 'feature_names_in_'):
            feature_names = xgb_model.feature_names_in_
        else:
            if hasattr(rf_model, 'feature_names_in_'):
                feature_names = rf_model.feature_names_in_
            else:
                st.error("ไม่สามารถดึง feature names จากโมเดลได้")
                return None, None
    
    # --- สร้างข้อมูลให้ตรงกับที่โมเดลถูกเทรนมา ---
    
    # 1. เริ่มด้วยการสร้าง DataFrame ที่มีคอลัมน์ครบตามโมเดล โดยใส่ค่า 0 ทั้งหมด
    processed_data = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # 2. ใส่ค่าตัวเลขลงไปโดยตรง
    numeric_cols = ['age', 'income', 'loanamount', 'loanterm', 'creditscore', 'dtiratio']
    for col in numeric_cols:
        if col in input_data and col in processed_data.columns:
            processed_data[col] = input_data[col]
    
    # 3. จัดการกับข้อมูล categorical โดยใช้ชื่อคอลัมน์ตรงตามที่โมเดลเรียนรู้
    
    # Education
    if 'education' in input_data:
        education = input_data['education']
        if education == 'highschool' and 'education_High School' in processed_data.columns:
            processed_data['education_High School'] = 1
        elif education == 'graduate' and 'education_Bachelor\'s' in processed_data.columns:
            processed_data['education_Bachelor\'s'] = 1
        elif education == 'postgraduate':
            if 'education_Master\'s' in processed_data.columns:
                processed_data['education_Master\'s'] = 1
            elif 'education_PhD' in processed_data.columns:
                processed_data['education_PhD'] = 1
    
    # Employment Status
    if 'employmentstatus' in input_data:
        emp_status = input_data['employmentstatus']
        if emp_status == 'fulltime' and 'employmenttype_Full-time' in processed_data.columns:
            processed_data['employmenttype_Full-time'] = 1
        elif emp_status == 'parttime' and 'employmenttype_Part-time' in processed_data.columns:
            processed_data['employmenttype_Part-time'] = 1
        elif emp_status == 'selfemployed' and 'employmenttype_Self-employed' in processed_data.columns:
            processed_data['employmenttype_Self-employed'] = 1
        elif emp_status == 'unemployed' and 'employmenttype_Unemployed' in processed_data.columns:
            processed_data['employmenttype_Unemployed'] = 1
    
    # Marital Status
    if 'maritalstatus' in input_data:
        marital = input_data['maritalstatus']
        if marital == 'single' and 'maritalstatus_Single' in processed_data.columns:
            processed_data['maritalstatus_Single'] = 1
        elif marital == 'married' and 'maritalstatus_Married' in processed_data.columns:
            processed_data['maritalstatus_Married'] = 1
        elif marital == 'divorced' and 'maritalstatus_Divorced' in processed_data.columns:
            processed_data['maritalstatus_Divorced'] = 1
    
    # ทำนายโดยใช้โมเดลที่เลือก
    try:
        if model_choice == "Random Forest":
            prediction = rf_model.predict(processed_data)
            prob = rf_model.predict_proba(processed_data)[0][1]
        else:  # XGBoost
            prediction = xgb_model.predict(processed_data)
            prob = xgb_model.predict_proba(processed_data)[0][1]
        
        return prediction[0], prob
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")
        return None, None

# UI
st.title("ระบบทำนายการผิดนัดชำระเงินกู้")
st.write("กรอกข้อมูลผู้กู้เพื่อประเมินความเสี่ยงในการผิดนัดชำระหนี้")

# แบ่งหน้าจอเป็น 2 คอลัมน์
col1, col2 = st.columns(2)

with col1:
    st.subheader("ข้อมูลส่วนตัว")
    age = st.slider("อายุ", 18, 80, 35)
    education = st.selectbox("ระดับการศึกษา", ["highschool", "graduate", "postgraduate"])
    employment_status = st.selectbox("สถานะการทำงาน", 
                                     ["fulltime", "parttime", "selfemployed", "unemployed"])
    marital_status = st.selectbox("สถานภาพ", ["single", "married", "divorced"])

with col2:
    st.subheader("ข้อมูลการเงิน")
    income = st.number_input("รายได้ต่อปี", min_value=0, value=50000)
    loan_amount = st.number_input("จำนวนเงินกู้", min_value=1000, value=10000)
    loan_term = st.slider("ระยะเวลากู้ (เดือน)", 6, 60, 36)
    credit_score = st.slider("คะแนนเครดิต", 300, 850, 650)
    
    # เพิ่ม DTI Ratio (Debt-to-Income)
    dti_ratio = st.slider("อัตราส่วนหนี้ต่อรายได้ (DTI)", 0.0, 1.0, 0.36, step=0.01)

# เลือกโมเดลที่ต้องการใช้
model_choice = st.radio("เลือกโมเดลที่ต้องการใช้ทำนาย", ["Random Forest", "XGBoost"])

# รวบรวมข้อมูลทั้งหมด
input_data = {
    'age': age,
    'education': education,
    'employmentstatus': employment_status,
    'maritalstatus': marital_status,
    'income': income,
    'loanamount': loan_amount,
    'loanterm': loan_term,
    'creditscore': credit_score,
    'dtiratio': dti_ratio
}

# เพิ่มปุ่มแสดงรายละเอียดโมเดล


# ปุ่มทำนาย
if st.button("ทำนายผล"):
    if rf_model is not None and xgb_model is not None:
        prediction, probability = predict(input_data, model_choice)
        
        if prediction is not None and probability is not None:
            # แสดงผลการทำนาย
            st.header("ผลการทำนาย")
            
            # สร้าง UI แสดงผลที่น่าสนใจ
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("มีความเสี่ยงที่จะผิดนัดชำระหนี้")
                else:
                    st.success("มีแนวโน้มที่จะชำระหนี้ได้ตามกำหนด")
            
            with col2:
                st.metric(label="โอกาสที่จะผิดนัดชำระหนี้", 
                         value=f"{probability*100:.2f}%")
            
            # แสดงข้อมูลเพิ่มเติม
            st.subheader("รายละเอียดการประเมิน")
            risk_level = "สูง" if probability > 0.7 else "ปานกลาง" if probability > 0.3 else "ต่ำ"
            
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.info(f"ระดับความเสี่ยง: {risk_level}")
                st.info(f"โมเดลที่ใช้: {model_choice}")
                
            with info_col2:
                # คำแนะนำสำหรับการพิจารณาอนุมัติ
                if probability > 0.7:
                    st.warning("คำแนะนำ: อาจต้องพิจารณาเงื่อนไขเพิ่มเติมก่อนอนุมัติ")
                elif probability > 0.3:
                    st.info("คำแนะนำ: ควรตรวจสอบประวัติเพิ่มเติม")
                else:
                    st.success("คำแนะนำ: สามารถพิจารณาอนุมัติได้")
    else:
        st.error("ไม่สามารถทำนายได้เนื่องจากไม่พบโมเดล")

# คำอธิบายเพิ่มเติม
with st.expander("คำอธิบายเกี่ยวกับโมเดล"):
    st.write("""
    - **Random Forest**: เป็นอัลกอริทึมแบบ ensemble learning ที่สร้างต้นไม้ตัดสินใจหลายต้น และรวมผลลัพธ์
    - **XGBoost**: เป็นอัลกอริทึมแบบ gradient boosting ที่มีประสิทธิภาพสูง เหมาะกับข้อมูลที่ซับซ้อน
    
    ข้อมูลที่ใช้ในการทำนาย ได้แก่ อายุ การศึกษา สถานะการทำงาน สถานภาพ รายได้ จำนวนเงินกู้ ระยะเวลากู้ คะแนนเครดิต และอัตราส่วนหนี้ต่อรายได้
    """)