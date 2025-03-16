import streamlit as st

# Setting the page configuration
st.set_page_config(page_title="โปรเจคการทำนาย", page_icon="🏠", layout="wide")

# Header of the homepage
st.title("INTELLIGENCE SYSTEM'S PROJECT")

# Description of Loan Default Prediction (Random Forest and XGBoost models)
st.subheader("วิเคราะห์และพยากรณ์การผิดนัดชำระหนี้ (Loan Default Prediction)")

st.markdown("""
ในส่วนของML Model จะใช้ **Random Forest** และ **XGBoost** เพื่อทำนายการผิดนัดชำระหนี้ โดยการใช้ข้อมูลการเงินของผู้กู้มาทำนายว่าผู้กู้มีแนวโน้มที่จะผิดนัดชำระหนี้หรือไม่
- **Random Forest**: โมเดลที่ใช้การรวมหลายๆ ต้นไม้การตัดสินใจเพื่อเพิ่มความแม่นยำในการทำนาย
- **XGBoost**: โมเดลที่มีประสิทธิภาพสูงในการทำนายที่ใช้การเรียนรู้แบบการเพิ่มประสิทธิภาพ (Boosting)
""")

# Description of Diabetes Prediction (MLP model)
st.subheader("ทำนายการเกิดโรคเบาหวาน")

st.markdown("""
ในส่วนของNeural Networkจะใช้ **MLP (Multilayer Perceptron)** เพื่อทำนายความเสี่ยงในการเกิดโรคเบาหวาน โดยการใช้ข้อมูลสุขภาพของผู้ป่วยในการทำนาย
- **MLP**: โมเดลที่ใช้การเรียนรู้แบบ Neural Network ที่สามารถเรียนรู้ความสัมพันธ์ที่ซับซ้อนได้จากข้อมูลที่มีมิติหลายมิติ
""")

# Personal information section
st.header("👨‍💻ข้อมูลผู้จัดทำ")
st.markdown("""
ชื่อ: นางสาวนลินรัตน์ รุ่งทรัพย์สิน   รหัสนักศึกษา: 6604062630277  เซค:  3
""")
