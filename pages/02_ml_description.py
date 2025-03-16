import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="วิเคราะห์และพยากรณ์การผิดนัดชำระหนี้",
    page_icon="💰",
    layout="wide"
)

# ส่วนหัว
st.title("⭐️ วิเคราะห์และพยากรณ์การผิดนัดชำระหนี้ (Loan Default Prediction)")
st.markdown("การวิเคราะห์ข้อมูลสินเชื่อและใช้ Machine Learning มาช่วยคาดการณ์ว่าผู้ขอสินเชื่อจะ \"ผิดนัดชำระหนี้\"หรือไม่")

tab0, tab1, tab2, tab3, tab4 = st.tabs(["📚 ทฤษฎี", "📊 ข้อมูล", "🔄 เตรียมข้อมูล", "🤖 โมเดล", "📈 ผลลัพธ์"])

with tab0:
    st.header("📚 ทฤษฎีของโมเดล")
    
    # อธิบายทฤษฎีของ Random Forest
    st.subheader("🌲 Random Forest")
    st.markdown("""
    **Random Forest** เป็นวิธีการเรียนรู้ที่ใช้การรวม (Ensemble Learning) ซึ่งประกอบด้วยหลายๆ ต้นไม้การตัดสินใจ (Decision Trees) ที่ทำงานร่วมกัน
    โดยที่แต่ละต้นไม้จะถูกสร้างจากการสุ่มข้อมูลในแต่ละรอบ (Bootstrapping) และในแต่ละการตัดสินใจจะสุ่มเลือกคุณลักษณะ (Feature) ที่จะทำการแยกข้อมูล

    การใช้หลายๆ ต้นไม้จะช่วยลดการ overfitting และทำให้โมเดลมีความยืดหยุ่นสูงในการทำงานกับข้อมูลที่มีความซับซ้อน

    **ข้อดีของ Random Forest**
    - ทนทานต่อการ overfitting
    - สามารถทำงานได้ดีในข้อมูลที่มีลักษณะซับซ้อน
    - ทำงานได้ทั้งสำหรับปัญหาการจำแนกประเภท (Classification) และการพยากรณ์ค่า (Regression)
    - สามารถใช้ Feature Importance เพื่อวิเคราะห์ความสำคัญของแต่ละคุณลักษณะ
    
    **ข้อเสียของ Random Forest**
    - โมเดลอาจจะมีขนาดใหญ่และใช้เวลาในการคำนวณสูง
    - ผลลัพธ์ที่ได้จากหลายๆ ต้นไม้อาจทำให้ไม่สามารถตีความได้ง่าย

    **วิธีการทำงาน**:
    1. การสุ่มข้อมูล (Bootstrapping)
    2. การสร้างต้นไม้การตัดสินใจ
    3. การรวมผลการตัดสินใจจากหลายๆ ต้นไม้ (Majority Voting สำหรับ Classification)
    """)

    # อธิบายทฤษฎีของ XGBoost
    st.subheader("🚀 XGBoost")
    st.markdown("""
    **XGBoost** (Extreme Gradient Boosting) เป็นเทคนิคการเรียนรู้แบบ Gradient Boosting ที่มีประสิทธิภาพสูงในการแข่งขันด้าน Machine Learning
    โดยใช้วิธีการสร้างโมเดลหลายๆ โมเดล (Tree-based Models) และทำการปรับแต่งให้ดีขึ้นในแต่ละรอบ (Boosting) ซึ่งมีการคำนวณค่าผลลัพธ์จากการเพิ่มน้ำหนักให้กับตัวอย่างที่โมเดลก่อนหน้านี้ทำนายผิด

    **ข้อดีของ XGBoost**
    - มีประสิทธิภาพสูงในการแข่งขัน
    - รองรับการปรับแต่งพารามิเตอร์มากมาย เช่น learning_rate, max_depth
    - สามารถลด overfitting ได้ดีด้วยการใช้ Regularization
    - ทำงานได้ดีทั้งสำหรับปัญหาการจำแนกประเภท (Classification) และการพยากรณ์ค่า (Regression)

    **ข้อเสียของ XGBoost**
    - ต้องการการปรับแต่งพารามิเตอร์ที่ดี
    - หากปรับแต่งไม่ดีอาจจะเกิด overfitting ได้

    **วิธีการทำงาน**:
    1. การสร้างโมเดลหลายๆ โมเดล (Trees)
    2. การคำนวณ Gradient (ความคลาดเคลื่อน) ในแต่ละรอบ
    3. การปรับปรุงโมเดลที่มีอยู่โดยเพิ่มน้ำหนักให้กับตัวอย่างที่ถูกทำนายผิด
    4. การ Regularization เพื่อลด overfitting
    """)

with tab1:
    st.header("โหลดไฟล์และเตรียมข้อมูล")
    
    # แสดงโค้ดการโหลดไฟล์
    st.subheader("⭐ โหลดไฟล์ CSV")
    with st.expander("แสดงโค้ด"):
        st.code("""
from google.colab import files
import pandas as pd

uploaded = files.upload()
df = pd.read_csv('/content/Loan_default.csv')
        """)
    
    # สร้างข้อมูลสาธิต
    @st.cache_data
    def generate_sample_data():
        np.random.seed(42)
        
        data = {
            'loanid': list(range(1000, 1010)),
            'age': np.random.randint(20, 70, 10),
            'income': np.random.randint(20000, 150000, 10), 
            'loanamount': np.random.randint(5000, 100000, 10),
            'creditscore': np.random.randint(300, 850, 10),
            'monthsemployed': np.random.randint(0, 240, 10),
            'numcreditlines': np.random.randint(0, 15, 10),
            'interestrate': np.random.uniform(3, 20, 10).round(2),
            'loanterm': np.random.choice([12, 24, 36, 48, 60], 10),
            'dtiratio': np.random.uniform(0.1, 0.6, 10).round(2),
            'education': np.random.choice(['HighSchool', 'Bachelor', 'Master', 'PhD'], 10),
            'employmentstatus': np.random.choice(['Employed', 'Self-employed', 'Unemployed'], 10),
            'maritalstatus': np.random.choice(['Single', 'Married', 'Divorced'], 10),
            'homeownership': np.random.choice(['Own', 'Rent', 'Mortgage'], 10),
            'default': np.random.choice([0, 1], 10, p=[0.8, 0.2])
        }
        
        return pd.DataFrame(data)
    
    df = generate_sample_data()
    
    # แสดงข้อมูล
    st.subheader("🔍 สำรวจข้อมูล")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 สถิติข้อมูล")
        st.dataframe(df.describe().round(2))
    
    with col2:
        st.subheader("ℹ️ ข้อมูลทั่วไป")
        st.write(f"จำนวนข้อมูล: {len(df)} แถว")
        st.write(f"จำนวนคอลัมน์: {len(df.columns)} คอลัมน์")
        st.write(f"อัตราการผิดนัดชำระหนี้: {df['default'].mean()*100:.2f}%")

     # แสดงที่มาของข้อมูล
    st.subheader("📚 ที่มาของข้อมูล")
    st.markdown("""
ข้อมูลนี้สามารถดาวน์โหลดได้จาก [Kaggle: Loan Default Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default)

- **ชื่อชุดข้อมูล:** Loan Default Dataset
- **ผู้จัดทำ:** Nikhil Sharma
- **ลักษณะข้อมูล:** ชุดข้อมูลการผิดนัดชำระหนี้ของลูกค้าซึ่งใช้ทำนายว่าลูกค้าคนนั้นจะผิดนัดชำระหนี้หรือไม่
""")

with tab2:
    st.header("🔄 การเตรียมข้อมูล")
    
    # แสดงการจัดการค่าหายไป
    st.subheader("❓ จัดการค่าหายไป")
    with st.expander("แสดงโค้ด"):
        st.code("""
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
df[['loanamount', 'creditscore']] = imputer.fit_transform(df[['loanamount', 'creditscore']])
        """)
    
    # สร้างการแสดงผลก่อน-หลัง
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ก่อนการจัดการค่าหายไป")
        # สมมติว่ามีค่าหายไป
        missing_df = df.copy()
        missing_df.loc[0:2, 'loanamount'] = np.nan
        missing_df.loc[3:4, 'creditscore'] = np.nan
        st.dataframe(missing_df[['loanamount', 'creditscore']])
        
        # แสดงจำนวนค่าหายไป
        missing_counts = missing_df.isna().sum()
        st.write(f"ค่าหายไปในคอลัมน์ loanamount: {missing_counts['loanamount']}")
        st.write(f"ค่าหายไปในคอลัมน์ creditscore: {missing_counts['creditscore']}")
    
    with col2:
        st.subheader("หลังการจัดการค่าหายไป")
        imputer = SimpleImputer(strategy='mean')
        fixed_df = missing_df.copy()
        fixed_df[['loanamount', 'creditscore']] = imputer.fit_transform(missing_df[['loanamount', 'creditscore']])
        st.dataframe(fixed_df[['loanamount', 'creditscore']])
        
        # แสดงจำนวนค่าหายไป
        fixed_missing_counts = fixed_df.isna().sum()
        st.write(f"ค่าหายไปในคอลัมน์ loanamount: {fixed_missing_counts['loanamount']}")
        st.write(f"ค่าหายไปในคอลัมน์ creditscore: {fixed_missing_counts['creditscore']}")
        
    # แสดงการแปลงข้อมูลเป็นตัวเลข
    st.subheader("🏢 แปลงข้อมูลเป็นตัวเลข")
    with st.expander("แสดงโค้ด"):
        st.code("""
X = pd.get_dummies(df.drop(columns=['default', 'loanid'], errors='ignore'), drop_first=True)
y = df['default']
        """)
    
    # สร้างตัวอย่าง One-hot encoding
    df_original = pd.DataFrame({
        'education': ['HighSchool', 'Bachelor', 'Master']
    })
    
    df_encoded = pd.get_dummies(df_original, drop_first=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ข้อมูลก่อน One-hot encoding")
        st.dataframe(df_original)
    
    with col2:
        st.subheader("ข้อมูลหลัง One-hot encoding")
        st.dataframe(df_encoded)
    
    # แบ่งข้อมูล Train/Test
    st.subheader("แบ่งข้อมูล Train/Test")
    with st.expander("แสดงโค้ด"):
        st.code("""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        """)
    
    # แสดงสัดส่วนการแบ่ง
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=['ข้อมูลทั้งหมด'],
        x=[100],
        name='ข้อมูลทั้งหมด',
        orientation='h',
        marker=dict(color='lightgrey')
    ))
    fig.add_trace(go.Bar(
        y=['แบ่งข้อมูล'],
        x=[80],
        name='ชุดข้อมูลฝึกฝน (Train)',
        orientation='h',
        marker=dict(color='royalblue')
    ))
    fig.add_trace(go.Bar(
        y=['แบ่งข้อมูล'],
        x=[20],
        name='ชุดข้อมูลทดสอบ (Test)',
        orientation='h',
        marker=dict(color='lightcoral')
    ))
    fig.update_layout(
        title='การแบ่งข้อมูลฝึกฝนและทดสอบ (80/20)',
        xaxis=dict(
            title='เปอร์เซ็นต์',
            ticksuffix='%'
        ),
        barmode='stack',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("🤖 การสร้างโมเดล Machine Learning")
    
    # Random Forest
    st.subheader("เทรนโมเดล Random Forest 🌲")
    with st.expander("แสดงโค้ด"):
        st.code("""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

rf_model = RandomForestClassifier(n_estimators=30, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
        """)
    
    # แสดงแผนภาพโมเดล Random Forest
    st.subheader("โครงสร้าง Random Forest")
    # สร้างแผนภาพด้วย Plotly แทน
    fig = go.Figure()
    
    # สร้างโครงสร้างต้นไม้อย่างง่าย
    for i in range(3):
        # ฐานของต้นไม้
        fig.add_shape(type="rect", 
                     xref="x", yref="y",
                     x0=i*35, y0=0, x1=i*35+20, y1=10,
                     line=dict(color="brown"),
                     fillcolor="brown")
        
        # กิ่งของต้นไม้
        fig.add_shape(type="circle", 
                     xref="x", yref="y",
                     x0=i*35-10, y0=10, x1=i*35+30, y1=50,
                     line=dict(color="green"),
                     fillcolor="green")
        
    fig.update_layout(
        height=200,
        width=600,
        title="Random Forest - Multiple Decision Trees",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # XGBoost
    st.subheader("เทรนโมเดล XGBoost 🚀")
    with st.expander("แสดงโค้ด"):
        st.code("""
import xgboost as xgb

xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)
        """)
    
    # แสดงพารามิเตอร์โมเดล
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("พารามิเตอร์ Random Forest")
        st.markdown("""
        - n_estimators: 30
        - criterion: gini
        - max_depth: None
        - min_samples_split: 2
        - min_samples_leaf: 1
        - random_state: 42
        """)
    
    with col2:
        st.subheader("พารามิเตอร์ XGBoost")
        st.markdown("""
        - n_estimators: 100
        - learning_rate: 0.1
        - max_depth: 6
        - booster: gbtree
        - subsample: 1
        - random_state: 42
        """)

with tab4:
    st.header("📈 ผลลัพธ์")
    
    # สร้างผลลัพธ์สมมติ
    rf_accuracy = 0.82
    xgb_accuracy = 0.87
    
    # แสดงผลลัพธ์
    st.subheader("🎯 สรุปผลลัพธ์")
    
    # สร้างกราฟเปรียบเทียบ
    models = ['Random Forest', 'XGBoost']
    accuracy = [rf_accuracy, xgb_accuracy]
    
    fig = px.bar(
        x=models, 
        y=accuracy,
        text=[f"{acc:.2%}" for acc in accuracy],
        color=accuracy,
        color_continuous_scale='Blues',
        title="เปรียบเทียบความแม่นยำของโมเดล"
    )
    fig.update_layout(
        xaxis_title="โมเดล",
        yaxis_title="ความแม่นยำ (Accuracy)",
        yaxis=dict(tickformat=".0%", range=[0, 1])
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # แสดงผลลัพธ์การทำนาย
    st.subheader("ตัวอย่างผลการทำนาย")
    
    # สร้างข้อมูลตัวอย่างการทำนาย
    predictions = {
        'ลูกค้า ID': [1001, 1002, 1003, 1004, 1005],
        'อายุ': [35, 42, 28, 55, 33],
        'รายได้': [50000, 80000, 35000, 120000, 65000],
        'ยอดสินเชื่อ': [25000, 65000, 15000, 85000, 35000],
        'ทำนาย (Random Forest)': ['ไม่ผิดนัด', 'ไม่ผิดนัด', 'ผิดนัด', 'ไม่ผิดนัด', 'ผิดนัด'],
        'ทำนาย (XGBoost)': ['ไม่ผิดนัด', 'ไม่ผิดนัด', 'ผิดนัด', 'ไม่ผิดนัด', 'ไม่ผิดนัด'],
        'ค่าจริง': ['ไม่ผิดนัด', 'ไม่ผิดนัด', 'ผิดนัด', 'ไม่ผิดนัด', 'ผิดนัด']
    }
    predictions_df = pd.DataFrame(predictions)
    st.dataframe(predictions_df)
    
    # แสดงข้อมูลปัจจัยที่สำคัญ
    st.subheader("ปัจจัยสำคัญในการทำนาย")
    
    # สร้างข้อมูลสำหรับแสดงความสำคัญของปัจจัย
    feature_importance = {
        'ปัจจัย': ['คะแนนเครดิต', 'รายได้', 'อายุ', 'อัตราดอกเบี้ย', 'จำนวนเดือนที่ทำงาน'],
        'ความสำคัญ': [0.35, 0.25, 0.15, 0.12, 0.08]
    }
    feature_df = pd.DataFrame(feature_importance)
    
    # สร้างกราฟแสดงความสำคัญของปัจจัย
    fig = px.bar(
        feature_df,
        x='ความสำคัญ',
        y='ปัจจัย',
        orientation='h',
        color='ความสำคัญ',
        color_continuous_scale='Blues',
        title="ความสำคัญของปัจจัยในการทำนาย"
    )
    fig.update_layout(
        xaxis_title="ความสำคัญ",
        yaxis_title="ปัจจัย",
        xaxis=dict(tickformat=".0%"),
        yaxis=dict(categoryorder='total ascending')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # สรุปความแม่นยำ
    st.subheader("สรุปความแม่นยำของโมเดล")
    st.markdown(f"""
    - Random Forest ได้ความแม่นยำประมาณ {rf_accuracy:.2%}
    - XGBoost ได้ความแม่นยำสูงขึ้น {xgb_accuracy:.2%}
    """)
    
    # แสดงแนวทางการพัฒนาต่อ
    st.subheader("แนวทางการพัฒนาต่อ")
    st.markdown("""
    1. ปรับแต่งพารามิเตอร์ของโมเดลให้เหมาะสมยิ่งขึ้น
    2. ทดลองใช้โมเดล Machine Learning อื่นๆ เช่น Neural Network
    3. เพิ่มข้อมูลเพื่อให้โมเดลเรียนรู้ได้ดีขึ้น
    4. ทำ Feature Engineering เพื่อสร้างปัจจัยใหม่ที่มีความสัมพันธ์สูงกับการผิดนัดชำระ
    """)