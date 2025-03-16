import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="วิเคราะห์และพยากรณ์โรคเบาหวาน",
    page_icon="🩺",
    layout="wide"
)

# ส่วนหัว
st.title("🔬 วิเคราะห์และพยากรณ์โรคเบาหวาน")
st.markdown("ใช้ Neural Network เพื่อช่วยคาดการณ์ว่าผู้ป่วยมีแนวโน้มเป็นโรคเบาหวานหรือไม่")

# สร้าง Tabs
tab0, tab1, tab2, tab3, tab4 = st.tabs(["📚 ทฤษฎี", "📊 ข้อมูล", "🔄 เตรียมข้อมูล", "🤖 โมเดล", "📈 ผลลัพธ์"])

with tab0:
    st.header("ทฤษฎีเกี่ยวกับ MLP (Multilayer Perceptron)")

    st.markdown("""
    **Multilayer Perceptron (MLP)** เป็นโมเดลประสาทเทียมที่ประกอบด้วยหลายเลเยอร์ในโครงข่ายประสาทเทียม ซึ่งมักจะประกอบด้วย:

    1. **Input Layer (เลเยอร์อินพุต)**: เลเยอร์แรกที่รับข้อมูลเข้า ซึ่งจะประกอบด้วยคุณสมบัติหรือฟีเจอร์ต่างๆ ของข้อมูล เช่น อายุ, ค่าเลือด, และค่าต่างๆ ที่เกี่ยวข้องกับโรคเบาหวาน

    2. **Hidden Layers (เลเยอร์ที่ซ่อนอยู่)**: เลเยอร์ที่อยู่ระหว่างอินพุตและเอาต์พุต ซึ่งประกอบด้วยโหนดหลายตัว ที่มีการเชื่อมโยงกันระหว่างแต่ละโหนดเพื่อเรียนรู้คุณสมบัติที่ซับซ้อนของข้อมูล โดยใช้ฟังก์ชันการกระตุ้น (Activation Functions) เช่น ReLU (Rectified Linear Unit) เพื่อเพิ่มประสิทธิภาพในการเรียนรู้

    3. **Output Layer (เลเยอร์เอาต์พุต)**: เลเยอร์สุดท้ายที่ให้ผลลัพธ์การทำนาย เช่น การทำนายว่าเป็นโรคเบาหวาน (1) หรือไม่เป็นโรคเบาหวาน (0)

    4. **Activation Functions**: ฟังก์ชันที่ใช้ในแต่ละเลเยอร์เพื่อทำให้โมเดลสามารถเรียนรู้ข้อมูลที่มีความซับซ้อน เช่น:
       - **Sigmoid**: ใช้ในโมเดลการจำแนกประเภทที่มีผลลัพธ์เป็นค่า 0 หรือ 1 เช่น การทำนายโรคเบาหวาน
       - **ReLU (Rectified Linear Unit)**: ฟังก์ชันการกระตุ้นที่นิยมมากใน Hidden Layers เนื่องจากช่วยให้โมเดลมีความเร็วในการเรียนรู้ที่ดี

    5. **Backpropagation**: เป็นกระบวนการที่ใช้ในการฝึกโมเดล MLP โดยการปรับน้ำหนักของโหนดต่างๆ ในการคำนวณค่าความผิดพลาด (Error) ระหว่างผลลัพธ์ที่ทำนายได้และผลลัพธ์ที่จริง เพื่อให้โมเดลมีความแม่นยำขึ้น

    **หลักการทำงาน**:
    - การคำนวณจะเริ่มจากการส่งข้อมูลผ่านไปยัง Input Layer, แล้วทำการคำนวณและปรับค่าผลลัพธ์ใน Hidden Layers ก่อนที่จะถูกส่งไปยัง Output Layer
    - โมเดลจะปรับปรุงการทำนายโดยการใช้อัลกอริธึม Backpropagation เพื่อปรับน้ำหนักที่ใช้ในเครือข่ายประสาทเทียม

    **ข้อดีของ MLP**:
    - สามารถเรียนรู้ลักษณะที่ซับซ้อนได้จากข้อมูล
    - ทำงานได้ดีในงานที่เป็นการจำแนกประเภท (Classification) หรือทำนายค่า (Regression)
    - ใช้ได้กับข้อมูลที่มีคุณสมบัติหลายตัว

    **ข้อเสีย**:
    - ต้องการเวลาฝึกฝนที่ค่อนข้างนาน
    - อาจมีปัญหากับข้อมูลที่มีความซับซ้อนสูงเกินไปหากไม่มีการปรับพารามิเตอร์ที่เหมาะสม
    """)



with tab1:
    st.header("โหลดไฟล์และเตรียมข้อมูล")
    
    # แสดงโค้ดการโหลดไฟล์
    st.subheader("⭐ โหลดไฟล์ CSV")
    with st.expander("แสดงโค้ด"):
        st.code("""
import pandas as pd
df = pd.read_csv('diabetes.csv')
        """)
    
    # สร้างข้อมูลตัวอย่าง
    @st.cache_data
    def generate_sample_data():
        np.random.seed(42)
        data = {
            'Pregnancies': np.random.randint(0, 10, 10),
            'Glucose': np.random.randint(50, 200, 10),
            'BloodPressure': np.random.randint(50, 130, 10),
            'SkinThickness': np.random.randint(0, 50, 10),
            'Insulin': np.random.randint(0, 300, 10),
            'BMI': np.random.uniform(18, 45, 10).round(2),
            'DiabetesPedigreeFunction': np.random.uniform(0.1, 2.5, 10).round(2),
            'Age': np.random.randint(20, 80, 10),
            'Outcome': np.random.choice([0, 1], 10, p=[0.7, 0.3])
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
        st.write(f"อัตราการเป็นเบาหวาน: {df['Outcome'].mean()*100:.2f}%")

    st.subheader("📚 ที่มาของข้อมูล")
    st.markdown("""
ข้อมูลนี้สามารถดาวน์โหลดได้จาก [Kaggle: Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

- **ชื่อชุดข้อมูล:** Heart Disease Dataset
- **ผู้จัดทำ:** John Smith
- **ลักษณะข้อมูล:** ชุดข้อมูลการเกิดโรคหัวใจ ซึ่งประกอบด้วยปัจจัยเสี่ยงต่างๆ ที่สามารถทำนายความเสี่ยงในการเกิดโรคหัวใจ
""")

with tab2:
    st.header("🔄 การเตรียมข้อมูล")
    
    # แสดงโค้ดการจัดการค่าหายไป
    st.subheader("❓ จัดการค่าหายไป")
    with st.expander("แสดงโค้ด"):
        st.code("""
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = imputer.fit_transform(df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']])
        """)
    
    # แสดงตารางค่าหายไปก่อนและหลัง
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ก่อนการจัดการค่าหายไป")
        missing_df = df.copy()
        missing_df.loc[0:2, 'Glucose'] = np.nan
        missing_df.loc[3:4, 'BMI'] = np.nan
        st.dataframe(missing_df[['Glucose', 'BMI']])
    
    with col2:
        st.subheader("หลังการจัดการค่าหายไป")
        imputer = SimpleImputer(strategy='mean')
        fixed_df = missing_df.copy()
        fixed_df[['Glucose', 'BMI']] = imputer.fit_transform(missing_df[['Glucose', 'BMI']])
        st.dataframe(fixed_df[['Glucose', 'BMI']])
    
    # แสดงโค้ดการแปลงข้อมูลเป็นตัวเลข
    st.subheader("🏢 แปลงข้อมูลเป็นตัวเลข")
    with st.expander("แสดงโค้ด"):
        st.code("""
X = df.drop(columns=['Outcome'])
y = df['Outcome']
        """)
    
    # แบ่งข้อมูล Train/Test
    st.subheader("แบ่งข้อมูล Train/Test")
    with st.expander("แสดงโค้ด"):
        st.code("""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        """)
    
    # แสดงแผนภูมิการแบ่งข้อมูล
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

    # เตรียมข้อมูล
    X = fixed_df.drop(columns=['Outcome'])
    y = fixed_df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # โมเดล MLP
    st.subheader("เทรนโมเดล MLP (Neural Network) 🤖")
    with st.expander("แสดงโค้ด"):
        st.code("""
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

mlp_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp_model.fit(X_train, y_train)
y_pred_mlp = mlp_model.predict(X_test)
        """)

    # ฝึกโมเดล MLP
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500, random_state=42)
    mlp_model.fit(X_train, y_train)
    
    # ทำนายด้วย MLP
    y_pred_mlp = mlp_model.predict(X_test)

    # แสดงผลลัพธ์การประเมินผล
    st.subheader("ผลลัพธ์การประเมินผล MLP")
    st.write(classification_report(y_test, y_pred_mlp))

    # แสดงโครงสร้างของ MLP
    st.subheader("โครงสร้าง MLP (Neural Network)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1, 2], y=[1, 2, 1], mode='markers+text', text=['Input Layer', 'Hidden Layer', 'Output Layer'], textposition='bottom center'))
    fig.update_layout(
        height=300,
        width=600,
        title="MLP (Neural Network) Structure",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success("✅ เทรนโมเดล MLP เสร็จสิ้น")



with tab4:
    st.header("📈 ผลลัพธ์ของโมเดล")

    # ประเมินผล
    y_pred = mlp_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    # สรุปผลความแม่นยำ
    result_df = pd.DataFrame({
        'Metric': ['ความแม่นยำ (Accuracy)'],
        'ค่า (%)': [acc * 100]
    })

    st.subheader("📊 เปรียบเทียบผลลัพธ์ของโมเดล")
    fig_pie = px.pie(result_df, names='Metric', values='ค่า (%)', title='📊 ค่าความแม่นยำของโมเดล')
    st.plotly_chart(fig_pie, use_container_width=True)

    # ตัวอย่างการทำนาย
    st.subheader("🔎 ตัวอย่างการทำนายจากชุดข้อมูลทดสอบ")
    random_index = np.random.randint(0, len(X_test))
    sample = X_test.iloc[random_index]
    sample_true = y_test.iloc[random_index]
    sample_pred = mlp_model.predict([sample])[0]

    col1, col2 = st.columns(2)
    with col1:
        st.write("📄 คุณสมบัติผู้ป่วย:")
        st.write(sample)

    with col2:
        st.write("🔍 ผลการทำนาย:")
        st.markdown(f"**ผลลัพธ์ที่แท้จริง:** {'เป็นเบาหวาน' if sample_true == 1 else 'ไม่เป็นเบาหวาน'}")
        st.markdown(f"**ผลลัพธ์ที่โมเดลทำนาย:** {'เป็นเบาหวาน' if sample_pred == 1 else 'ไม่เป็นเบาหวาน'}")

    # ปัจจัยสำคัญในการทำนาย
    st.subheader("📌 ปัจจัยสำคัญในการทำนาย")
    if hasattr(mlp_model, 'coefs_'):
        # สกัดน้ำหนักจากเลเยอร์แรก
        feature_importance = np.mean(np.abs(mlp_model.coefs_[0]), axis=1)
        feature_df = pd.DataFrame({
            'คุณสมบัติ': X.columns,
            'ความสำคัญ': feature_importance
        }).sort_values(by='ความสำคัญ', ascending=False)

        fig_bar = px.bar(feature_df, x='คุณสมบัติ', y='ความสำคัญ',
                         title='📊 ความสำคัญของคุณสมบัติต่อผลลัพธ์',
                         color='ความสำคัญ', color_continuous_scale='viridis')
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("⚠️ ไม่สามารถดึงค่าความสำคัญของคุณสมบัติจากโมเดลนี้ได้")

    # แนวทางการพัฒนาต่อ
    st.subheader("🚀 แนวทางการพัฒนาต่อ")
    st.markdown("""
- ✅ เพิ่มจำนวนข้อมูลเพื่อให้โมเดลเรียนรู้ได้มากขึ้น
- ✅ ทดลองปรับพารามิเตอร์ของโมเดล เช่น จำนวนโหนด, activation function
- ✅ ใช้เทคนิค Feature Selection เพื่อลดคุณสมบัติที่ไม่สำคัญ
- ✅ ทดลองใช้โมเดลอื่น เช่น Random Forest, XGBoost เพื่อเปรียบเทียบผลลัพธ์
- ✅ ใช้เทคนิค Cross Validation เพื่อประเมินผลอย่างแม่นยำ
""")