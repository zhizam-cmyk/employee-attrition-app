import streamlit as st
import pandas as pd
import pickle

# تحميل النموذج
model = pickle.load(open('model.pkl', 'rb'))

# تحميل أسماء الأعمدة لتطابق النموذج المُدرّب
columns = pd.read_csv('columns.csv', header=None)[0].tolist()

# عنوان التطبيق بتنسيق جميل
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>📉 Predict Employee Attrition</h1>",
    unsafe_allow_html=True
)
st.sidebar.header("Enter Employee Data")

# دالة لإدخال البيانات
def user_input_features():
    age = st.sidebar.slider('Age', 18, 60, 30)
    income = st.sidebar.slider('Monthly Income', 1000, 20000, 5000)
    overtime = st.sidebar.selectbox('OverTime', ['Yes', 'No'])
    satisfaction = st.sidebar.slider('Job Satisfaction', 1, 4, 3)
    years = st.sidebar.slider('Years at Company', 0, 40, 5)
    distance = st.sidebar.slider('Distance From Home (km)', 0, 30, 10)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    education = st.sidebar.selectbox('Education', ['High School', 'Bachelor', 'Master', 'Doctor'])
    department = st.sidebar.selectbox('Department', ['HR', 'Sales', 'R&D'])
    marital_status = st.sidebar.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])

    data = {
        'Age': age,
        'MonthlyIncome': income,
        'OverTime': overtime,
        'JobSatisfaction': satisfaction,
        'YearsAtCompany': years,
        'DistanceFromHome': distance,
        'Gender': gender,
        'Education': education,
        'Department': department,
        'MaritalStatus': marital_status
    }

    return pd.DataFrame([data])

# جمع البيانات
input_df = user_input_features()

# تحويل القيم النصية إلى رقمية بنفس طريقة التدريب
input_df['OverTime'] = input_df['OverTime'].map({'Yes': 1, 'No': 0})
input_df['Gender'] = input_df['Gender'].map({'Male': 1, 'Female': 0})
input_df['Education'] = input_df['Education'].map({'High School': 1, 'Bachelor': 2, 'Master': 3, 'Doctor': 4})
input_df['Department'] = input_df['Department'].map({'HR': 0, 'Sales': 1, 'R&D': 2})
input_df['MaritalStatus'] = input_df['MaritalStatus'].map({'Single': 0, 'Married': 1, 'Divorced': 2})

# إنشاء DataFrame يحتوي على جميع الأعمدة المطلوبة
full_input = pd.DataFrame(columns=columns)
full_input.loc[0] = 0  # تعيين كل القيم إلى صفر مبدئيًا

# تحديث القيم المُدخلة
for col in full_input.columns:
    if col in input_df.columns:
        full_input[col] = input_df[col]
    else:
        if col in ['OverTime', 'Gender', 'Education', 'Department', 'MaritalStatus']:
            full_input[col] = ['Unknown']
        elif full_input[col].dtype == 'O':
            full_input[col] = ['Unknown']
        else:
            full_input[col] = [0]

# زر التنبؤ
if st.button("Predict"):
    # التنبؤ بالنتيجة
    prediction = model.predict(full_input)[0]
    prediction_proba = model.predict_proba(full_input)[0]

    # عرض المدخلات النهائية
    st.subheader("🗂 Final Input to Model:")
    st.dataframe(full_input)

    # عرض النتيجة
    st.subheader("🎯 Prediction Result")
    if prediction == 1:
        st.success("⚠️ The employee is likely to leave.")
    else:
        st.success("✅ The employee is likely to stay.")

    # عرض الاحتمالية
    st.subheader("📊 Probability of Attrition")
    st.write(f"{prediction_proba[1]*100:.2f}% chance the employee will leave.")

