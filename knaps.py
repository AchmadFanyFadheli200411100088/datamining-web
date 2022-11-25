import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import streamlit as st

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import altair as alt

from sklearn.utils.validation import joblib

st.title("PENAMBANGAN DATA")
st.write("By: Indyra Januar - 200411100022")
st.write("Grade: Penambangan Data C")
upload_data, preporcessing, modeling, implementation = st.tabs(["Upload Data", "Prepocessing", "Modeling", "Implementation"])


with upload_data:
    st.write("""# Upload File""")
    st.write("Dataset yang digunakan adalah Heart Attack Dataset yang diambil dari https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset")
    st.write("Heart Attack (Serangan Jantung) adalah kondisi medis darurat ketika darah yang menuju ke jantung terhambat.")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)


with preporcessing:
    st.write("""# Preprocessing""")

    "### There's no need for categorical encoding"
    x = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values
    x,y

    "### Splitting the dataset into training and testing data"
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 0)
    st.write("Shape for training data", x_train.shape, y_train.shape)
    st.write("Shape for testing data", x_test.shape, y_test.shape)

    "### Feature Scaling"
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_train,x_test

with modeling:
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=4)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    st.write("""# Modeling """)
    st.subheader("Berikut ini adalah pilihan untuk Modeling")
    st.write("Pilih Model yang Anda inginkan untuk Cek Akurasi")
    naive = st.checkbox('Naive Bayes')
    kn = st.checkbox('K-Nearest Neighbor')
    des = st.checkbox('SVM')
    mod = st.button("Modeling")

    # NB
    model = GaussianNB()
    model.fit(x_train, y_train)

    predicted = model.predict(x_test)

    akurasi_nb = round(accuracy_score(y_test, predicted)*100)

    #KNN
    model = KNeighborsClassifier(n_neighbors = 1)  
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    
    akurasi_knn = round(accuracy_score(y_test, predicted.round())*100)

    #SVM
    model = SVC()
    model.fit(x_train, y_train)
    
    predicted = model.predict(x_test)
    akurasi_svm = round(accuracy_score(y_test, predicted)*100)

    if naive :
        if mod :
            st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(akurasi_nb))
    if kn :
        if mod:
            st.write("Model KNN accuracy score : {0:0.2f}" . format(akurasi_knn))
    if des :
        if mod :
            st.write("Model SVM score : {0:0.2f}" . format(akurasi_svm))

    eval = st.button("Evaluasi semua model")
    if eval :
        # st.snow()
        source = pd.DataFrame({
            'Nilai Akurasi' : [akurasi_nb,akurasi_knn,akurasi_svm],
            'Nama Model' : ['Naive Bayes','KNN','SVM']
        })

        bar_chart = alt.Chart(source).mark_bar().encode(
            y = 'Nilai Akurasi',
            x = 'Nama Model'
        )

        st.altair_chart(bar_chart,use_container_width=True)

with implementation:
    st.write("# Implementation")
    Age = st.number_input('Masukkan Umur Pasien')

    #sex
    sex = st.radio("Sex",('Male', 'Female', 'Other'))
    if sex == "Male":
        sex_Female = 0
        sex_Male = 1
        sex_Other = 0
    elif sex == "Female" :
        sex_Female = 1
        sex_Male = 0
        sex_Other = 0
    elif sex == "Other" :
        sex_Female = 0
        sex_Male = 0
        sex_Other = 1

    #chest pain
    cp = st.radio("Chest Pain",('Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'))
    if cp == "Typical Angina":
        cp_ta = 1
        cp_aa = 0
        cp_na = 0
        cp_a = 0
    elif cp == "Atypical Angina":
        cp_ta = 0
        cp_aa = 1
        cp_na = 0
        cp_a = 0
    elif cp == "Non-anginal Pain":
        cp_ta = 0
        cp_aa = 0
        cp_na = 1
        cp_a = 0
    elif cp == "Asymptomatic":
        cp_ta = 0
        cp_aa = 0
        cp_na = 0
        cp_a = 1
    
    #blood pressure
    trtbps = st.number_input('Tekanan Darah (mm Hg)')

    #cholestoral
    chol = st.number_input('Rasio Kolesterol (mg/dl)')

    #fasting blood sugar
    fbs = st.radio("Glukosa Puasa > 120 mg/dl",('No', 'Yes'))
    if fbs == "Yes":
        fbs_y = 1
        fbs_n = 0
    elif fbs == "No":
        fbs_y = 0
        fbs_n = 1
    
    #electrocardiographic results
    restecg = st.radio("Elektrokardiografi",('Normal', 'ST-T wave normality', 'Left ventricular hypertrophy'))
    if restecg == "Normal":
        restecg_1 = 1
        restecg_2 = 0
        restecg_3 = 0
    elif restecg == "ST-T wave normality" :
        restecg_1 = 0
        restecg_2 = 1
        restecg_3 = 0
    elif restecg == "Left ventricular hypertrophy" :
        restecg_1 = 0
        restecg_2 = 0
        restecg_3 = 1
    
    #Maximum heart rate achieved
    thalachh = st.number_input('Detak jantung maksimal')

    #Exercise induced angina
    exang = st.radio("Exercise induced angina",('No', 'Yes'))
    if exang == "Yes":
        exang_y = 1
        exang_n = 0
    elif exang == "No":
        exang_y = 0
        exang_n = 1

    #old peak
    oldpeak = st.number_input('ST depression induced by exercise relative to rest')

    #slope
    slope = st.radio("Kemiringan segmen ST latihan puncak",('Downsloping', 'Flat', 'Upsloping'))
    if slope == "Downsloping":
        slope_1 = 1
        slope_2 = 0
        slope_3 = 0
    elif slope == "Flat" :
        slope_1 = 0
        slope_2 = 1
        slope_3 = 0
    elif slope == "Upsloping" :
        slope_1 = 0
        slope_2 = 0
        slope_3 = 1

    #Number of major vessels
    caa = st.number_input('Number of major vessels (0-3) colored by flourosopy')

    #thall
    thall = st.radio("Hasil Tes Stres Talium",('Normal', 'Fixed Defect', 'Reversable Defect'))
    if thall == "Normal":
        thall_1 = 1
        thall_2 = 0
        thall_3 = 0
    elif thall == "Fixed Defect" :
        thall_1 = 0
        thall_2 = 1
        thall_3 = 0
    elif thall == "Reversable Defect" :
        thall_1 = 0
        thall_2 = 0
        thall_3 = 1
    
    def submit():
        # input
        inputs = np.array([[
            sex, sex_Female,  sex_Male, sex_Other,
            cp, cp_ta, cp_aa, cp_na, cp_a,
            trtbps,
            chol,
            fbs, fbs_y, fbs_n,
            restecg, restecg_1, restecg_2, restecg_3,
            thalachh,
            exang, exang_y, exang_n,
            oldpeak,
            slope, slope_1, slope_2, slope_3,
            caa,
            thall, thall_1, thall_2, thall_3
            ]])

        le = joblib.load("le.save")
        model1 = joblib.load("knn.joblib")
        y_pred3 = model1.predict(inputs)
        st.write(f"Berdasarkan data yang Anda masukkan, maka anda dinyatakan : {le.inverse_transform(y_pred3)[0]}")
    
    all = st.button("Submit")
    if all :
        st.balloons()
        submit()
    