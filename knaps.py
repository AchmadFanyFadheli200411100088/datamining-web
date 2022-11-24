import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import streamlit as st

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
    x = heart.iloc[:, 1:-1].values
    y = heart.iloc[:, -1].values
    x,y

    "### Splitting the dataset into training and testing data"
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 0)

    "### Feature Scaling"
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_train,x_test
