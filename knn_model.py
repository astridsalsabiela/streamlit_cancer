import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
import sklearn.metrics as metrics
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier

import pickle
import warnings
warnings.filterwarnings("ignore")

# Load the Breast Cancer Wisconsin dataset (replace with the actual path to your dataset)
df = pd.read_csv('data.csv')

# Preprocessing dataset
columns_to_drop = ['id', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst', 'Unnamed: 32']
df.drop(columns=columns_to_drop, inplace=True)

# Pisahkan fitur dan label
X = df[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']]
y = df['diagnosis']

# Inisialisasi scaler dan normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Inisialisasi model KNN
knn_model = KNeighborsClassifier(n_neighbors=3)

# Latih model
knn_model.fit(X_scaled, y)

# Simpan model KNN menggunakan pickle
with open('knn_model.pickle', 'wb') as model_file:
    pickle.dump(knn_model, model_file)

# Simpan scaler menggunakan pickle
with open('scaler.pickle', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Streamlit App
st.title('Breast Cancer Wisconsin Data Analysis with KNN')

# Sidebar untuk input pengguna
st.sidebar.header('Input Pengguna')

# Masukkan nilai fitur
radius_mean = st.sidebar.slider('Radius Mean', float(X['radius_mean'].min()), float(X['radius_mean'].max()), float(X['radius_mean'].mean()))
texture_mean = st.sidebar.slider('Texture Mean', float(X['texture_mean'].min()), float(X['texture_mean'].max()), float(X['texture_mean'].mean()))
perimeter_mean = st.sidebar.slider('Perimeter Mean', float(X['perimeter_mean'].min()), float(X['perimeter_mean'].max()), float(X['perimeter_mean'].mean()))
area_mean = st.sidebar.slider('Area Mean', float(X['area_mean'].min()), float(X['area_mean'].max()), float(X['area_mean'].mean()))
smoothness_mean = st.sidebar.slider('Smoothness Mean', float(X['smoothness_mean'].min()), float(X['smoothness_mean'].max()), float(X['smoothness_mean'].mean()))
compactness_mean = st.sidebar.slider('Compactness Mean', float(X['compactness_mean'].min()), float(X['compactness_mean'].max()), float(X['compactness_mean'].mean()))
concavity_mean = st.sidebar.slider('Concavity Mean', float(X['concavity_mean'].min()), float(X['concavity_mean'].max()), float(X['concavity_mean'].mean()))
concave_points_mean = st.sidebar.slider('Concave Points Mean', float(X['concave points_mean'].min()), float(X['concave points_mean'].max()), float(X['concave points_mean'].mean()))
symmetry_mean = st.sidebar.slider('Symmetry Mean', float(X['symmetry_mean'].min()), float(X['symmetry_mean'].max()), float(X['symmetry_mean'].mean()))
fractal_dimension_mean = st.sidebar.slider('Fractal Dimension Mean', float(X['fractal_dimension_mean'].min()), float(X['fractal_dimension_mean'].max()), float(X['fractal_dimension_mean'].mean()))

# Preprocessing data pengguna
user_data = [[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean]]
user_data_scaled = scaler.transform(user_data)

# Prediksi dengan model KNN
prediction = knn_model.predict(user_data_scaled)

# Tampilkan hasil prediksi
st.subheader('Hasil Prediksi:')
if prediction[0] == 0:
    st.write('Sel tumor bersifat jinak (M)')
else:
    st.write('Sel tumor bersifat ganas (B)')
