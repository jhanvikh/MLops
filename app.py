import streamlit as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target
model = LogisticRegression(max_iter=200).fit(X, y)

st.title("Iris Prediction")

inputs = [st.slider(label, min_value=val[0], max_value=val[1], value=val[2]) for label, val in zip(
    ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
    [(4.0, 8.0, 5.1), (2.0, 4.5, 3.5), (1.0, 7.0, 1.4), (0.1, 2.5, 0.2)]
)]

if st.button('Predict'):
    result = model.predict([inputs])[0]
    st.success(f"Prediction: {iris.target_names[result]}")
