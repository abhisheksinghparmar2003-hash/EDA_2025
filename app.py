import streamlit as st
import joblib 
import sklearn
import numpy as np

st.set_page_config(page_title="Stock Price Predictor 🍷", page_icon="🍷", layout="centered")


st.markdown("<h1 style='text-align: center; color: purple;'>STOCK PRICE PREDICTION</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:2px solid purple;'>", unsafe_allow_html=True)

model = joblib.load('stock_price.joblib')

cols=['Open',
 'High',
 'Low',
 'Volume',
 'pct_return_lag1',
 'log_return_lag1',
 'hl_range_lag1',
 'oc_change_lag1',
 'volume_change_lag1',
 'MA10_lag1',
 'MA30_lag1']

input_cols = st.columns(2)
values = []


for idx, feature in enumerate(cols):
    with input_cols[idx % 2]: 
        val = st.number_input(f'Enter {feature}', format="%.4f")
        values.append(val)

if st.button(' Predict Stock Price'):
    lr_model = model.predict([values])
    st.success(f"Predicted Stock Price Score: **{lr_model[0]}**")

st.markdown("<hr style='border:1px solid purple;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Model powered by Streamlit & scikit-learn</p>", unsafe_allow_html=True)

          