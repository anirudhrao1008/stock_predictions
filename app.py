import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("matplotlib")
install("pandas")
install("numpy")
install("streamlit")
install("yfinance")
install("pandas-datareader")
install("keras")
install("tensorflow")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler


def apply_custom_css():
    st.markdown(
        """
        <style>
        body {
            background-image: url('https://images.pexels.com/photos/952670/pexels-photo-952670.jpeg');
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }
        .stApp {
            background: transparent; 
        }
        .sidebar .sidebar-content {
            background-color: rgba(255, 255, 255, 0.9); 
        }
        .sidebar .sidebar-content > * {
            margin: 0;
        }
        .stButton > button {
            background-color: #f0f0f0;
            color: #333;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            transition: background-color 0.3s, color 0.3s;
            padding: 10px 20px;
            font-size: 16px;
            width: 100%;
        }
        .stButton > button:hover {
            background-color: #e0e0e0;
            color: #000;
        }
        .stButton > button:active {
            background-color: #d0d0d0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


apply_custom_css()


start = '2010-01-01'
end = '2023-12-31'


st.sidebar.title('Navigation')
home_button = st.sidebar.button('Home')
data_overview_button = st.sidebar.button('Data Overview')
closing_price_charts_button = st.sidebar.button('Closing Price Charts')
ma_charts_button = st.sidebar.button('MA Charts')
model_prediction_button = st.sidebar.button('Model Prediction')


if 'page' not in st.session_state:
    st.session_state.page = 'Home'

if home_button:
    st.session_state.page = 'Home'
elif data_overview_button:
    st.session_state.page = 'Data Overview'
elif closing_price_charts_button:
    st.session_state.page = 'Closing Price Charts'
elif ma_charts_button:
    st.session_state.page = 'MA Charts'
elif model_prediction_button:
    st.session_state.page = 'Model Prediction'


if st.session_state.page == 'Home':
    st.title('Stock Trend Prediction')
    st.subheader('Select Company Stock')
    with st.container():
        st.markdown("<div class='centered'>", unsafe_allow_html=True)
        user_input = st.text_input('Enter stock ticker (e.g., SBIN.NS)', key='stock_ticker')
        st.markdown("</div>", unsafe_allow_html=True)
    if user_input:
        st.session_state.user_input = user_input

else:
    
    user_input = st.session_state.get('user_input', 'SBIN.NS')

    if not user_input:
        st.warning("Please go to the Home page and select a company stock.")
    else:
        df = yf.download(user_input, start=start, end=end)
        st.write(df.head())

        if st.session_state.page == 'Data Overview':
            st.subheader('Data from 2010-2023')
            st.write(df.describe(include='all'))

        elif st.session_state.page == 'Closing Price Charts':
            # Visualization of closing price
            st.subheader('Closing Price vs Time Chart')
            fig = plt.figure(figsize=(12, 6))
            plt.plot(df.Close)
            plt.title('Closing Price vs Time')
            plt.xlabel('Date')
            plt.ylabel('Closing Price')
            st.pyplot(fig)

        elif st.session_state.page == 'MA Charts':
            # Visualization of closing price with 100MA
            st.subheader('Closing Price vs Time Chart with 100MA')
            ma100 = df.Close.rolling(100).mean()
            fig = plt.figure(figsize=(12, 6))
            plt.plot(df.Close, label='Closing Price')
            plt.plot(ma100, label='100-Day MA')
            plt.legend()
            st.pyplot(fig)

            # Visualization of closing price with 100MA and 200MA
            st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
            ma100 = df.Close.rolling(100).mean()
            ma200 = df.Close.rolling(200).mean()
            fig = plt.figure(figsize=(12, 6))
            plt.plot(df.Close, label='Closing Price')
            plt.plot(ma100, label='100-Day MA')
            plt.plot(ma200, label='200-Day MA')
            plt.legend()
            st.pyplot(fig)

        elif st.session_state.page == 'Model Prediction':
            # Data preparation
            data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
            data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

            scaler = MinMaxScaler(feature_range=(0, 1))
            data_training_array = scaler.fit_transform(data_training)

            # Load the model
            model = load_model('my_model.keras')

            # Testing
            past_100_days = data_training.tail(100)
            final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
            input_data = scaler.fit_transform(final_df)

            x_test = []
            y_test = []

            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i-100:i])
                y_test.append(input_data[i, 0])

            x_test, y_test = np.array(x_test), np.array(y_test)

            y_predicted = model.predict(x_test)
            scaler = scaler.scale_

            scale_factor = 1 / scaler[0]
            y_predicted = y_predicted * scale_factor
            y_test = y_test * scale_factor

            # Final graph
            st.subheader('Predictions vs Original')
            fig2 = plt.figure(figsize=(12, 6))
            plt.plot(y_test, 'b', label='Original Price')
            plt.plot(y_predicted, 'r', label='Predicted Price')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(fig2)
