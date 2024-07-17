import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Path to the trained models
price_model_path = 'predict_close_price_model.pkl'
volume_model_path = 'predict_volume_model.pkl'

# Load the trained models
price_model = joblib.load(price_model_path)
volume_model = joblib.load(volume_model_path)

st.image('reliance_logo.png',width=400)

# Streamlit application title
st.title('Close Price and Volume Prediction from 2024 to 2029')

# Function to predict future values
def predict_future_values(start_date, end_date):
    # Create future dates from start_date to end_date on a daily basis
    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    future_dates_num = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)

    # Predict future values
    future_pred = price_model.predict(future_dates_num)

    # Create a DataFrame for the results
    prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_pred})
    prediction_df.set_index('Date', inplace=True)
    return prediction_df

# Function to predict future volume values
def predict_future_volumes(start_date, end_date):
    # Create future dates from start_date to end_date on a daily basis
    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    future_dates_num = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)

    # Predict future values
    future_pred = volume_model.predict(future_dates_num)

    # Create a DataFrame for the results
    prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted_Volume': future_pred})
    prediction_df.set_index('Date', inplace=True)
    return prediction_df


# Function to predict the closing price for a specific date
def predict_price(date):
    date_num = pd.Timestamp(date).toordinal()
    prediction = price_model.predict(np.array([[date_num]]))
    return prediction[0]

# Function to predict the volume for a specific date
def predict_volume(date):
    date_num = pd.Timestamp(date).toordinal()
    prediction = volume_model.predict(np.array([[date_num]]))
    return prediction[0]

# Layout with two columns
col1, col2 = st.columns(2)

with col1:
    st.image('close_price.png')
    st.header("Close Price Predictor")
    # Input field for a specific date for stock price prediction
    price_date = st.date_input('Select a date for stock price prediction', pd.to_datetime('2024-01-01'))
    if st.button('Predict Stock Price', key='price'):
        price_prediction = predict_price(price_date)
        st.write(f"The predicted closing price for {price_date} is: {price_prediction:.2f}")
    st.header('Select Date range')
    # Input fields for start and end dates
    start_date= st.date_input('Start Date', pd.to_datetime('2024-01-01'))
    end_date = st.date_input('End Date', pd.to_datetime('2029-12-31'))
    # Predict button
    if st.button('Predict'):
        if start_date < end_date:
            predictions = predict_future_values(start_date, end_date)
            st.write(predictions)

            # Plotting the results
            st.line_chart(predictions)
        else:
            st.error('Error: End date must fall after start date.')


with col2:
    st.image('volume.png')
    st.header("Volume Predictor")
    # Input field for a specific date for volume prediction
    volume_date = st.date_input('Select a date for volume prediction', pd.to_datetime('2024-01-01'))
    if st.button('Predict Volume', key='volume'):
        volume_prediction = predict_volume(volume_date)
        st.write(f"The predicted volume for {volume_date} is: {volume_prediction:.2f}")
    st.header('Select Date range')
    # Input fields for start and end dates
    start_date = st.date_input('Start date', pd.to_datetime('2024-01-01'))
    end_date = st.date_input('End date', pd.to_datetime('2029-12-31'))
    if st.button('predict'):
        if start_date < end_date:
            predictions = predict_future_volumes(start_date, end_date)
            st.write(predictions)

            # Plotting the results
            st.line_chart(predictions)
        else:
            st.error('Error: End date must fall after start date.')

