#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 12:21:16 2024

@author: jorge
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from datetime import date, datetime, timedelta
from ta import momentum,  volatility  # Technical Analysis library
# from config import POLYGON_API_KEY
import requests

st.set_option('deprecation.showPyplotGlobalUse', False)

#Title
st.title("Gradient Boosting Model (GBM) Stock Price Predictor")

#Intro
st.write("Welcome! This lightweight application attempts to predict the future stock price of any stock using a GBM machine learning model. However, the model has it's limitations and should definitely NOT be construed as investment or financial advice.")
st.write("A more detailed explanation can be found on both the GitHub repository and on my personal portfolio website (jrgdlc.github.io/portfolio).")
st.write("This program aims to provide the user with as much control as possible over the model, highlighting the dynamism of this script. Therefore, there are three inputs that the user controls: the stock ticker, the start date for training the model, and the number of days to forecast. You should experiment playing around with these! Typically, we would expect that more data would mean a more powerful model, thus if you set a start date very far back the model should in theory be more accurate as it can observe more of the trends in the stock.")

# Using Polygon.io API (replace with your own key)
api_key = st.secrets["POLYGON_API_KEY"]

st.subheader("Inputs")

ticker = st.text_input("Enter the ticker symbol of the stock: ")
start_date = st.date_input("Enter the start date in YYYY-MM-DD format: ")
if start_date <= date.today() - timedelta(days=200):
  st.success("Start date is valid and at least 200 days ago.")
else:
  st.error("Error: Please enter a start date that is at least 100 days ago.")

n = int(st.slider("How far would you like to forecast (max 12):", 1, 12))
end_date = pd.to_datetime('today').strftime('%Y-%m-%d')  # Get today's date in YYYY-MM-DD format

st.subheader("Outputs")

def predict_stock_price():
        # Construct the URL for the API endpoint
    endpoint = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?apiKey={api_key}'
    
        # Send a GET request to the API
    response = requests.get(endpoint)
    
        # Check if the request was successful (status code 200)
    if response.status_code == 200:
            # Parse the JSON response
        data = response.json()
        results = data['results']
    
            # Transform the data into a DataFrame
        df = pd.DataFrame(results)
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')  # Convert timestamp to datetime
        df.set_index('timestamp', inplace=True)
    else:
            # Raise an exception for failed request
            raise Exception("Error fetching data:", response.status_code)
    
    # Function to calculate technical indicators
    def calculate_indicators(df):
        df['SMA_50'] = df['c'].rolling(window=50).mean()  # 50-day Simple Moving Average
        df['RSI_14'] = momentum.RSIIndicator(close=df['c'], window=14).rsi()  # 14-day Relative Strength Index
        bollinger_bands = volatility.BollingerBands(close=df['c'], window=20)
        df['Bollinger_Bands_20_2_Upper'] = bollinger_bands.bollinger_hband()  # Upper Bollinger Band (UBB)
        df['Bollinger_Bands_20_2_Lower'] = bollinger_bands.bollinger_lband()  # Lower Bollinger Band (LBB)
    
    calculate_indicators(df)
    
    st.write("Below is the dataframe used to train and test the model, including all the different variables. Typically, the model will perform better when given more data so the further back that you set the start date, the closer that the model will track the stock.")
    
    # Setting up for forecasting by moving columns so X_t-a
    features = ['o', 'h', 'l', 'SMA_50', 'RSI_14', 'Bollinger_Bands_20_2_Upper', 'Bollinger_Bands_20_2_Lower']
    
    num_cols = len(df.columns) - 1
    # Slice to extract 'c' and the remaining columns
    col_c = df['c']
    remaining_cols = df.drop('c', axis=1)
    
    # Create shifted DataFrame with 'c' leading by 5 steps
    df = pd.concat([col_c.shift(num_cols - (15+n)), remaining_cols], axis=1)
    
    original_shape = df.shape
    # Drop NaN values for the first 50 rows
    df.iloc[:50] = df.iloc[:50].dropna()
    #df = df[:50].drop_duplicates()
    df = df.fillna(df.rolling(5, min_periods=1).mean())
    df = df[50:]
    
    if df.isnull().any().any():
        print("Warning: Some NaN values could not be filled.")
        df.iloc[:original_shape[0], :]  # Return partially filled DataFrame
    else:
        df
        
    
    st.write("The most time-consuming and resource-intensive function is the hyperparameter tuning that needs to take place in order to best adjust the model and ensure a good performance.")
    
    # Split data into features (X) and target (y)
    X = df[features].iloc[:-n]  # Select features for training
    y = df['c'].iloc[:-n]  # Closing price as target variable
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [2, 3, 4],
        'subsample': [0.5, 0.75, 1.0]
    }

    # Create GBM model
    gbm = GradientBoostingRegressor()
    
    # Create GridSearchCV object
    grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid, cv=5)
    
    # Fit the grid search to training data
    grid_search.fit(X_train, y_train)
    
    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    
    print("Best Model Parameters:", best_params)
    print("Best Model:", best_model)
    
    gbm = best_model
    
    
    # Fit (train) the model on the training data
    model = gbm.fit(X_train, y_train)
    print(len(model.estimators_))
    
    score = model.score(X_test, y_test)
    print("Model R^2 Score:", score)
    
    
    a = len(y_test)
    
    X_test = df[features].iloc[-a:]
    
    y_pred = model.predict(X_test)
    print(y_pred)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print("Root Mean Squared Error:", rmse)
    
    df = df.shift((5+n), freq = 'D')
    X_test = X_test.shift((5+n), freq = 'D')
    
    # Get today's date
    today = datetime.today() + timedelta(days=n + 10)
    
    # Calculate the start date of the past week (adjust for Sunday or other start day if needed)
    past_month_start = today - pd.DateOffset(months=round((n/2)-1))
    
    # Display results using Streamlit
    st.write("Model R^2 Score:", score)
    st.write("Model Root Mean Squared Error (RMSE) Value:", rmse)
    
    st.write("Above we have the model's R^2 and RMSE value. These values are typically used as a reference for how well the model is performing and you can try playing around with the different inputs to see how they affect these values. As before, the more data the model has, the better we expect this score to be. So stocks that are traded frequently and have been around for a while will have a wealth of information that will train the model. If the R^2 is relatively low, or the RMSE is very high, then the forecasts should are generally more questionable as the model is essentially more 'unsure.'")
    
    y_flip = y_pred.reshape(1,-1)
    
    st.write("Predicted Closing Prices:", y_flip[:, -n:])
    
    st.subheader("Plot of Historical Data vs GBM Model Forecast")
    
    # Scatter plot of actual vs. predicted values
    plt.figure(figsize=(10, 6))
    plt.xlim(past_month_start, today)
    plt.plot(df.index, df['c'], label='Actual Closing Price')
    plt.plot(X_test.index, y_pred, label='Predicted Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title('Actual vs. Predicted Closing Prices')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot()

    st.write("Here we have the plot of our historical data (blue) and the predicted data (orange). Everything in orange is data that has the model has not yet seen, thus if it has been trained well it should track the historical data fairly closely. It is worth reiterating that this model should not be construed as financial advice and is more of a proof-of-concept.")

if st.button("Predict Stock Price"):
  predict_stock_price()

st.subheader("About this Project")
st.write("This project was my first venture into using a machine learning model with hyperparameter tuning. The goal for the project was to emphasize user interaction through a simple front-end as my other projects have instead been detailed reports on the statistical methods and diagnostics used. Therefore, this project uses streamlit as a frontend, numpy and pandas for data manipulation and preprocessing, technical analysis (ta) for feature engineering, and scikit-learn for building the GradientBoostingRegressor Model and for hyperparameter tuning. Finally, the plot uses matplotlib. Aside from Python packages, the project also integrates an API to access the stock information from Polygon.ai.")
