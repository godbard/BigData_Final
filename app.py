import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objs as go
import plotly.subplots as sp
from datetime import datetime, timedelta
from vnstock import stock_historical_data
import importlib.util
import sys
import gdown
import zipfile
import pickle



# Stock symbols
stock_symbols = ["VCB", "VNM", "MWG", "VIC", "SSI", "DGC", "CTD", "FPT", "MSN",
                 "GVR", "GAS", "POW", "HPG", "REE", "DHG", "GMD", "VHC", "KBC",
                 "CMG", "VRE"]


# Directory to store the models
storage_dir = "Model_Storage"
os.makedirs(storage_dir, exist_ok=True)


# Function to download files
def download_file(file_id, output_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)


# Function to unzip files
def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


# Function to load models from directory
def load_model_data(model_type):
    model_dir = os.path.join(storage_dir, model_type)
    model_dict = {}


    for file_name in os.listdir(model_dir):
        if file_name.endswith(".pkl"):
            stock_symbol = file_name.split("_")[-1].replace(".pkl", "")
            file_path = os.path.join(model_dir, file_name)
            with open(file_path, 'rb') as file:
                model_dict[stock_symbol] = pickle.load(file)
    return model_dict


# Download, unzip, and load models
def setup_models():
    def download_and_unzip(file_id, zip_path, extract_to):
        # Download the file if directory doesn't exist
        if not os.path.exists(extract_to):
            try:
                download_file(file_id, zip_path)
                unzip_file(zip_path, extract_to)
            except Exception as e:
                st.warning(f"Could not download or unzip model: {e}")
                return False
        return True


    # LSTM models
    lstm_file_id = '10O8z6Y8B2sHvv5AA0aRzeVJm1aQU0VBG'
    lstm_zip_path = os.path.join(storage_dir, 'LSTM_models.zip')
    if download_and_unzip(lstm_file_id, lstm_zip_path, os.path.join(storage_dir, 'LSTM')):
        lstm_models = load_model_data("LSTM")
    else:
        lstm_models = {}


    # GRU models
    gru_file_id = '11u02kLudVUF1KBqXzs7MoH4gjNNxilSe'
    gru_zip_path = os.path.join(storage_dir, 'GRU_models.zip')
    if download_and_unzip(gru_file_id, gru_zip_path, os.path.join(storage_dir, 'GRU')):
        gru_models = load_model_data("GRU")
    else:
        gru_models = {}


    # CNN models
    cnn_file_id = '11rJoOtu97g0gqzUalM-DSC-ChG-agAf6'
    cnn_zip_path = os.path.join(storage_dir, 'CNN_models.zip')
    if download_and_unzip(cnn_file_id, cnn_zip_path, os.path.join(storage_dir, 'CNN')):
        cnn_models = load_model_data("CNN")
    else:
        cnn_models = {}


    return lstm_models, gru_models, cnn_models




# Load all models
lstm_models, gru_models, cnn_models = setup_models()


# Display the technical indicators and predictions for a stock symbol
def plot_technical_indicators(stock_symbol):
    today = datetime.today()
    one_year_ago = today - timedelta(days=1461)


    # Fetch data from API
    data = stock_historical_data(
        stock_symbol,
        start_date=one_year_ago.strftime('%Y-%m-%d'),
        end_date=today.strftime('%Y-%m-%d')
    )


    # Check if data is empty
    if data.empty:
        st.warning(f"No data available for {stock_symbol}.")
        return


    # Prepare data
    df = pd.DataFrame(data)[['time', 'open', 'close', 'high', 'low', 'volume']]
    df = df.rename(columns={'time': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')


    # Convert date to string format to use categorical axis
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')


    # Calculate technical indicators
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['MA100'] = df['close'].rolling(window=100).mean()
    df['BB_upper'] = df['close'].rolling(window=20).mean() + 2 * df['close'].rolling(window=20).std()
    df['BB_lower'] = df['close'].rolling(window=20).mean() - 2 * df['close'].rolling(window=20).std()
    df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()


    # RSI calculation
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))


    # Create subplots
    fig = sp.make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.05,
        specs=[[{"secondary_y": True}], [{}], [{}]]
    )


    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['date_str'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='Candlestick'
    ), row=1, col=1, secondary_y=False)


    # Volume bar chart on secondary y-axis (y2)
    fig.add_trace(go.Bar(
        x=df['date_str'], y=df['volume'],
        marker_color='blue', opacity=0.7, name='Volume'
    ), row=1, col=1, secondary_y=True)


    # MA50 and MA100
    fig.add_trace(go.Scatter(x=df['date_str'], y=df['MA50'], mode='lines', name='MA50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date_str'], y=df['MA100'], mode='lines', name='MA100'), row=1, col=1)


    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df['date_str'], y=df['BB_upper'], line=dict(color='lightgray'), name='BB Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date_str'], y=df['BB_lower'], line=dict(color='lightgray'), name='BB Lower'), row=1, col=1)


    # MACD and Signal line
    fig.add_trace(go.Scatter(x=df['date_str'], y=df['MACD'], mode='lines', name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['date_str'], y=df['MACD_Signal'], mode='lines', name='MACD Signal'), row=2, col=1)


    # RSI
    fig.add_trace(go.Scatter(x=df['date_str'], y=df['RSI'], mode='lines', name='RSI'), row=3, col=1)
    fig.add_shape(type='line', x0=df['date_str'].min(), x1=df['date_str'].max(), y0=70, y1=70,
                  line=dict(color='red', dash='dash'), row=3, col=1)
    fig.add_shape(type='line', x0=df['date_str'].min(), x1=df['date_str'].max(), y0=30, y1=30,
                  line=dict(color='green', dash='dash'), row=3, col=1)


    # Update layout to remove range slider, add unified hover, and configure y-axes
    fig.update_layout(
        title=f'{stock_symbol} - Technical Indicators and Volume',
        template='plotly_white',
        height=1000,
        hovermode='x unified',
        xaxis=dict(
            type='category',
            tickangle=45,
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(title='Price', side='left'),
        yaxis2=dict(title='Volume', overlaying='y', side='right')
    )


    st.plotly_chart(fig)


def display_prediction_chart(stock_symbol, model_data, model_name):
    y_test = model_data['y_test']
    y_pred = model_data['y_pred']
    dates = model_data['dates']


    if len(y_test) == 0 or len(y_pred) == 0:
        st.warning("No prediction data available.")
        return


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_test, mode='lines', name='Actual Price', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=dates, y=y_pred, mode='lines', name=f'Predicted Price ({model_name})', line=dict(color='orange')))
    fig.update_layout(
        title=f"Stock Price Prediction for {stock_symbol} using {model_name}",
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white'
    )
    st.plotly_chart(fig)


    # Show performance metrics
    st.write(f"### Performance Metrics for {model_name}")
    st.write(f"RMSE: {model_data['rmse']:.2f}")
    st.write(f"MAE: {model_data['mae']:.2f}")
    st.write(f"R-squared: {model_data['r_squared']:.2f}")
    st.write(f"MAPE: {model_data['mape']:.2f}%")


# Streamlit App
st.title("Stock Price Prediction and Technical Analysis")


# Sidebar for selecting the stock symbol
stock_symbol = st.sidebar.selectbox("Choose a Stock Symbol", stock_symbols)


# Show technical indicators
if st.sidebar.button("Show Technical Indicators"):
    plot_technical_indicators(stock_symbol)


# Show prediction results
if st.sidebar.button("Show Prediction Results"):
    if stock_symbol in lstm_models:
        display_prediction_chart(stock_symbol, lstm_models[stock_symbol], "LSTM")
    if stock_symbol in gru_models:
        display_prediction_chart(stock_symbol, gru_models[stock_symbol], "GRU")
    if stock_symbol in cnn_models:
        display_prediction_chart(stock_symbol, cnn_models[stock_symbol], "CNN")



