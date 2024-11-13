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
from tensorflow.keras.models import load_model
import pickle

# Stock symbols
stock_symbols = ["VCB", "VNM", "MWG", "VIC", "SSI", "DGC", "CTD", "FPT", "MSN",
                 "GVR", "GAS", "POW", "HPG", "REE", "DHG", "GMD", "VHC", "KBC",
                 "CMG", "VRE"]

# Directory to store the models
storage_dir = "Model_storage"
os.makedirs(storage_dir, exist_ok=True)

# Function to download files
def download_file(file_id, output_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)

# Function to unzip files
def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Function to load LSTM models from directory
def load_model_data(model_type):
    model_dir = os.path.join(storage_dir, model_type)
    model_dict = {}

    for file_name in os.listdir(model_dir):
        stock_symbol = file_name.split("_")[-1].split(".")[0]
        file_path = os.path.join(model_dir, file_name)
        
        # Load as Keras model (.h5)
        if file_name.endswith(".h5"):
            try:
                model_dict[stock_symbol] = load_model(file_path)
            except Exception as e:
                st.warning(f"Failed to load Keras model for {stock_symbol}: {e}")
                
    return model_dict

# Download, unzip, and load LSTM models
def setup_lstm_models():
    lstm_file_id = '10eo2UHTAsrYAYkK_ol4jGTfAm_ksv3Vx'
    lstm_zip_path = os.path.join(storage_dir, 'LSTM_models.zip')
    if not os.path.exists(os.path.join(storage_dir, 'LSTM')):
        try:
            download_file(lstm_file_id, lstm_zip_path)
            unzip_file(lstm_zip_path, os.path.join(storage_dir, 'LSTM'))
        except Exception as e:
            st.warning(f"Could not download or unzip model: {e}")
            return {}
    return load_model_data("LSTM")

# Load all LSTM models
lstm_models = setup_lstm_models()

# Display technical indicators and predictions
def plot_technical_indicators(stock_symbol):
    # [Omitted code for brevity; use original technical indicators code here]
    pass

def display_prediction_chart(stock_symbol, model_data, model_name):
    # [Omitted code for brevity; use original prediction chart code here]
    pass

# Streamlit App
st.title("Stock Price Prediction and Technical Analysis")

# Sidebar for selecting the stock symbol
stock_symbol = st.sidebar.selectbox("Choose a Stock Symbol", stock_symbols)

# Show technical indicators
if st.sidebar.button("Show Technical Indicators"):
    plot_technical_indicators(stock_symbol)

# Show LSTM prediction results only
if st.sidebar.button("Show Prediction Results"):
    if stock_symbol in lstm_models:
        display_prediction_chart(stock_symbol, lstm_models[stock_symbol], "LSTM")
