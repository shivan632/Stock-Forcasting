import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import math
import warnings
from datetime import datetime
import io
import sys
import os

# Add the current virtual environment's site-packages to path
venv_path = os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages')
if os.path.exists(venv_path) and venv_path not in sys.path:
    sys.path.insert(0, venv_path)

# Try importing tensorflow with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Stock Time Series Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Time Series Forecasting for Stock Prices")
st.markdown("""
This dashboard implements **ARIMA** and **LSTM** models for stock price forecasting.
Upload your stock data CSV file to begin analysis.
""")

# Upload CSV
uploaded_file = st.file_uploader("Upload Stock CSV File", type=["csv"])

if uploaded_file:
    
    # Try to read CSV with proper parsing
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check if data needs cleaning (based on the malformed sample)
        if df.shape[1] < 7:  # If columns are missing
            st.warning("CSV format seems incorrect. Attempting to fix...")
            
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Read raw lines and clean
            lines = uploaded_file.read().decode('utf-8').split('\n')
            cleaned_lines = []
            
            for line in lines:
                if line.strip() and not line.startswith('Col'):  # Skip invalid lines
                    # Split by comma and clean
                    parts = line.split(',')
                    if len(parts) >= 7:  # If we have enough parts
                        # Clean the date part if it's malformed
                        date_part = parts[0]
                        if '2013-02-14149414.96' in date_part:
                            parts[0] = '2013-02-14'
                        cleaned_lines.append(','.join(parts[:7]))  # Take first 7 columns
            
            # Create new CSV string
            from io import StringIO
            cleaned_csv = '\n'.join(cleaned_lines)
            
            # Read the cleaned CSV
            df = pd.read_csv(StringIO(cleaned_csv))
        
        st.success("✅ Data loaded successfully!")
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Data preprocessing
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Convert date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df.sort_values('date', inplace=True)
        df.set_index('date', inplace=True)
    else:
        st.error("Date column not found in the dataset")
        st.stop()
    
    # Check for required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume', 'name']
    for col in required_cols:
        if col not in df.columns:
            st.warning(f"Column '{col}' not found. Some features may be limited.")
    
    # Get unique stock names
    if 'name' in df.columns:
        stock_names = df['name'].unique()
        if len(stock_names) > 0:
            stock = st.selectbox("Select Stock Name", stock_names)
            stock_df = df[df['name'] == stock].copy()
        else:
            stock = "Unknown"
            stock_df = df.copy()
    else:
        stock = "Unknown"
        stock_df = df.copy()
    
    # Ensure we have enough data
    if len(stock_df) < 50:
        st.warning(f"Limited data points: {len(stock_df)}. Forecasting may be unreliable.")
    
    close_data = stock_df['close']
    
    # Sidebar for model parameters
    st.sidebar.header("⚙️ Model Parameters")
    
    # ARIMA parameters
    st.sidebar.subheader("ARIMA Parameters")
    arima_p = st.sidebar.slider("ARIMA p (AR order)", 0, 5, 5)
    arima_d = st.sidebar.slider("ARIMA d (Differencing)", 0, 2, 1)
    arima_q = st.sidebar.slider("ARIMA q (MA order)", 0, 5, 0)
    
    # LSTM parameters (only if TensorFlow is available)
    if TENSORFLOW_AVAILABLE:
        st.sidebar.subheader("LSTM Parameters")
        lstm_epochs = st.sidebar.slider("LSTM Epochs", 5, 50, 20)
        lstm_batch_size = st.sidebar.selectbox("LSTM Batch Size", [16, 32, 64], index=1)
        lstm_lookback = st.sidebar.slider("LSTM Lookback (days)", 30, 100, 60)
    
    # Train-Test split
    train_ratio = st.sidebar.slider("Train-Test Split Ratio", 0.6, 0.9, 0.8)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Overview",
        "📈 ARIMA Model",
        "🤖 LSTM Model" if TENSORFLOW_AVAILABLE else "🤖 LSTM (Disabled)",
        "📑 Comparison & Report"
    ])
    
    with tab1:
        st.header("Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Date Range", f"{stock_df.index.min().date()} to {stock_df.index.max().date()}")
        with col2:
            st.metric("Total Days", len(stock_df))
        with col3:
            st.metric("Avg Close Price", f"${stock_df['close'].mean():.2f}")
        with col4:
            st.metric("Price Range", f"${stock_df['close'].min():.2f} - ${stock_df['close'].max():.2f}")
        
        st.subheader("Raw Data Sample")
        st.dataframe(stock_df.head(10))
        
        st.subheader("Closing Price History")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(stock_df.index, stock_df['close'], color='blue', linewidth=1)
        ax.set_xlabel('Date')
        ax.set_ylabel('Closing Price ($)')
        ax.set_title(f'{stock} - Closing Price History')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Basic statistics
        if all(col in stock_df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            st.subheader("Descriptive Statistics")
            st.dataframe(stock_df[['open', 'high', 'low', 'close', 'volume']].describe().round(2))
        
        # Stationarity Test
        st.subheader("🔎 Augmented Dickey-Fuller Test (Stationarity)")
        
        result = adfuller(close_data.dropna())
        adf_stat = result[0]
        p_value = result[1]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ADF Statistic", f"{adf_stat:.4f}")
        with col2:
            st.metric("P-value", f"{p_value:.4f}")
        
        if p_value < 0.05:
            stationarity_msg = "✅ Data is Stationary (suitable for ARIMA without differencing)"
            st.success(stationarity_msg)
        else:
            stationarity_msg = "⚠️ Data is NOT Stationary (ARIMA will need differencing)"
            st.error(stationarity_msg)
        
        # ACF and PACF plots
        st.subheader("Autocorrelation Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_acf(close_data.dropna(), lags=20, ax=ax)
            ax.set_title('Autocorrelation Function (ACF)')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_pacf(close_data.dropna(), lags=20, ax=ax)
            ax.set_title('Partial Autocorrelation Function (PACF)')
            st.pyplot(fig)
    
    # Train Test Split
    train_size = int(len(close_data) * train_ratio)
    train, test = close_data[:train_size].dropna(), close_data[train_size:].dropna()
    
    # Store in session state for other tabs
    st.session_state['train'] = train
    st.session_state['test'] = test
    st.session_state['stock'] = stock
    st.session_state['close_data'] = close_data
    st.session_state['adf_stat'] = adf_stat
    st.session_state['p_value'] = p_value
    st.session_state['stationarity_msg'] = stationarity_msg
    
    with tab2:
        st.header("ARIMA Model Analysis")
        
        if len(train) < 30:
            st.error("Insufficient training data for ARIMA model")
        else:
            with st.spinner("Training ARIMA model..."):
                try:
                    # Fit ARIMA model
                    model_arima = ARIMA(train, order=(arima_p, arima_d, arima_q))
                    model_arima_fit = model_arima.fit()
                    
                    # Make predictions
                    arima_pred = model_arima_fit.forecast(steps=len(test))
                    
                    # Calculate metrics
                    arima_mae = mean_absolute_error(test, arima_pred)
                    arima_rmse = math.sqrt(mean_squared_error(test, arima_pred))
                    arima_mape = np.mean(np.abs((test.values - arima_pred) / test.values)) * 100
                    
                    # Store in session state
                    st.session_state['arima_pred'] = arima_pred
                    st.session_state['arima_mae'] = arima_mae
                    st.session_state['arima_rmse'] = arima_rmse
                    st.session_state['arima_mape'] = arima_mape
                    
                    # Display model summary
                    st.subheader("Model Summary")
                    st.text(str(model_arima_fit.summary()))
                    
                    # Plot ARIMA results
                    st.subheader("ARIMA Forecast vs Actual")
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(train.index, train, label='Training Data', color='blue', alpha=0.5)
                    ax.plot(test.index, test, label='Actual Test Data', color='green')
                    ax.plot(test.index, arima_pred, label='ARIMA Forecast', color='red', linestyle='--')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Closing Price ($)')
                    ax.set_title(f'{stock} - ARIMA({arima_p},{arima_d},{arima_q}) Forecast')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Performance metrics
                    st.subheader("ARIMA Performance Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MAE", f"${arima_mae:.2f}")
                    with col2:
                        st.metric("RMSE", f"${arima_rmse:.2f}")
                    with col3:
                        st.metric("MAPE", f"{arima_mape:.2f}%")
                    
                    # Residual analysis
                    st.subheader("Residual Analysis")
                    
                    residuals = test.values - arima_pred
                    
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Residuals over time
                    axes[0].plot(test.index, residuals)
                    axes[0].axhline(y=0, color='red', linestyle='--')
                    axes[0].set_xlabel('Date')
                    axes[0].set_ylabel('Residuals')
                    axes[0].set_title('Residuals Over Time')
                    axes[0].grid(True, alpha=0.3)
                    
                    # Residuals histogram
                    axes[1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
                    axes[1].axvline(x=0, color='red', linestyle='--')
                    axes[1].set_xlabel('Residuals')
                    axes[1].set_ylabel('Frequency')
                    axes[1].set_title('Residuals Distribution')
                    axes[1].grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error training ARIMA model: {e}")
    
    with tab3:
        if TENSORFLOW_AVAILABLE:
            st.header("LSTM Model Analysis")
            
            if len(train) < lstm_lookback + 10:
                st.error(f"Insufficient data for LSTM. Need at least {lstm_lookback + 10} data points.")
            else:
                with st.spinner("Training LSTM model... (this may take a moment)"):
                    try:
                        # Prepare data for LSTM
                        scaler = MinMaxScaler()
                        
                        # Scale the entire dataset
                        all_data = close_data.values.reshape(-1, 1)
                        scaled_data = scaler.fit_transform(all_data)
                        
                        # Create sequences
                        X, y = [], []
                        for i in range(lstm_lookback, len(scaled_data)):
                            X.append(scaled_data[i-lstm_lookback:i, 0])
                            y.append(scaled_data[i, 0])
                        
                        X, y = np.array(X), np.array(y)
                        
                        # Reshape X for LSTM [samples, time steps, features]
                        X = X.reshape(X.shape[0], X.shape[1], 1)
                        
                        # Split into train and test
                        split_idx = int(len(X) * train_ratio)
                        X_train, X_test = X[:split_idx], X[split_idx:]
                        y_train, y_test = y[:split_idx], y[split_idx:]
                        
                        # Build LSTM model
                        model_lstm = Sequential()
                        model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
                        model_lstm.add(Dropout(0.2))
                        model_lstm.add(LSTM(50, return_sequences=False))
                        model_lstm.add(Dropout(0.2))
                        model_lstm.add(Dense(25))
                        model_lstm.add(Dense(1))
                        
                        model_lstm.compile(optimizer='adam', loss='mse')
                        
                        # Early stopping
                        early_stop = EarlyStopping(monitor='loss', patience=3, verbose=0)
                        
                        # Train model
                        history = model_lstm.fit(
                            X_train, y_train,
                            epochs=lstm_epochs,
                            batch_size=lstm_batch_size,
                            validation_data=(X_test, y_test),
                            callbacks=[early_stop],
                            verbose=0
                        )
                        
                        # Make predictions
                        lstm_train_pred = model_lstm.predict(X_train, verbose=0)
                        lstm_test_pred = model_lstm.predict(X_test, verbose=0)
                        
                        # Inverse transform predictions
                        lstm_train_pred = scaler.inverse_transform(lstm_train_pred)
                        lstm_test_pred = scaler.inverse_transform(lstm_test_pred)
                        
                        # Get actual values for test period
                        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                        
                        # Calculate metrics
                        lstm_mae = mean_absolute_error(y_test_actual, lstm_test_pred)
                        lstm_rmse = math.sqrt(mean_squared_error(y_test_actual, lstm_test_pred))
                        lstm_mape = np.mean(np.abs((y_test_actual - lstm_test_pred) / y_test_actual)) * 100
                        
                        # Store in session state
                        st.session_state['lstm_pred'] = lstm_test_pred
                        st.session_state['lstm_mae'] = lstm_mae
                        st.session_state['lstm_rmse'] = lstm_rmse
                        st.session_state['lstm_mape'] = lstm_mape
                        st.session_state['y_test_actual'] = y_test_actual
                        st.session_state['lstm_lookback'] = lstm_lookback
                        st.session_state['lstm_split_idx'] = split_idx
                        
                        # Plot training history
                        st.subheader("LSTM Training History")
                        
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(history.history['loss'], label='Training Loss')
                        ax.plot(history.history['val_loss'], label='Validation Loss')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Loss')
                        ax.set_title('Model Loss Over Epochs')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
                        # Plot LSTM results
                        st.subheader("LSTM Forecast vs Actual")
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Plot test data with proper indices
                        test_indices = close_data.index[split_idx + lstm_lookback:]
                        
                        ax.plot(test_indices, y_test_actual, label='Actual Test Data', color='green')
                        ax.plot(test_indices, lstm_test_pred, label='LSTM Forecast', color='red', linestyle='--')
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Closing Price ($)')
                        ax.set_title(f'{stock} - LSTM Forecast')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
                        # Performance metrics
                        st.subheader("LSTM Performance Metrics")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("MAE", f"${lstm_mae:.2f}")
                        with col2:
                            st.metric("RMSE", f"${lstm_rmse:.2f}")
                        with col3:
                            st.metric("MAPE", f"{lstm_mape:.2f}%")
                        
                    except Exception as e:
                        st.error(f"Error training LSTM model: {e}")
        else:
            st.warning("⚠️ LSTM model is disabled because TensorFlow is not installed.")
            st.info("To enable LSTM, install TensorFlow with: `pip install tensorflow`")
            st.code("pip install tensorflow", language="bash")
    
    with tab4:
        st.header("Model Comparison & Analysis Report")
        
        # Check if models have been trained
        arima_trained = 'arima_rmse' in st.session_state
        lstm_trained = 'lstm_rmse' in st.session_state if TENSORFLOW_AVAILABLE else False
        
        if arima_trained:
            st.subheader("📈 ARIMA Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE", f"${st.session_state['arima_mae']:.2f}")
            with col2:
                st.metric("RMSE", f"${st.session_state['arima_rmse']:.2f}")
            with col3:
                st.metric("MAPE", f"{st.session_state['arima_mape']:.2f}%")
        
        if lstm_trained:
            st.subheader("🤖 LSTM Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE", f"${st.session_state['lstm_mae']:.2f}")
            with col2:
                st.metric("RMSE", f"${st.session_state['lstm_rmse']:.2f}")
            with col3:
                st.metric("MAPE", f"{st.session_state['lstm_mape']:.2f}%")
        
        if arima_trained and lstm_trained:
            # Comparison table
            st.subheader("📊 Model Performance Comparison")
            
            comparison_df = pd.DataFrame({
                "Model": ["ARIMA", "LSTM"],
                "MAE ($)": [f"{st.session_state['arima_mae']:.2f}", f"{st.session_state['lstm_mae']:.2f}"],
                "RMSE ($)": [f"{st.session_state['arima_rmse']:.2f}", f"{st.session_state['lstm_rmse']:.2f}"],
                "MAPE (%)": [f"{st.session_state['arima_mape']:.2f}", f"{st.session_state['lstm_mape']:.2f}"]
            })
            
            st.dataframe(comparison_df, use_container_width='stretch')
            
            # Determine best model
            if st.session_state['lstm_rmse'] < st.session_state['arima_rmse']:
                best_model = "LSTM"
                improvement = ((st.session_state['arima_rmse'] - st.session_state['lstm_rmse']) / st.session_state['arima_rmse']) * 100
                st.success(f"✅ **{best_model} performs better** with {improvement:.1f}% lower RMSE than ARIMA")
            else:
                best_model = "ARIMA"
                improvement = ((st.session_state['lstm_rmse'] - st.session_state['arima_rmse']) / st.session_state['lstm_rmse']) * 100
                st.success(f"✅ **{best_model} performs better** with {improvement:.1f}% lower RMSE than LSTM")
            
            # Combined plot - FIXED VERSION
            st.subheader("Combined Forecast Comparison")
            
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # Plot training data (last 100 points for clarity)
            train_data = st.session_state['train']
            ax.plot(train_data.index[-100:], train_data.values[-100:], 
                   label='Training Data', color='blue', alpha=0.5)
            
            # Plot test data
            test_data = st.session_state['test']
            ax.plot(test_data.index, test_data.values, label='Actual Test', color='green', linewidth=2)
            
            # Plot ARIMA predictions
            if 'arima_pred' in st.session_state:
                ax.plot(test_data.index, st.session_state['arima_pred'], 
                       label='ARIMA Forecast', color='red', linestyle='--', linewidth=2)
            
            # Plot LSTM predictions - FIXED
            if 'lstm_pred' in st.session_state:
                # Calculate the correct indices for LSTM predictions
                lstm_lookback = st.session_state.get('lstm_lookback', 60)
                split_idx = st.session_state.get('lstm_split_idx', 0)
                
                # LSTM predictions start after lookback period and training split
                lstm_start_idx = split_idx + lstm_lookback
                lstm_end_idx = lstm_start_idx + len(st.session_state['lstm_pred'])
                
                # Get the corresponding dates
                all_dates = st.session_state['close_data'].index
                
                # Ensure we don't go out of bounds
                if lstm_end_idx <= len(all_dates):
                    lstm_dates = all_dates[lstm_start_idx:lstm_end_idx]
                    
                    # Flatten the predictions if they're 2D
                    lstm_pred_flat = st.session_state['lstm_pred'].flatten()
                    
                    # Ensure lengths match before plotting
                    if len(lstm_dates) == len(lstm_pred_flat):
                        ax.plot(lstm_dates, lstm_pred_flat, 
                               label='LSTM Forecast', color='orange', linestyle=':', linewidth=2)
                    else:
                        st.warning(f"LSTM prediction length mismatch: dates={len(lstm_dates)}, predictions={len(lstm_pred_flat)}")
                else:
                    st.warning("LSTM prediction indices out of bounds")
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Closing Price ($)')
            ax.set_title(f'{st.session_state["stock"]} - Model Forecast Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Generate and download report
        st.subheader("📄 Download Analysis Report")
        
        def generate_report():
            report = io.StringIO()
            report.write("=" * 60 + "\n")
            report.write("STOCK TIME SERIES FORECASTING ANALYSIS REPORT\n")
            report.write("=" * 60 + "\n\n")
            report.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            report.write(f"Stock: {st.session_state['stock']}\n")
            report.write(f"Data Period: {st.session_state['close_data'].index.min().date()} to {st.session_state['close_data'].index.max().date()}\n")
            report.write(f"Total Data Points: {len(st.session_state['close_data'])}\n\n")
            
            report.write("1. STATIONARITY TEST\n")
            report.write("-" * 30 + "\n")
            report.write(f"ADF Statistic: {st.session_state['adf_stat']:.4f}\n")
            report.write(f"P-value: {st.session_state['p_value']:.4f}\n")
            report.write(f"Conclusion: {st.session_state['stationarity_msg']}\n\n")
            
            if arima_trained:
                report.write("2. ARIMA MODEL PERFORMANCE\n")
                report.write("-" * 30 + "\n")
                report.write(f"ARIMA Order: ({arima_p},{arima_d},{arima_q})\n")
                report.write(f"MAE: ${st.session_state['arima_mae']:.2f}\n")
                report.write(f"RMSE: ${st.session_state['arima_rmse']:.2f}\n")
                report.write(f"MAPE: {st.session_state['arima_mape']:.2f}%\n\n")
            
            if lstm_trained:
                report.write("3. LSTM MODEL PERFORMANCE\n")
                report.write("-" * 30 + "\n")
                report.write(f"LSTM Lookback: {lstm_lookback} days\n")
                report.write(f"LSTM Epochs: {lstm_epochs}\n")
                report.write(f"MAE: ${st.session_state['lstm_mae']:.2f}\n")
                report.write(f"RMSE: ${st.session_state['lstm_rmse']:.2f}\n")
                report.write(f"MAPE: {st.session_state['lstm_mape']:.2f}%\n\n")
            
            if arima_trained and lstm_trained:
                report.write("4. MODEL COMPARISON\n")
                report.write("-" * 30 + "\n")
                report.write(f"Best Model: {best_model}\n")
                report.write(f"Improvement: {improvement:.1f}% lower RMSE\n\n")
            
            return report.getvalue()
        
        report_content = generate_report()
        
        st.download_button(
            label="📥 Download Complete Analysis Report",
            data=report_content,
            file_name=f"{st.session_state['stock']}_forecast_report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 10px;'>
    <p>Time Series Forecasting Dashboard | ARIMA & LSTM Models</p>
    <p>Skills: Time Series Analysis, ARIMA, LSTM, Model Evaluation</p>
</div>
""", unsafe_allow_html=True)