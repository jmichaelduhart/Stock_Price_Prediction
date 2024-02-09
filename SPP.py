import pandas as pd
import numpy as np

import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

plt.style.use('ggplot')

import yfinance as yf

# Download historical stock data
df = yf.download('AAPL', start='2011-01-01', end='2024-01-01')

# Display the downloaded data
print(df)

# Plot the close price history
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Price', fontsize=18)
plt.show()

# Data preprocessing
data = df.dropna()
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
training_data_len = math.ceil(len(data) * 0.8)

# Prepare training data
train_data = scaled_data[0:training_data_len, :]
x_train, y_train = [], []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build and compile the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Prepare testing data
test_data = scaled_data[training_data_len - 60:, :]
x_test, y_test = [], []
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    y_test.append(test_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Initialize the 'valid' DataFrame with the actual prices
valid = data[training_data_len:].copy()
valid['Prediction'] = np.nan

# Make predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Add the prediction results into 'valid' DataFrame
valid.iloc[:, valid.columns.get_loc('Prediction')] = predictions.flatten()

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(np.mean((predictions - y_test)**2))
print("Root Mean Squared Error (RMSE):", rmse)

plt.figure(figsize=(16,8))
plt.title('Model Performance')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Price', fontsize=18)
plt.plot(valid[['Close', 'Prediction']])
plt.legend(['Original', 'Predicted'])
plt.show()