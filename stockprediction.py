import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Load the data
df = pd.read_csv('AAPL.csv')
df = df[['Close']]

# Prepare the data
scaler = MinMaxScaler(feature_range=(0,1))
df = scaler.fit_transform(df)

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train_data, test_data = df[0:train_size,:], df[train_size:len(df),:]

# Define a function to create the X and y data for the LSTM model
def create_dataset(dataset, time_steps=1):
    X_data, y_data = [], []
    for i in range(len(dataset) - time_steps - 1):
        a = dataset[i:(i + time_steps), 0]
        X_data.append(a)
        y_data.append(dataset[i + time_steps, 0])
    return np.array(X_data), np.array(y_data)

# Create the training and testing data for the LSTM model
time_steps = 60
X_train, y_train = create_dataset(train_data, time_steps)
X_test, y_test = create_dataset(test_data, time_steps)

# Reshape the data to fit the LSTM input shape
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_steps, 1)))
model.add(LSTM(50))
model.add(Dense(1))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=1)

# Make predictions with the LSTM model
y_pred = model.predict(X_test)

# Inverse the scaling to get the actual stock prices
y_test = y_test.reshape(-1, 1)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# Plot the actual and predicted stock prices
plt.plot(y_test, label='Actual Stock Price')
plt.plot(y_pred, label='Predicted Stock Price')
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('AAPL Stock Price')
plt.legend()
plt.show()
