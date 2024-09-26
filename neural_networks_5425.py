import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import yfinance as yf

# Λήψη δεδομένων μετοχών
stock_symbol = "MBG.DE"  # Mercedes Benz Group AG stock symbol στο Yahoo Finance
df = yf.download(stock_symbol, start="2010-01-01", end="2023-12-31")  # Προσαρμόστε τις ημερομηνίες όπως χρειάζεται

# Διατηρώ τις ημερομηνίες σε ξεχωριστές μεταβλητές
dates = df.index

# Προεπεξεργασία των δεδομένων
data = df["Close"].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)


# Δημιουργία συνόλου δεδομένων για την εκπαίδευση
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)


time_step = 100
X, Y = create_dataset(data_scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Διαίρεση του συνόλου δεδομένων σε τμήμα εκπαίδευσης και δοκιμής
training_size = int(len(X) * 0.67)
X_train, X_test = X[:training_size], X[training_size:]
Y_train, Y_test = Y[:training_size], Y[training_size:]

# Δημιουργία μοντέλου LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compiling του μοντέλου
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Εκπαίδευση του μοντέλου
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=64, verbose=1)

# Πρόβλεψη και απεικόνιση των αποτελεσμάτων
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Μετατροπή πίσω στην αρχική μορφή
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Δημιουργία γραφικής απεικόνισης
plt.figure(figsize=(15,6))

# Δημιουργία γραφήματος των αρχικών δεδομένων
plt.plot(dates, scaler.inverse_transform(data_scaled), label='Original Data', color='blue')

# Προσαρμογή των ημερομηνιών για τις προβλέψεις εκπαίδευσης και δοκιμής
train_dates = dates[time_step:training_size + time_step]

# Επιβεβαίωση συμβατώτητας των ημερομηνιών
test_dates = dates[-len(test_predict):]

# Δημιουργία γραφικής απεικόνισης (plot) των προβλέψεων εκπαίδευσης
plt.plot(train_dates, train_predict, label='Training Predictions', color='green')

# Δημιουργία γραφικής απεικόνισης (plot) των προβλέψεων δοκιμής
plt.plot(test_dates, test_predict, label='Test Predictions', color='red')

# Προσθήκη τίτλου και ετικετών στο γράφημα
plt.title('Stock Price Prediction - Mercedes Benz Group AG')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.xticks(rotation=45)  # Περιστροφή των ημερομηνιών για καλύτερη ορατότητα

# Προσθήκη λεζάντας
plt.legend()

plt.show()


# Πρόβλεψη μελλοντικών τιμών μετοχών
def predict_future(num_prediction, model):
    prediction_list = data_scaled[-time_step:]

    for _ in range(num_prediction):
        x = prediction_list[-time_step:]
        x = x.reshape((1, time_step, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, [[out]], axis=0)
    return prediction_list


def predict_dates(num_prediction):
    last_date = dates[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction + 1).tolist()
    return prediction_dates[1:]


num_prediction = 365  # Πρόβλεψη για την επόμενη χρονιά
forecast = predict_future(num_prediction, model)
forecast_dates = predict_dates(num_prediction)

# Απεικόνιση της πρόβλεψης
plt.figure(figsize=(15, 6))
plt.plot(dates, scaler.inverse_transform(data_scaled), label='Historical Data', color='blue')
plt.plot(forecast_dates, scaler.inverse_transform(forecast[-num_prediction:].reshape(-1, 1)), label='Forecast',
         color='orange')
plt.title('Future Stock Price Forecast - Mercedes Benz Group AG')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.xticks(rotation=45)
plt.legend()
plt.show()
