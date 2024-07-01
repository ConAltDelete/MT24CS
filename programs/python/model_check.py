from keras.layers import LSTM, Bidirectional,Dense,Input
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import pandas as pd
import numpy as np

num_classes = 1

model = Sequential([
    Input((12,2)),
    # Bidirectional(LSTM(num_classes)),
    LSTM(num_classes),
    #Dense(num_classes, activation='softmax')
])
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error','r2_score'])

r = pd.date_range("2017-03-01 01:00","2017-03-01 23:00",freq="1h")

test_data = pd.DataFrame({
    "Time": r,
    "TM": np.linspace(1,10,num=len(r)),
    "TJM20": np.linspace(10,1,num=len(r))
})

test_data["Time"] = test_data["Time"].transform({"Time":lambda x: x.day_of_year*24 + x.hour}).astype(np.float32)

X, y = check_X_y(test_data[["Time","TM"]],test_data["TJM20"])

scaler_x = StandardScaler()
scaler_y = StandardScaler()

trans_X = scaler_x.fit_transform(X).astype("float32")
trans_y = scaler_y.fit_transform(y.reshape(-1,1)).astype("float32")

test_data_mod = TimeseriesGenerator(trans_X,trans_y,length=12)