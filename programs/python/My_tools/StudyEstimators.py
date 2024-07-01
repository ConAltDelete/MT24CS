from typing import Literal
from matplotlib.pylab import RandomState
import numpy as np
import pandas as pd
import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from keras.models import Sequential, Model
from keras.layers import LSTM, GRU, Bidirectional, Input, Conv1D, Concatenate
from keras import Input
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.utils import timeseries_dataset_from_array

class KerasBiLSTM(MLPRegressor):
    def __init__(self, input_shape = 24, num_classes = 1, lstm_units=24, epochs=10, batch_size = 756,**kwargs):
        super().__init__(**kwargs)
        self.input_shape: int = input_shape
        self.num_classes: int = num_classes
        self.lstm_units: int = lstm_units
        self.epochs: int = epochs
        self.model = None
        self.spaceing: int = 1
        self.batch_size: int = batch_size
        self.is_fitted_ = False
        self.other_params = kwargs

        self.history = None

        self.scaler_x = None
        self.scaler_y = None

    def fit(self, X, y):
        if isinstance(X,list) and isinstance(y,list):
            self.iterartive_fit(X,y)
            return self
        elif (isinstance(X,list) and not(isinstance(y,list))) or (isinstance(y,list) and not(isinstance(X,list))):
            raise ValueError("If X is a {} then y has to be the same, got {}".format(type(X),type(y)))
        if "Time" in X.columns:
            X["Time"] = X["Time"].transform({"Time":lambda x: x.day_of_year*24 + x.hour})

        check_X_y(X, y) # Checks if values are finite and not too sparce.
        # Data treatment
        All_data = self._data_treatment(X,y) # Takes both just incase.
        # Setting up model
        self.model_fit(All_data)
        return self

    def _log_model(self, h):
        f = open("logs/{}".format(type(self).__name__) + "_" + str(datetime.datetime.today().strftime('%Y-%m-%d')) + ".hist","a+")
        f.write("[{}]: ".format(str(datetime.datetime.now())))
        f.write(str(h.history))
        f.write("\n{}".format(str(self.__dict__)))
        f.write("\n{}\n".format(str(self.model.__dict__)))
        f.close()

    def model_fit(self, All_data):
        #print(All_data.__dict__)
        if not(self.is_fitted_):
            self.model = Sequential()
            self.model.add(Input( shape = (self.input_shape, All_data._structure[0].shape[-1]) ))
            #? add a convelution layer or two here?
            self.model.add(Bidirectional(LSTM(self.lstm_units,return_sequences=True),merge_mode="concat"))
            #self.model.add(LSTM(self.num_classes))
            self.model.add(LSTM(int(self.lstm_units / 2)+1,return_sequences=True)) # conjestion
            self.model.add(LSTM(self.num_classes))
            self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error','r2_score'],jit_compile=True)
            #print(self.model.output_shape)
        # fitting model
        #print(self.model.summary())
        #print("Input shape for LSTM:", All_data[0][0].shape)

        h = self.model.fit(All_data, epochs=self.epochs, verbose=1,**self.other_params)
        self.history = h.history
        self._log_model(h)
        self.is_fitted_ = True
        return h
    
    def iterartive_fit(self, X: list, y:list):
        """
            If the data is segmented the model will be trained on the individual parts.
        Args:
            X : list of training data
            y : list of ground truth

        Returns:
            self
        """
        # check if consistent
        if len(X) != len(y):
            raise ValueError("Inconsistent length: X length is {} while y length is {}".format(len(X),len(y)))
        
        total_data = None #! må fikses
        for data in zip(X,y):
            if "Time" in data[0].columns:
                data[0]["Time"] = data[0]["Time"].transform({"Time":lambda x: x.day_of_year*24 + x.hour})
            check_X_y(data[0], data[1])
            new_data = self._data_treatment(data[0], data[1])
            
            if total_data is None:
                total_data = new_data
            else:
                total_data = total_data.concatenate(new_data)
        h = self.model_fit(total_data)
        self._log_model(h)
        return h

    def _data_treatment(self,X,y=None, just_transform = False):
        """
            Treats data so it fits model. Scalers are saved to scaler_x and scaler_y.
        Args:
            X : Traning data
            y : target data
        
        Returns:
            data TimeSeriesGenerator
        """
        if just_transform:
            if "Time" in X.columns:
                sep_time: pd.DataFrame = X["Time"] / (24*365) # Treating time seperated to range (0,1]
                new_X = self.scaler_x.transform(X.loc[:,X.columns != "Time"].to_numpy()).astype("float32")
                new_X = np.concatenate(
                    (sep_time.to_numpy().reshape((-1,1)),new_X),axis=1
                )
            else:
                new_X = self.scaler_x.transform(X.to_numpy()).astype("float32").to_numpy()
            new_y = self.scaler_y.transform(y.reshape(-1, 1)).astype("float32") if y is not None else None
            return self._data_generate(new_X,new_y)

        # convert dataframe to numpy array
        if self.scaler_x is None:
            self.scaler_x = MinMaxScaler(feature_range=(-1,1)) # StandardScaler() #
            self.scaler_x.data_min_ = -10 # pre-setting values to improve initial performance during CV
            self.scaler_x.data_max_ = 25
            self.scaler_x.n_samples_seen_ = 2
        if self.scaler_y is None:
            self.scaler_y = MinMaxScaler(feature_range=(-1,1)) # StandardScaler() #
            self.scaler_y.data_min_ = -1
            self.scaler_y.data_max_ = 20
            self.scaler_y.n_samples_seen_ = 2
        
        if "Time" in X.columns:
            sep_time: pd.DataFrame = X["Time"] / (24*365) # Treating time seperated to range (0,1]
            new_X = self.scaler_x.partial_fit(X.loc[:,X.columns != "Time"].to_numpy())
            new_X = self.scaler_x.transform(X.loc[:,X.columns != "Time"].to_numpy()).astype("float32")
            #print("before:",new_X,"sep:",sep_time)
            new_X = np.concatenate(
                (sep_time.to_numpy().reshape((-1,1)),new_X),axis=1
            )
            #print("after:",new_X)
        else:
            new_X = self.scaler_x.partial_fit(X.to_numpy())
            new_X = self.scaler_x.transform(X.to_numpy()).astype("float32")
        if y is not None:
            new_y = self.scaler_y.partial_fit(y.to_numpy().reshape(-1, 1))
            new_y = self.scaler_y.transform(y.to_numpy().reshape(-1, 1)).astype("float32")
        else:
            new_y = None
        
        #print("Shape of new_X:", new_X.shape)
        #print("Shape of new_y:", new_y.shape)
        if np.isnan(new_X).any():
            raise ValueError("before:{} after:{}".format(X,new_X))

        self.transformed_data = self._data_generate(new_X,new_y)

        return self.transformed_data

    def _data_generate(self,X,y = None):
        transformed_data = timeseries_dataset_from_array(X,targets = y,
                            sequence_length=self.input_shape,
                            sequence_stride=self.spaceing,
                            shuffle=False,
                            **self.other_params
                )
        return transformed_data

    def predict(self, X):
        X = X.copy()
        check_is_fitted(self, 'is_fitted_')
        if "Time" in X.columns:
            X["Time"] = X["Time"].transform({"Time":lambda x: x.day_of_year*24 + x.hour})
        check_array(X)
        #if "Time" in X and isinstance(X.iloc[0,"Time"],pd.Timestamp):
        #    X["Time"] = X["Time"].transform({"Time":lambda x: x.day_of_year*24 + x.hour})
        #print(X.shape)
        trans_X = self._data_treatment(X,just_transform=True)

        pred_y = self.model.predict(
                trans_X
            )
        #print(pred_y)
        trans_y = self.scaler_y.inverse_transform(pred_y.reshape((-1,1)))
        #print(np.shape(trans_y))
        return trans_y.flatten()
    
    def score(self,X,y, sample_weight=None):
        """
            Modified score function that removes the first target values.
        """
        return super().score(X = X,
                            y = y[:-(self.input_shape-1):self.spaceing],
                            sample_weight=sample_weight
                            )


class l1KerasBiLSTM(KerasBiLSTM):
    def model_fit(self, All_data):
        #print(All_data.__dict__)
        if not(self.is_fitted_):
            self.model = Sequential()
            self.model.add(Input( shape = (self.input_shape, All_data._structure[0].shape[-1]) ))
            #? add a convelution layer or two here?
            self.model.add(Bidirectional(LSTM(self.num_classes),merge_mode="ave"))
            self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error','r2_score'],jit_compile=True)

        h = self.model.fit(All_data, epochs=self.epochs, verbose=1,**self.other_params)
        self.history = h.history
        self._log_model(h)
        self.is_fitted_ = True
        return h

class l2KerasBiLSTM(KerasBiLSTM):
    def model_fit(self, All_data):
        #print(All_data.__dict__)
        if not(self.is_fitted_):
            self.model = Sequential()
            self.model.add(Input( shape = (self.input_shape, All_data._structure[0].shape[-1]) ))
            #? add a convelution layer or two here?
            self.model.add(LSTM(self.num_classes))
            self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error','r2_score'],jit_compile=True)

        h = self.model.fit(All_data, epochs=self.epochs, verbose=1,**self.other_params)
        self.history = h.history
        self._log_model(h)
        self.is_fitted_ = True
        return h

class KerasGRU(KerasBiLSTM):
    def model_fit(self, All_data):
        #print(All_data.__dict__)
        if not(self.is_fitted_):
            self.model = Sequential()
            self.model.add(Input( shape = (self.input_shape, All_data._structure[0].shape[-1]) ))
            #? add a convelution layer or two here?
            self.model.add(GRU(self.num_classes))
            self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error','r2_score'],jit_compile=True)

        h = self.model.fit(All_data, epochs=self.epochs, verbose=1,**self.other_params)
        self.history = h.history
        self._log_model(h)
        self.is_fitted_ = True
        return h

class KerasBiGRU(KerasBiLSTM):
    def model_fit(self, All_data):
        #print(All_data.__dict__)
        if not(self.is_fitted_):
            self.model = Sequential()
            self.model.add(Input( shape = (self.input_shape, All_data._structure[0].shape[-1]) ))
            #? add a convelution layer or two here?
            self.model.add(Bidirectional(GRU(self.num_classes),merge_mode="ave"))
            self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error','r2_score'],jit_compile=True)
            #self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error','r2_score'])

        h = self.model.fit(All_data, epochs=self.epochs, verbose=1,**self.other_params)
        self.history = h.history
        self._log_model(h)
        self.is_fitted_ = True
        return h

class modKerasBiLSTM(KerasBiLSTM):
    def model_fit(self, All_data):
        #print(All_data.__dict__)
        if not(self.is_fitted_):
            input_layer = Input( shape = (self.input_shape, All_data._structure[0].shape[-1]) )
            #? add a convelution layer or two here?
            conv_layer = Conv1D(2*self.lstm_units, int(self.input_shape / 2), padding = "same")(input_layer)
            Concatenate_layer = Concatenate()([input_layer,conv_layer])
            #Bidirectional(LSTM(self.lstm_units,return_sequences=True))(Concatenate_layer)
            
            bi_layer = Bidirectional(LSTM(self.lstm_units,return_sequences=True))(Concatenate_layer)
            lstm_layer = LSTM(self.num_classes,activation=None)(bi_layer) # conjestion
            #summary_layer = Dense(self.num_classes, activation='softmax')(lstm_layer) # conjegtion
            self.model = Model(inputs = input_layer,outputs = lstm_layer)
            self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error','r2_score'],jit_compile=True)

        h = self.model.fit(All_data, epochs=self.epochs, verbose=1,**self.other_params)
        self.history = h.history
        self._log_model(h)
        self.is_fitted_ = True
        return h

class PlauborgRegresson(LinearRegression):
    def __init__(self,lag_max = 2, fourier_sin_length = 2, fourier_cos_length = 2, is_day = False,fit_intercept = False,**kvarg):
        super().__init__(fit_intercept = fit_intercept,**kvarg)
        self.lag_max = lag_max
        self.fourier_sin_length = fourier_sin_length
        self.fourier_cos_length = fourier_cos_length
        self.is_day = is_day
        self.fit_intercept = fit_intercept

    def fit(self, X, y=None):
        self.is_fitted_ = True
        if isinstance(X,list):
            new_X = self.F_plauborg(X[0])
            new_y = y[0]
            for TX,Ty in zip(X[1:],y[1:]):
                new_X = pd.concat([new_X,self.F_plauborg(TX)],axis=0,ignore_index=True)
                new_y = pd.concat([new_y,Ty],axis=0,ignore_index=True)
        else:
            new_X = self.F_plauborg(X)
            new_y = y
        return super().fit(new_X,new_y)
    def F_plauborg(self,df: pd.DataFrame):
        """
            Fxn is based on a full year while df could have any range.
        """
        new_df = df.set_index("Time")
        #if self.is_day:
        #   new_df = new_df.infer_objects(copy=False).resample("1D").mean().resample("1h").ffill()
        data_ret = pd.DataFrame(index=new_df.index)

        data_ret = pd.DataFrame({
            "B0":new_df["TM"].values
            })
        freq = 2*np.pi / (365 if self.is_day else 24)
        fourier = [self.fourier_sin_length,self.fourier_cos_length]
        order_of_fourier = fourier if fourier[0] < fourier[1] else fourier[::-1]
        freq_index = freq * ( new_df.index.dayofyear if self.is_day else (new_df.index.dayofyear*24 + new_df.index.hour))
        for i in range(1,self.lag_max+1): 
           data_ret[f"B{i}"] = new_df["TM"].shift(i).values
        for i in range(1,order_of_fourier[0]+1):
           c = i * freq_index
           data_ret[f"FS{i}"] = np.sin(c)
           data_ret[f"FC{i}"] = np.cos(c)
        func = [np.sin,np.cos][fourier[1] > fourier[0]]
        max_string = "S" if fourier[0] > fourier[1] else "C"
        for i in range(order_of_fourier[0]+1,order_of_fourier[1]+1):
           c = i * freq_index
           data_ret[f"F{max_string}{i}"] = func(c)
        return data_ret.infer_objects(copy=False).fillna(0) #! Shit..., this changes the output if it already contains Nan.
    
    def predict(self,X):
        check_is_fitted(self, 'is_fitted_')
        new_X = self.F_plauborg(X)
        return super().predict(new_X)

class MultiLinearRegresson(LinearRegression):
    def __init__(self,lag_max = 2,fit_intercept = False,**kvarg):
        super().__init__(fit_intercept = fit_intercept,**kvarg)
        self.lag_max = lag_max
        self.fit_intercept = fit_intercept

    def fit(self, X, y=None):
        self.is_fitted_ = True
        if isinstance(X,list):
            new_X = self.F_plauborg(X[0])
            new_y = y[0]
            for TX,Ty in zip(X[1:],y[1:]):
                new_X = pd.concat([new_X,self.F_plauborg(TX)],axis=0,ignore_index=True)
                new_y = pd.concat([new_y,Ty],axis=0,ignore_index=True)
        else:
            new_X = self.F_plauborg(X)
            new_y = y
        return super().fit(new_X,new_y)

    def F_plauborg(self,df: pd.DataFrame):
        """
            Fxn is based on a full year while df could have any range.
        """
        new_df = df.set_index("Time")
        #if self.is_day:
        #   new_df = new_df.infer_objects(copy=False).resample("1D").mean().resample("1h").ffill()
        data_ret = pd.DataFrame(index=new_df.index)

        data_ret = pd.DataFrame({
            "B0":new_df["TM"].values
            })

        for i in range(1,self.lag_max+1): 
           data_ret[f"B{i}"] = new_df["TM"].shift(i).values

        return data_ret.infer_objects(copy=False).fillna(0) #! Shit..., this changes the output if it already contains Nan.
    
    def predict(self,X):
        #print("Input:",X.shape)
        check_is_fitted(self, 'is_fitted_')
        new_X = self.F_plauborg(X)
        #print("reshape:",new_X.shape)
        pred_y = super().predict(new_X)
        #print("predicted:",pred_y.shape)
        return pred_y