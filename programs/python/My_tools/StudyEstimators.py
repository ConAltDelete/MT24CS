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
from keras.layers import LSTM, Dense, Bidirectional, Input, Conv1D, Concatenate
from keras import Input
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.utils import timeseries_dataset_from_array


import torch

class KerasBiLSTM(MLPRegressor):
    def __init__(self, input_shape = 24, num_classes = 1, lstm_units=24, epochs=10, batch_size = 768,**kwargs):
        super().__init__(**kwargs)
        self.input_shape: int = input_shape
        self.num_classes: int = num_classes
        self.lstm_units: int = lstm_units
        self.epochs: int = epochs
        self.model = None
        self.spaceing: int = 1
        self.batch_size: int = batch_size
        self.is_fitted_ = False

        #self.history = None

        self.scaler_x = None
        self.scaler_y = None
    def fit(self, X, y):
        if "Time" in X.columns:
            X["Time"] = X["Time"].transform({"Time":lambda x: x.day_of_year*24 + x.hour})

        X, y = check_X_y(X, y) # Checks if values are finite and not too sparce.
        # Data treatment
        All_data = self._data_treatment(X,y) # Takes both just incase.
        # Setting up model
        if not(self.is_fitted_):
            self.model = Sequential()
            self.model.add(Input(All_data[0][0].shape[1:]))
            #? add a convelution layer or two here?
            self.model.add(Bidirectional(LSTM(self.lstm_units,return_sequences=True),merge_mode="concat"))
            #self.model.add(LSTM(self.num_classes))
            self.model.add(LSTM(int(self.lstm_units / 2)+1,return_sequences=True)) # conjestion
            self.model.add(LSTM(self.num_classes))
            self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error','r2_score'])
            #print(self.model.output_shape)
        # fitting model
        #print(self.model.summary())
        #print("Input shape for LSTM:", All_data[0][0].shape)

        h = self.model.fit(All_data, epochs=self.epochs, verbose=1)

        f = open("logs/KerasBiLSTM" + "_" + str(datetime.datetime.today().strftime('%Y-%m-%d')) + ".hist","a+")
        f.write("[{}]: ".format(str(datetime.datetime.now())))
        f.write(str(h.history))
        f.write("\n{}".format(str(self.__dict__)))
        f.write("\n{}\n".format(str(self.model.__dict__)))
        f.close()

        self.is_fitted_ = True
        return self

    def _data_treatment(self,X,y=None):
        """
            Treats data so it fits model. Saves the inverse function at `_inv_data`
        Args:
            X : Traning data
            y : target data
        
        Returns:
            data TimeSeriesGenerator
        """

        # convert dataframe to numpy array
        if self.scaler_x is None:
            self.scaler_x = MinMaxScaler(feature_range=(-1,1)) # StandardScaler() #
            self.scaler_x.data_min_ = -10
            self.scaler_x.data_max_ = 25
            self.scaler_x.n_samples_seen_ = 2
        if self.scaler_y is None:
            self.scaler_y = MinMaxScaler(feature_range=(-1,1)) # StandardScaler() #
            self.scaler_y.data_min_ = -1
            self.scaler_y.data_max_ = 20
            self.scaler_y.n_samples_seen_ = 2

        new_X = self.scaler_x.partial_fit(X)
        new_X = self.scaler_x.transform(X).astype("float32")
        if y is not None:
            new_y = self.scaler_y.partial_fit(y.reshape(-1, 1))
            new_y = self.scaler_y.transform(y.reshape(-1, 1)).astype("float32")
        else:
            new_y = None
        
        #print("Shape of new_X:", new_X.shape)
        #print("Shape of new_y:", new_y.shape)

        self.transformed_data = self._data_generate(new_X,new_y)

        return self.transformed_data

    def _data_generate(self,X,y):
        transformed_data = TimeseriesGenerator(X, y, # Due to spaceing we remove the first target values so we get the relavant target
                               length=self.input_shape, stride=self.spaceing,
                               batch_size=self.batch_size)

        return transformed_data

    def predict(self, X):
        X = X.copy()
        check_is_fitted(self, 'is_fitted_')
        if "Time" in X.columns:
            X["Time"] = X["Time"].transform({"Time":lambda x: x.day_of_year*24 + x.hour})
        X = check_array(X)
        if "Time" in X and isinstance(X.iloc[0,"Time"],pd.Timestamp):
            X["Time"] = X["Time"].transform({"Time":lambda x: x.day_of_year*24 + x.hour})
        #print(X.shape)
        trans_X = timeseries_dataset_from_array(self.scaler_x.transform(X).astype("float32"),targets = None,
                            sequence_length=self.input_shape,
                            sequence_stride=self.spaceing,
                            shuffle=False             
                )

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
    def fit(self, X, y):
        if "Time" in X.columns:
            X["Time"] = X["Time"].transform({"Time":lambda x: x.day_of_year*24 + x.hour})

        X, y = check_X_y(X, y) # Checks if values are finite and not too sparce.
        # Data treatment
        All_data = self._data_treatment(X,y) # Takes both just incase.
        # Setting up model
        if not(self.is_fitted_):
            self.model = Sequential()
            self.model.add(Input(All_data[0][0].shape[1:]))
            #? add a convelution layer or two here?
            self.model.add(Bidirectional(LSTM(self.num_classes),merge_mode="ave"))
            self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error','r2_score'])

        h = self.model.fit(All_data, epochs=self.epochs, verbose=1)

        f = open("logs/l1KerasBiLSTM" + "_" + str(datetime.datetime.today().strftime('%Y-%m-%d')) + ".hist","a+")
        f.write("[{}]: ".format(str(datetime.datetime.now())))
        f.write(str(h.history))
        f.write("\n{}".format(str(self.__dict__)))
        f.write("\n{}\n".format(str(self.model.__dict__)))
        f.close()

        self.is_fitted_ = True
        return self

class l2KerasBiLSTM(KerasBiLSTM):
    def fit(self, X, y):
        if "Time" in X.columns:
            X["Time"] = X["Time"].transform({"Time":lambda x: x.day_of_year*24 + x.hour})

        X, y = check_X_y(X, y) # Checks if values are finite and not too sparce.
        # Data treatment
        All_data = self._data_treatment(X,y) # Takes both just incase.
        # Setting up model
        if not(self.is_fitted_):
            self.model = Sequential()
            self.model.add(Input(All_data[0][0].shape[1:]))
            #? add a convelution layer or two here?
            self.model.add(LSTM(self.num_classes))
            self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error','r2_score'])

        h = self.model.fit(All_data, epochs=self.epochs, verbose=1)

        f = open("logs/l2KerasBiLSTM" + "_" + str(datetime.datetime.today().strftime('%Y-%m-%d')) + ".hist","a+")
        f.write("[{}]: ".format(str(datetime.datetime.now())))
        f.write(str(h.history))
        f.write("\n{}".format(str(self.__dict__)))
        f.write("\n{}\n".format(str(self.model.__dict__)))
        f.close()

        self.is_fitted_ = True
        return self

class modKerasBiLSTM(KerasBiLSTM):
    def fit(self, X, y):
        if "Time" in X.columns:
            X["Time"] = X["Time"].transform({"Time":lambda x: x.day_of_year*24 + x.hour})

        X, y = check_X_y(X, y) # Checks if values are finite and not too sparce.
        # Data treatment
        All_data = self._data_treatment(X,y) # Takes both just incase.
        # Setting up model
        if not(self.is_fitted_):
            
            input_layer = Input(All_data[0][0].shape[1:])
            #? add a convelution layer or two here?
            conv_layer = Conv1D(2*self.lstm_units, int(self.input_shape / 2), padding = "same")(input_layer)
            Concatenate_layer = Concatenate()([input_layer,conv_layer])
            #Bidirectional(LSTM(self.lstm_units,return_sequences=True))(Concatenate_layer)
            
            bi_layer = Bidirectional(LSTM(self.lstm_units,return_sequences=True))(Concatenate_layer)
            lstm_layer = LSTM(self.num_classes,activation=None)(bi_layer) # conjestion
            #summary_layer = Dense(self.num_classes, activation='softmax')(lstm_layer) # conjegtion
            self.model = Model(inputs = input_layer,outputs = lstm_layer)
            self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error','r2_score'])
            #print(self.model.output_shape)
        # fitting model
        #print(self.model.summary())
        #print("Input shape for LSTM:", All_data[0][0].shape)

        h = self.model.fit(All_data, epochs=self.epochs, verbose=1)

        f = open("logs/modKerasBiLSTM" + "_" + str(datetime.datetime.today().strftime('%Y-%m-%d')) + ".hist","a+")
        f.write("[{}]: ".format(str(datetime.datetime.now())))
        f.write(str(h.history))
        f.write("\n{}".format(str(self.__dict__)))
        f.write("\n{}\n".format(str(self.model.__dict__)))
        f.close()

        self.is_fitted_ = True
        return self

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

class RankinRegresson(LinearRegression):
    """
        Performs a regression inspired by the paper DOI:10.5194/hess-8-706-2004 so 
        it only relies on 2 parameters (air temperature and rain, though other parameters can be suplied to increase accuretsy.)
    """
    def __init__(self,alpha_t = None, fs = None, dt = None, depth = 0.15, padepuiseux = False):
        self.alpha_t = alpha_t
        self.fs = fs
        self.dt = dt
        self.depth = depth
        self._is_rain = False
        self._is_snow = False
        self.padepuiseux = padepuiseux

    def fit(self, X: pd.DataFrame, y : pd.DataFrame):
        self.Tdiff = X.loc[:,"Time"].diff().abs().infer_objects(copy=False).fillna(pd.Timedelta(hours=1)) # Abs since we might have wrap-around
        if "RR" in X.columns:
            self._is_rain = True
        if self.dt is None:
            diffrange = self.Tdiff.nsmallest(2)
            self.dt = diffrange.iloc[0].total_seconds()/(60*60) if diffrange.iloc[0] > pd.Timedelta(0) else diffrange.iloc[1].total_seconds()/(60*60)
        if (self.alpha_t is None) and not(self.padepuiseux):
            alpha_t_bot = (X.TM.iloc[0] - 2*X.TJM10.iloc[0] + X.TJM20.iloc[0]) / (0.1)**2 # looking around 10cm and assuming it is applicible to both 0.1 and 0.2
            alpha_t_top = (X.TM.iloc[0] - X.TJM20.iloc[0]) / (2*self.dt)
            self.alpha_t = alpha_t_top / alpha_t_bot
            if self.alpha_t <= 0:
                self.alpha_t = 1 # scales up to avoid 0/0
            self.soilDamp = self.alpha_t/(2*self.depth)**2
        elif self.padepuiseux and ("RR" in X.columns) and (self.alpha_t is None):
            self._is_rain = True
            bot = np.array([(a-b)*self.dt for a,b in zip(X.TM.to_numpy() , y.to_numpy())]) 
            top = np.array([(a-b)*(2*self.depth)**2 for a,b in zip(y[1:].to_numpy(),X.TM[:X.shape[0]-2].to_numpy())])
            self.rainMatrix = pd.DataFrame(data = {"const" : [1 for _ in range(X.shape[0])],
                                                   "R1" : X.RR.infer_objects(copy=False).fillna(0).to_numpy(),
                                                   "R2" : np.sqrt(X.RR).infer_objects(copy=False).fillna(0).to_numpy()} ) #! gets nan values, must be fixed either here or in pre-prosess
            reg_bot, reg_top = LinearRegression(), LinearRegression()
            reg_top.fit(self.rainMatrix.iloc[:self.rainMatrix.shape[0]-2,:],top)
            reg_bot.fit(self.rainMatrix.loc[:,["const","R1"]],bot)
            self.coef_top = list(reg_top.coef_[0])
            self.coef_bot = list(reg_bot.coef_[0])
            del reg_bot, reg_top # saves memory since we don't need these intermediet calculations
            # print(len(self.coef_top),len(self.coef_bot))
            self.alpha_t = lambda theta: (self.coef_top[0] + self.coef_top[1]*theta + self.coef_top[2]*np.sqrt(theta)) / \
                ((self.coef_bot[0] if self.coef_bot[0] != 0 else 0.01) + self.coef_bot[1]*theta) # addinga small epsilon to avoid 0/0 at RR = 0
            self.soilDamp = lambda RR: self.alpha_t(RR)/(2*self.depth)**2
        
        self._is_snow = False if "snow" not in X.columns else [not(pd.isna(r)) for r in X["snow"]]
        if "snow" in X.columns: # assumes constatn TM at infinity and convergence
            if self.fs is None:
                fs_inter = -0.5*np.ln((y.iloc[1:(24*3 + 1)]+274.15).mean() / \
                    (y.iloc[0:(24*3)].mean+274.15 + self.soilDamp*(y.iloc[0:(24*3)] - X["TM"].iloc[0:(24*3)]).mean())) / \
                    (X.iloc[0:(24*3),"snow"].mean() if X.iloc[0:(24*3),"snow"].mean() > 0 else np.inf) # converts to celsius to avoid 0/0
                self.fs = fs_inter
            if callable(self.soilDamp):
                k = self.dt*self.soilDamp(X.RR.iloc[0:(24*3)]).mean()
            else:
                k = self.dt*self.soilDamp
            D = -fs_inter*X[0:(24*3),"snow"].mean()
            self.T_init = ((k*np.exp(D))/(1+np.exp(D)*(k-1)))*X["TM"].iloc[0:(24*3)].mean()
        else:
            if callable(self.alpha_t):
                a_t = self.alpha_t(X.RR.iloc[0:(24*3)]).mean()
                d = np.sqrt(2*(a_t if a_t > 0 or not(pd.isna(a_t)) else 1)/self.dt)
            else:
                d = np.sqrt(2*self.alpha_t/self.dt)
            w_big = 2*np.pi / (24)
            w_small = 2*np.pi / (365*24)
            self.T_init =  X["TM"].iloc[0:24*3].mean() + \
                np.abs(np.max(np.abs(X["TM"].iloc[0:(24)])) - X["TM"].iloc[0:(24)].mean())*\
                np.exp(-0.15/d)*np.sin(w_big*(X.Time.iloc[0].dayofyear*24 + X.Time.iloc[0].hour)-self.depth/d) + \
                np.abs(np.max(np.abs(X["TM"].iloc[0:(24*3)])) - X["TM"].iloc[0:(24*3)].mean())*\
                np.exp(-0.15/d)*np.sin(w_small*(X.Time.iloc[0].dayofyear)-self.depth/d)
            self.fs = 1 # Does not matter really
            # 0.15 = middle of 10cm and 15cm
        
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        T_z = np.zeros(X.shape[0])
        T_z[0] = self.T_init
        
        for t in range(X.shape[0]-1):
            if callable(self.soilDamp):
                T_z[t+1] = T_z[t] + self.Tdiff.iloc[t+1].total_seconds()/(60*60)*self.soilDamp(X.RR.iloc[t])*(X["TM"].iloc[t]-T_z[t])
            else:
                T_z[t+1] = T_z[t] + self.Tdiff.iloc[t+1].total_seconds()/(60*60)*self.soilDamp*(X.TM.iloc[t]-T_z[t])
            if self._is_snow:
                T_z[t+1] = T_z[t+1]*np.exp(-X["snow"].iloc[t]*self.fs)
        return T_z

from .lstm_interprety_soil_moisture import nibio_data_transform, LSTMDataGenerator, ILSTM_SV, train_lstm, create_predictions 
import torch.utils.data as Data

class ILSTM(MLPRegressor):
    def __init__(self,lead_time = 1, batch_size = 64,seq_length = 24, hidden_size = 2**5,learning_rate = 0.5,total_epoch = 3,**kwarg):
        super().__init__(**kwarg)
        self.lead_time = lead_time
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.total_epoch = total_epoch
        self.is_fitted_ = False

    def fit(self,X, y):

        # TODO:  remember to add data["Time"].transform({"Time":lambda x: x.day_of_year*24 + x.hour})
        data,self.scaler,self.scaler1 = nibio_data_transform(X, y)
        data = self.scaler1.transform(data)

        # TODO: Generate the tensor for lstm model

        [data_x, data_y,_] = LSTMDataGenerator(data, self.lead_time, self.batch_size, self.seq_length)

        # concat all variables.
        # TODO: Flexible valid split
        data_train_x = data_x[:int((data_x.shape[0])-400)]
        data_train_y = data_y[:int(data_x.shape[0]-400)]

        train_data = Data.TensorDataset(data_train_x, data_train_y)
        train_loader = Data.DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

        data_valid_x      = data_x[int(data_x.shape[0]-400):int(data_x.shape[0]-365)]
        data_valid_y      = data_y[int(data_x.shape[0]-400):int(data_x.shape[0]-365)]

        # TODO: Flexible input shapes and optimizer
        # IMVTensorLSTM,IMVFullLSTM
        if not(self.is_fitted_):
            self.model = ILSTM_SV(data_x.shape[2],data_x.shape[1], 1, self.hidden_size)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        # TODO: Trian LSTM based on the training and validation sets
        self.model,_,_= train_lstm(self.model,self.learning_rate,self.total_epoch,train_loader,data_valid_x,data_valid_y,"./results/lstm_1d.h5")
        self.is_fitted_ = True
        return self
    
    def predict(self,X):
        # predicting area from here!
        check_is_fitted(self,"is_fitted_")
        data,self.scaler,self.scaler1 = nibio_data_transform(X, None)
        data = self.scaler1.transform(data)

        # TODO: Create predictions based on the test sets
        pred, _, _, _ = create_predictions(self.model, data,self.scaler)

        return self.scaler.inverse_transform(pred).flatten()


# def ILSTM_train(raw_data, target_label,total_epoch = 50,hidden_size=16,lerningrate=1e-3, lead_time=1, seq_length=24, batch_size=16):
#     data,scaler,scaler1 = ILSTM.nibio_data_transform(raw_data, target_label)
#     data = scaler1.transform(data)

#     # TODO: Generate the tensor for lstm model

#     [data_x, data_y,data_z] = ILSTM.LSTMDataGenerator(data, lead_time, batch_size, seq_length)

#        # concat all variables.
#     # TODO: Flexible valid split
#     data_train_x=data_x[:int((data_x.shape[0])-400*24)]
#     data_train_y = data_y[:int(data_x.shape[0]-400*24)]

#     train_data = Data.TensorDataset(data_train_x, data_train_y)
#     train_loader = Data.DataLoader(
#         dataset=train_data,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0
#     )

#     data_valid_x=data_x[int(data_x.shape[0]-400*24):int(data_x.shape[0]-365*24)] # -> trener 35 dager
#     data_valid_y=data_y[int(data_x.shape[0]-400*24):int(data_x.shape[0]-365*24)] # -> tester 35 dager
#     data_test_x=data_x[int(data_x.shape[0]-365*24):int(1.0 * data_x.shape[0])] # -> validerer på resterende
#     #data_testd_z=data_z[int(data_x.shape[0]-365*24):int(1.0 * data_x.shape[0])] # -> stat på rest

#     # TODO: Flexible input shapes and optimizer
#     # IMVTensorLSTM,IMVFullLSTM
#     model = ILSTM.ILSTM_SV(data_x.shape[2],data_x.shape[1], 1, hidden_size)
#     if torch.cuda.is_available():
#         model = model.cuda()
#     # TODO: Trian LSTM based on the training and validation sets
#     model,predicor_import,temporal_import=ILSTM.train_lstm(model,lerningrate,total_epoch,train_loader,data_valid_x,data_valid_y,"./saved_models/lstm_1d.h5")

#     # TODO: Create predictions based on the test sets
#     pred, mulit_FV_aten, predicor_import,temporal_import = ILSTM.create_predictions(model, data_test_x,scaler)
#     # TODO: Computer score of R2 and RMSE
#     print(pred)

#     return pred
    


# Example usage
if __name__ == "__main__":
    # testing area
    pass
