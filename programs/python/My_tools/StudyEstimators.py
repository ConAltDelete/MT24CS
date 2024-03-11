import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Input
from tensorflow.keras import Input
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.utils import timeseries_dataset_from_array


class KerasBiLSTM(MLPRegressor):
    def __init__(self, input_shape = None, num_classes = 1, lstm_units=None, epochs=None, batch_size = 32):
        super().__init__()
        self.input_shape: int = input_shape
        self.num_classes: int = num_classes
        self.lstm_units: int = lstm_units
        self.epochs: int = epochs
        self.model = None
        self.spaceing: int = 1
        self.batch_size: int = batch_size

        self.scaler_x = None
        self.scaler_y = None

    def fit(self, X, y):
        X, y = check_X_y(X, y) # Checks if values are finite and not too sparce.
        # Data treatment
        All_data = self._data_treatment(X,y) # Takes both just incase.
        # Setting up model
        self.model = Sequential()
        self.model.add(Input((All_data[0][0].shape[1], All_data[0][0].shape[2])))
        self.model.add(Bidirectional(LSTM(self.lstm_units)))
        self.model.add(Dense(self.num_classes, activation='softmax'))
        self.model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error','r2_score'])
        print(self.model.output_shape)
        # fitting model
        #print(self.model.summary())
        #print("Input shape for LSTM:", All_data[0][0].shape)

        self.model.fit(All_data, epochs=self.epochs, verbose=1)

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

        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        new_X = self.scaler_x.fit_transform(X).astype("float32")
        if y is not None:
            new_y = self.scaler_y.fit_transform(y.reshape(-1, 1)).astype("float32")
        else:
            new_y = None
        
        #print("Shape of new_X:", new_X.shape)
        #print("Shape of new_y:", new_y.shape)

        self.transformed_data = self._data_generate(new_X,new_y)

        return self.transformed_data

    def _data_generate(self,X,y):
        transformed_data = TimeseriesGenerator(np.flip(X,axis=0), np.flip(y,axis=0), # Due to spaceing we remove the first target values so we get the relavant target
                               length=self.input_shape, stride=self.spaceing,
                               batch_size=self.batch_size)

        return transformed_data

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        #print(X.shape)
        trans_X = timeseries_dataset_from_array(np.flip(self.scaler_x.transform(X).astype("float32"),axis=0),targets = None,
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
        return trans_y
    
    def score(self,X,y, sample_weight=None):
        """
            Modified score function that removes the first target values.
        """
        return super().score(X = X,
                            y = np.flip(y[self.input_shape-1::self.spaceing],axis=0),
                            sample_weight=sample_weight
                            )

class PlauborgRegresson(LinearRegression):

    def __init__(self):
        self.lag_max = 2
        self.fourier_sin_length = 2
        self.fourier_cos_length = 2

        super().__init__()

    def fit(self, X, y=None):
        self.is_fitted_ = True
        new_X = self.F_plauborg(X)
        new_y = y
        return super().fit(new_X,new_y)
    def F_plauborg(self,df):
        """
            Fxn is based on a full year while df could have any range.
        """

        new_df = df.set_index("Time")

        data_ret = pd.DataFrame({"B0":new_df.loc[:,"TM"].values},
                                columns = ["B"+str(i) for i in range(self.lag_max)] + \
                                        ["FS" + str(i) for i in range(1,self.fourier_sin_length + 1)] + \
                                        ["FC" + str(i) for i in range(1,self.fourier_cos_length + 1)])

        for i in range(1,self.fourier_sin_length): # 1,2
            data_ret.loc[:,"B"+str(i)] = new_df.loc[:,"TM"].shift(i).values
            data_ret.loc[:,"FS"+str(i)] = np.sin(2*np.pi/(365*24) * ( new_df.index.day*24 + new_df.index.hour) * i)
            data_ret.loc[:,"FC"+str(i)] = np.cos(2*np.pi/(365*24) * ( new_df.index.day*24 + new_df.index.hour ) * i)
        return data_ret.infer_objects(copy=False).fillna(0)
    
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

class GAN(BaseEstimator, TransformerMixin):
    def __init__(self, num_epochs=100, batch_size=64, learning_rate=0.0002):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.generator = None  # Initialize your generator network
        self.discriminator = None  # Initialize your discriminator network

    def fit(self, X, y=None):
        # Training loop for GAN
        for epoch in range(self.num_epochs):
            for _ in range(len(X) // self.batch_size):
                # Generate random noise (latent vectors)
                noise = np.random.randn(self.batch_size, latent_dim)  # Replace latent_dim with your desired dimension

                # Generate fake samples using the generator
                generated_samples = self.generator.predict(noise)

                # Real samples (from your dataset)
                real_samples = X[np.random.randint(0, len(X), self.batch_size)]

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(real_samples, np.ones((self.batch_size, 1)))
                d_loss_fake = self.discriminator.train_on_batch(generated_samples, np.zeros((self.batch_size, 1)))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Train the generator
                g_loss = self.combined.train_on_batch(noise, np.ones((self.batch_size, 1)))

            # Print progress (optional)
            print(f"Epoch {epoch}/{self.num_epochs} - D Loss: {d_loss[0]} - G Loss: {g_loss}")

        return self

    def transform(self, X):
        # Generate synthetic data using the trained GAN
        noise = np.random.randn(len(X), latent_dim)
        synthetic_data = self.generator.predict(noise)

        return synthetic_data



# Example usage
if __name__ == "__main__":
    # testing area
    pass
