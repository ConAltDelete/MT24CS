import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from keras.models import Sequential
from keras.layers import LSTM, Dense

class KerasLSTMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_shape, num_classes, lstm_units=64, epochs=10, batch_size=32):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.model = Sequential()
        self.model.add(LSTM(self.lstm_units, input_shape=self.input_shape))
        self.model.add(Dense(self.num_classes, activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X):
        check_is_fitted(self, 'model')
        X = check_array(X)
        return np.argmax(self.model.predict(X), axis=1)

class PlauborgRegresson(LinearRegression):
    def __init__(self, *, param=1):
        self.param = param
    def fit(self, X, y=None):
        self.is_fitted_ = True
        new_X = self.F_plauborg(X)
        new_y = y
        super().fit(new_X,new_y)
        return self
    def F_plauborg(self,df):
        """
            Fxn is based on a full year while df could have any range.
        """
        new_df = df.set_index("Time")

        data_ret = pd.DataFrame({"B0":new_df.loc[:,"TM"].values},columns = ["B"+str(i) for i in range(13)]+ ["FS" + str(i) for i in range(1,13)] + ["FC" + str(i) for i in range(1,13)])

        for i in range(1,13): # 1,2
            data_ret.loc[:,"B"+str(i)] = new_df.loc[:,"TM"].shift(i).values
            data_ret.loc[:,"FS"+str(i)] = np.sin(2*np.pi/(365*24) * ( new_df.index.day*24 + new_df.index.hour) * i)
            data_ret.loc[:,"FC"+str(i)] = np.cos(2*np.pi/(365*24) * ( new_df.index.day*24 + new_df.index.hour ) * i)
        return data_ret

class RankinRegresson(LinearRegression):
    def __init__(self, *, param=1):
        self.param = param
    def fit(self, X, y=None):
        self.Tdiff = X["Time"].diff().fillna(0)
        self.dt = pd.infer_freq(X.Time)
        alpha_t_bot = (X.TM.iloc[0] - 2*X.TJM10.iloc[0] + X.TJM20.iloc[0]) / (0.1)**2
        alpha_t_top = (X.TM.iloc[0] - X.TJM20.iloc[0]) / (2*self.dt)
        alpha_t = alpha_t_top / alpha_t_bot
        self.soilDamp = alpha_t/(2*0.15)**2
        if "snow" in X.columns: # assumes constatn TM at infinity and convergence
            self._is_snow = True
            k = self.dt*self.soilDamp
            D = -self.fs*X["snow"]
            self.T_init = ((k*np.exp(D))/(1+np.exp(D)*(k-1)))*X["TM"].mean
        else:
            d = np.sqrt(2*alpha_t/self.dt)
            w = 2*np.pi / (365*24)
            self.T_init =  X["TM"].iloc[0:24*3].mean + np.abs(np.max(np.abs(X["TM"].iloc[0:(24*3)])) - X["TM"].iloc[0:(24*3)].mean)*np.exp(-0.15/d)*np.sin(w*(X.Time.iloc[0].dayofyear*24 + X.Time.iloc[0].hour)-0.15/d)
        
        self.is_fitted_ = True
        return self
    def predict(self, X):
        T_z = np.zeros(X.shape[0])
        T_z[0] = self.T_init
        
        for t in range(X.shape[0]-1):
            T_z[t+1] = T_z[t] + self.Tdiff[t+1]*self.soilDamp*(X["TM"].iloc[t]-T_z[t])
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

class KerasLSTMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_shape, num_classes, lstm_units=64, epochs=10, batch_size=32):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.model = Sequential()
        self.model.add(LSTM(self.lstm_units, input_shape=self.input_shape))
        self.model.add(Dense(self.num_classes, activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X):
        check_is_fitted(self, 'model')
        X = check_array(X)
        return np.argmax(self.model.predict(X), axis=1)

# Example usage
if __name__ == "__main__":
    # testing area
    pass
