#!/usr/bin/env python
# coding: utf-8

# # Data visualisation
# 
# We start by importing the data

# In[1]:


import sklearn
import datetime
#import numba

import os
import copy
import pickle

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import sklearn.model_selection
import statsmodels as sm
import torch.utils.data as Data

#from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation 
#from tensorflow.keras.layers import MaxPooling2D, Dropout, Conv2DTranspose
#from tensorflow.keras.layers import concatenate, Concatenate
#from tensorflow.keras.models import Model
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras import metrics

#sklearn → model trening
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics         import accuracy_score, mean_squared_error, r2_score, mean_absolute_error

#sklearn → data treatment
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


#from ILSTM_Soil_model_main import lstm_interprety_soil_moisture as ILSTM
from My_tools import DataFileLoader as DFL # min egen
from My_tools import StudyEstimators as SE
# path definitions

ROOT = "../../"

RESULT_PATH = ROOT + "results/"

DATA_PATH = ROOT + "data/"
PLOT_PATH = RESULT_PATH + "plots/"
TABLE_PATH = RESULT_PATH + "tables/"
OTHER_PATH = RESULT_PATH + "other/"

METADATA_PRELOAD_DATA_PATH = OTHER_PATH + "bin_data/"

DATA_INFO = DATA_PATH + "info/"
DATA_INFO_NIBIO_FILE = DATA_INFO  + "lmt.nibio.csv"
DATA_INFO_FROST_FILE = DATA_INFO + "Frost_stations.csv"
DATA_INFO_NIBIO2FROST_FILE = DATA_INFO + "StationIDInfo.csv"
DATA_FILE_SOIL_STATIONS = DATA_INFO + "'Stasjonsliste jordtemperatur modellering.xlsx'"

DATA_COLLECTION = DATA_PATH + "raw_data/"
DATA_COLLECTION_STAT = DATA_COLLECTION + "Veret paa Aas 2013- 2017/" # pattern -> 'Veret paa Aas 2013- 2017/Veret paa Aas {YYYY}.pdf'
DATA_COLLECTION_TIME = DATA_COLLECTION + "Time 2013- 2023/" # pattern -> Time{YYYY}.xlsx
DATA_COLLECTION_NIBIO = DATA_COLLECTION + "nibio/" # pattern -> weather_data_hour_stID{id}_y{year}.csv
DATA_COLLECTION_MET = DATA_COLLECTION + "MET/" # pattern -> StationTo_{id}_FROM_{FrostID}.csv

# ID definitions
station_names = pd.read_csv(DATA_INFO_NIBIO_FILE,
                          header=0,
                          index_col = "ID")

nibio_id = {
    "Innlandet" : ["11","17","26","27"],
    "Trøndelag" : ["15","57","34","39"],
    "Østfold" : ["37","41","52","118"],
    "Vestfold" : ["30","38","42","50"] # Fjern "50" for å se om bedre resultat
}


# Loading data from folders

# ## Function definitions

# In[2]:

from typing import Any
from collections.abc import Iterable

def show_plot(data,plot_kwarg):
    """
        plots timeseries, assumes dataframe with a 'Time' columns
    """
    for d in range(len(data)):
        if d not in plot_kwarg:
            plt.plot(data[d].Time, data[d].iloc[:,data[d].columns != "Time"])
        else:
            plt.plot(data[d].Time, data[d].iloc[:,data[d].columns != "Time"],**plot_kwarg[d])
    
    if "xlabel" in plot_kwarg: 
        plt.xlabel = plot_kwarg["xlabel"]
    else:
        plt.xlabel = "Time"
        
    if "ylabel" in plot_kwarg: 
        plt.ylabel = plot_kwarg["ylabel"]
    else:
        plt.ylabel = "celsius degrees ℃"

#@numba.njit
def stat_model(y_true: Iterable ,y_pred: Iterable) -> dict[str,float]: 
    """
        Returns a dict with following statitics
        - SSE
        - SST
        - SAE
        - --R^2--
        - bias
        - n

        unscaled: to unscale the relevant metric by |y| (R^2,bias)
    """
    stats = {
        "SSE": ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64),
        "SAE": np.abs(y_true - y_pred).sum(axis=0, dtype=np.float64),
        "bias": (y_pred - y_true).sum(axis=0, dtype=np.float64),
        "SST": ((y_true - np.average(y_true, axis=0)) ** 2).sum(axis=0, dtype=np.float64),
        "n": y_true.shape[0]
    }
    return stats


# In[3]:


force_load = False
if force_load:
    nibio_data_ungroup = DFL.DataFileLoader(DATA_COLLECTION_NIBIO,r"weather_data_hour_stID(\d{1,3})_y(\d{4}).csv",_iter_key = True)
    nibio_data_ungroup.load_data(names = ["Time","TM","RR","TJM10","TJM20"])
    nibio_data = nibio_data_ungroup.group_layer(nibio_id)

    nibio_data_raw_ungroup = DFL.DataFileLoader(DATA_COLLECTION_NIBIO,r"weather_data_raw_hour_stID(\d{1,3})_y(\d{4}).csv",_iter_key = True)
    nibio_data_raw_ungroup.load_data(names = ["Time","TM","RR","TJM10","TJM20"])
    nibio_data_raw = nibio_data_raw_ungroup.group_layer(nibio_id)

    frost_raw_ungroup = DFL.DataFileLoader(DATA_COLLECTION_MET,r"weather_data_raw_hour_stID(\d{1,3})_y(\d{4}).csv",_iter_key = True)

    def dataframe_merge_func(x,y):
        y.iloc[y.iloc[:,1].notna() & (y.iloc[:,1] <= 0),2] = pd.NA
        x.iloc[0:y.shape[0],2] = y.iloc[0:y.shape[0],2]
        return x

    imputed_nibio_data = nibio_data.combine(nibio_data_raw,merge_func = dataframe_merge_func)
    imputed_nibio_data.dump(METADATA_PRELOAD_DATA_PATH + "weatherdata.bin")

    del nibio_data, nibio_data_raw, frost_raw_ungroup, nibio_data_raw_ungroup, nibio_data_ungroup
else: 
    imputed_nibio_data = DFL.DataFileLoader().load(METADATA_PRELOAD_DATA_PATH + "weatherdata_cleaned.bin")

terskel_data = pd.read_csv(TABLE_PATH + "na_run_count_simp.csv",delimiter=";")
terskel = int(next(t.split(">")[-1] for t in terskel_data.columns if ">" in t))


# In[4]:


for regi in nibio_id.keys(): 
    show_plot([station.loc[:,["Time","TJM20"]] for station in imputed_nibio_data[regi,:].shave_top_layer().merge_layer(level=1).flatten()],{})
    plt.legend(nibio_id[regi])
    plt.title("Område: {}, feature: {}".format(regi,"TJM20"))
    plt.show()


def all_permute(L):
    """
        makes a list of size 2^len(L)-1 with all combinations
    """
    from itertools import permutations
    final_list = list(L)
    for n in range(2,len(L)+1): 
        final_list.extend(set(permutations(L,n)))
    return final_list

def mediant(x: float,y: float):
    """
        Takes the mediant of two fractions
            a/b + c/d = (a+c)/(b+d)
    """
    frac_x = x.as_integer_ratio()
    frac_y = y.as_integer_ratio()
    comb_xy = (frac_x[0] + frac_y[0],frac_x[1] + frac_y[1])
    return comb_xy[0]/comb_xy[1]

def combine_years(X,Y):
    """
        Combines two dataframes
    """
    if isinstance(X,list) or isinstance(Y,list):
        pass
    if X.index == Y.index:
        return [X,Y]

#@numba.njit
def find_non_nan_ranges(df: pd.Series) -> list[tuple[int,int]]:
    """
    Finds the ranges of indexes where rows do not contain NaNs in the DataFrame.
    Assumes there is a 'Time' column with timestamps.

    Args:
        df (pd.DataFrame): Input DataFrame with NaNs.

    Returns:
        list of tuples: List of (start, end) index ranges where rows do not contain NaNs.
    """

    # Initialize variables
    non_nan_ranges = []
    start_idx = None

    # Iterate over rows
    for idx, row in df.items():
        if not(np.isnan(row)):
            # If the row does not contain NaNs
            if start_idx is None:
                # If this is the start of a new range
                start_idx = idx
        else:
            # If the row contains NaNs
            if start_idx is not None:
                # If this is the end of a range
                non_nan_ranges.append((start_idx, idx - 1))
                start_idx = None

    # Check if the last range is still open
    if start_idx is not None:
        non_nan_ranges.append((start_idx, df.index[-1]))

    return non_nan_ranges


# ### Plauborg regression
# 
# Author Plauborg used the above model to predict soil temperature, but used previus time to make the model more time dependent and fourier terms to reflect changes during the year.

# In[6]:

#@numba.njit
def model_traning_testing(datafile,base_model,parameters,feature_target,min_length):
    nibio_id = {
        "Innlandet" : ["11","17","26","27"],
        "Trøndelag" : ["15","57","34","39"],
        "Østfold" : ["37","41","52","118"],
        "Vestfold" : ["30","38","42","50"] # Fjern "50" for å se om bedre resultat
    }
    def calc_stat_from_data(y_true: Iterable ,x_true: Iterable,model: Any ,s_model_stat: dict[str,Any] = dict()) -> dict[str,Any]:
        current_stat = stat_model(y_true,model.predict(x_true))
        for metric in current_stat: 
            s_model_stat.setdefault(metric,0)
            s_model_stat[metric] += current_stat[metric]
        return s_model_stat
    
    base_model_stats = {
        "global":{},
        "region":{},
        "station":{}
    }
    def merge_func(left: list[Any, ...] | Any, right: list[Any, ...] | Any) -> list[Any,...]:
                if isinstance(left, list) or isinstance(right, list):
                    if not isinstance(left, list):
                        left = [left]
                    if not isinstance(right, list):
                        right = [right]
                    # Concatenate the lists
                    combined_list = left + right
                else:
                    # Create a new list with left and right as elements
                    combined_list = [left, right]
                return combined_list

    global_model = copy.deepcopy(base_model)
    for regi in nibio_id.keys():
        for station in nibio_id[regi]:
            print(regi,station)
            data = datafile[regi,station,"2014":"2020"].shave_top_layer().merge_layer(level = 1,merge_func = merge_func).flatten() # looks at all previus years including this year
            test = datafile[regi,station,"2021":"2022"].shave_top_layer().merge_layer(level = 1,merge_func = merge_func).flatten() # looks at the next year
            test = [t.infer_objects(copy=False) for t in test]
            for t in test:
                t.loc[t["TM"].isna() | t[feature_target].isna(),["TM",feature_target]] = np.nan
            t = test # too lazy
            # First we fetch region (regi), all stations (:), then relevant years ("2014":str(i)). Since we only look at one region at the time
            # we remove the root group (shave_top_layer()), then we merge the years (merge_layer(level = 1), level 1 since level 0 would be the stations at this point)
            # then make a list (flatten(), the default handeling is to put leafs in a list)
            for d in data: # fitting model with all stations
                    d = d.infer_objects(copy=False)
                    d.loc[d["TM"].isna() | d[feature_target].isna(),["TM",feature_target]] = np.nan
                    for dt in find_non_nan_ranges(d[feature_target]):
                        if dt[1]-dt[0] < min_length:
                            continue
                        global_model.fit(d.loc[dt[0]:dt[1],parameters],d.loc[dt[0]:dt[1],feature_target])
            collected_test = []
            for yr in range(len(test)):
                for deltaT in find_non_nan_ranges(test[yr][feature_target]):
                    collected_test.append((yr, *deltaT ))  
            for dt in collected_test: 
                    if dt[2]-dt[1] < min_length:
                        continue
                    base_model_stats["global"] = calc_stat_from_data(t[dt[0]].loc[dt[1]:dt[2],feature_target].to_numpy(),t[dt[0]].loc[dt[1]:dt[2],parameters],global_model,s_model_stat=base_model_stats["global"])
                    base_model_stats["region"][regi] = calc_stat_from_data(t[dt[0]].loc[dt[1]:dt[2],feature_target].to_numpy(),t[dt[0]].loc[dt[1]:dt[2],parameters],global_model,s_model_stat=base_model_stats["region"].setdefault(regi,dict()))
                    base_model_stats["station"][station] = calc_stat_from_data(t[dt[0]].loc[dt[1]:dt[2],feature_target].to_numpy(),t[dt[0]].loc[dt[1]:dt[2],parameters],global_model,s_model_stat=base_model_stats["station"].setdefault(station,dict()))
    base_model_stats["global"]["model"] = global_model
    return base_model_stats


# In[9]:


#parameters = ["TM"]
#feature_target = "TJM20"
#min_length = 24 # minimum number of rows used in sequense

#base_model = LinearRegression()

#def time2float(x):
#    if "Time" in x.columns:
#        x["Time"] = x["Time"].transform({"Time":lambda x: x.day_of_year*24 + x.hour})
#    return x

#adapted_data = imputed_nibio_data#.data_transform(time2float)

#lin_reg_stats_20 = model_traning_testing(
#    datafile = adapted_data,
#    base_model = base_model,
#    parameters = parameters,
#    feature_target = feature_target,
#    min_length = min_length
#)
#pickle.dump(lin_reg_stats_20,f := open(METADATA_PRELOAD_DATA_PATH + "lin_stat_20.bin","wb"))
#f.close()

#parameters = ["TM"]
#feature_target = "TJM10"
#min_length = 24 # minimum number of rows used in sequense

#base_model = LinearRegression()

#def time2float(x):
#    if "Time" in x.columns:
#        x["Time"] = x["Time"].transform({"Time":lambda x: x.day_of_year*24 + x.hour})
#    return x

#adapted_data = imputed_nibio_data#.data_transform(time2float)

#lin_reg_stats_10 = model_traning_testing(
#    datafile = adapted_data,
#    base_model = base_model,
#    parameters = parameters,
#    feature_target = feature_target,
#    min_length = min_length
#)

# In[ ]:


#pickle.dump(lin_reg_stats_10,f := open(METADATA_PRELOAD_DATA_PATH + "lin_stat_10.bin","wb"))
#f.close()


#del lin_reg_stats_10, lin_reg_stats_20 # removes unesseserry data
# In[ ]:

## ------------------ Plauborg -------------------------

parameters = ["Time","TM"]
feature_target = "TJM20"
min_length = 24 # minimum number of rows used in sequense
search_area_day = {"lag_max":range(2,15,2), "fourier_sin_length":range(2,15,2),"fourier_cos_length":range(2,15,2),"is_day" : [True]}
#search_area = {"lag_max":range(2,15,2), "fourier_sin_length":range(2,15,2),"fourier_cos_length":range(2,15,2)}
#base_model = GridSearchCV(SE.PlauborgRegresson(),param_grid=search_area,n_jobs = -1)

#Plauborg_stats_20 = model_traning_testing(
#    datafile = imputed_nibio_data,
#    base_model = base_model,
#    parameters = parameters,
#    feature_target = feature_target,
#    min_length = min_length
#)

#pickle.dump(Plauborg_stats_20,f := open(METADATA_PRELOAD_DATA_PATH + "Plauborg_stat_20.bin","wb"))
#f.close()
base_model = GridSearchCV(SE.PlauborgRegresson(),param_grid=search_area_day)

#def hour2day(df):
#    hourly_df = df.infer_objects(copy=False).set_index("Time")[["TM","TJM10", "TJM20"]].resample("1D").mean().ffill().reset_index()  # Forward fill missing values
#    return hourly_df

augmented_nibio_data = imputed_nibio_data.data_transform(hour2day)

Plauborg_daily_stats_20 = model_traning_testing(
    datafile = augmented_nibio_data,
    base_model = base_model,
    parameters = parameters,
    feature_target = feature_target,
    min_length = min_length
)

pickle.dump(Plauborg_daily_stats_20,f := open(METADATA_PRELOAD_DATA_PATH + "Plauborg_day_stat_20.bin","wb"))
f.close()

#parameters = ["Time","TM"]
feature_target = "TJM10"
#min_length = 24 # minimum number of rows used in sequense
#base_model = GridSearchCV(SE.PlauborgRegresson(),param_grid=search_area,n_jobs = -1)

#Plauborg_stats_10 = model_traning_testing(
#    datafile = imputed_nibio_data,
#    base_model = base_model,
#    parameters = parameters,
#    feature_target = feature_target,
#    min_length = min_length
#)
#pickle.dump(Plauborg_stats_10,f := open(METADATA_PRELOAD_DATA_PATH + "Plauborg_stat_10.bin","wb"))
#f.close()

base_model = GridSearchCV(SE.PlauborgRegresson(),param_grid=search_area_day,n_jobs = -1)

Plauborg_daily_stats_10 = model_traning_testing(
    datafile = augmented_nibio_data,
    base_model = base_model,
    parameters = parameters,
    feature_target = feature_target,
    min_length = min_length
)

# In[8]:

pickle.dump(Plauborg_daily_stats_10,f := open(METADATA_PRELOAD_DATA_PATH + "Plauborg_day_stat_10.bin","wb"))
f.close()

del Plauborg_daily_stats_10, Plauborg_daily_stats_20 # removes unesseserry data


## -------------------------------------- BiLSTM ---------------------------------------------

parameters = ["Time","TM"]
feature_target = "TJM20"
min_length = 200 # minimum number of rows used in sequense
search_area = {"input_shape":[12*n for n in range(2,14,4)],"lstm_units":[2**k for k in range(3,10,2)],"epochs":[2*n for n in range(5,15,3)]}
base_model = GridSearchCV(SE.KerasBiLSTM(),param_grid=search_area,pre_dispatch = 5,n_jobs = -1)

KerasBiLSTM_stats_20 = model_traning_testing(
    datafile = imputed_nibio_data,
    base_model = base_model,
    parameters = parameters,
    feature_target = feature_target,
    min_length = min_length
)
pickle.dump(KerasBiLSTM_stats_20,f := open(METADATA_PRELOAD_DATA_PATH + "KerasBiLSTM_stat_20.bin","wb"))
f.close()

feature_target = "TJM10"
KerasBiLSTM_stats_10 = model_traning_testing(
    datafile = imputed_nibio_data,
    base_model = base_model,
    parameters = parameters,
    feature_target = feature_target,
    min_length = min_length
)


# In[ ]:

pickle.dump(KerasBiLSTM_stats_10,f := open(METADATA_PRELOAD_DATA_PATH + "KerasBiLSTM_stat_10.bin","wb"))
f.close()

del KerasBiLSTM_stats_10, KerasBiLSTM_stats_20 # removes unesseserry data

# In[ ]:


# ---------------------------- ILSTM ----------------------------------------

parameters = ["Time","TM"]
feature_target = "TJM20"
min_length = 12*14+2 # minimum number of rows used in sequense
search_area = {"input_shape":[12*n for n in range(2,14,4)],"lstm_units":[2**k for k in range(3,15,2)],"epochs":[2*n for n in range(5,15,3)]}
base_model = GridSearchCV(SE.ILSTM(),param_grid=search_area,n_jobs = -1)

ILSTM_stats_20 = model_traning_testing(
    datafile = imputed_nibio_data,
    base_model = base_model,
    parameters = parameters,
    feature_target = feature_target,
    min_length = min_length
)
pickle.dump(ILSTM_stats_20,f := open(METADATA_PRELOAD_DATA_PATH + "ILSTM_stat_20.bin","wb"))
f.close()

feature_target = "TJM10"
ILSTM_stats_10 = model_traning_testing(
    datafile = imputed_nibio_data,
    base_model = base_model,
    parameters = parameters,
    feature_target = feature_target,
    min_length = min_length
)


# In[ ]:

pickle.dump(ILSTM_stats_10,f := open(METADATA_PRELOAD_DATA_PATH + "ILSTM_stat_10.bin","wb"))
f.close()

del ILSTM_stats_10, ILSTM_stats_20 # removes unesseserry data

