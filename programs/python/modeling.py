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

# from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation 
# from tensorflow.keras.layers import MaxPooling2D, Dropout, Conv2DTranspose
# from tensorflow.keras.layers import concatenate, Concatenate
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras import metrics

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
def find_non_nan_ranges(df):
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
nibio_id = {
        "Innlandet" : ["11","17","26","27"],
        "Trøndelag" : ["15","57","34","39"],
        "Østfold" : ["37","41","52","118"],
        "Vestfold" : ["30","38","42","50"] # Fjern "50" for å se om bedre resultat
    }
#@numba.njit
def model_traning_testing(datafile,base_model,parameters,feature_target,min_length):
    #progess_file = open(OTHER_PATH + "M_{}.txt".format(model_name := type(base_model)),"a+")
    
    base_model_stats = {
        "global":{},
    }

    #progess_file.write("[{} : {}] Begun training\n".format(datetime.datetime.now(),model_name))
    global_model = copy.deepcopy(base_model)
    all_data = None
    for key,yr in datafile:
            #progess_file.write("[{} : {}] started {}\n".format(datetime.datetime.now(),model_name,key))
            print("[{}] started {}\n".format(datetime.datetime.now(),key))
            if all_data is None:
                all_data = yr.dropna(subset = ["TM","TJM10","TJM20"],how = "any")
            else:
                all_data = pd.concat([all_data, yr.dropna(subset = ["TM","TJM10","TJM20"],how = "any")],axis = 0, ignore_index=True)
            #data = datafile[regi,station,"2014":"2020"].shave_top_layer().merge_layer(level = 1,merge_func = merge_func).flatten() # looks at all previus years including this year
            # First we fetch region (regi), all stations (:), then relevant years ("2014":str(i)). Since we only look at one region at the time
            # we remove the root group (shave_top_layer()), then we merge the years (merge_layer(level = 1), level 1 since level 0 would be the stations at this point)
            # then make a list (flatten(), the default handeling is to put leafs in a list)
            #d = yr.infer_objects(copy=False)
            #d.loc[d["TM"].isna() | d[feature_target].isna(),["TM",feature_target]] = np.nan
            #for dt in find_non_nan_ranges(d[feature_target]):
            #    if dt[1]-dt[0] < min_length:
            #        continue
            #    global_model.fit(d.loc[dt[0]:dt[1],parameters],d.loc[dt[0]:dt[1],feature_target])
            #progess_file.write("[{} : {}] ended {}\n".format(datetime.datetime.now(),model_name,key))
    #progess_file.write("[{} : {}] Ended training\n".format(datetime.datetime.now(),model_name))
    #progess_file.close()
    global_model.fit(all_data.loc[:,parameters],all_data.loc[:,feature_target]])
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

# parameters = ["Time","TM"]
# feature_target = "TJM20"
# min_length = 24 # minimum number of rows used in sequense
# #search_area_day = {"lag_max":range(0,15,2), "fourier_sin_length":range(0,15,2),"fourier_cos_length":range(0,15,2),"is_day" : [True]}
# search_area = {"lag_max":range(0,15,2), "fourier_sin_length":range(0,15,2),"fourier_cos_length":range(0,15,2),"fit_intercept":[False]}
# base_model = GridSearchCV(SE.PlauborgRegresson(fit_intercept=False),param_grid=search_area,n_jobs = 3)

# Plauborg_stats_20 = model_traning_testing(
#    datafile = imputed_nibio_data,
#    base_model = base_model,
#    parameters = parameters,
#    feature_target = feature_target,
#    min_length = min_length
# )
# pickle.dump(Plauborg_stats_20,f := open(METADATA_PRELOAD_DATA_PATH + "Plauborg_stat_20.bin","wb"))
# f.close()
#base_model = GridSearchCV(SE.PlauborgRegresson(),param_grid=search_area_day, n_jobs = 3)

#def hour2day(df):
#    hourly_df = df.infer_objects(copy=False).set_index("Time")[["TM","TJM10", "TJM20"]].resample("1D").mean().ffill().reset_index()  # Forward fill missing values
#    return hourly_df

#augmented_nibio_data = imputed_nibio_data.data_transform(hour2day)
#
#Plauborg_daily_stats_20 = model_traning_testing(
#    datafile = augmented_nibio_data,
#    base_model = base_model,
#    parameters = parameters,
#    feature_target = feature_target,
#    min_length = min_length
#)

#pickle.dump(Plauborg_daily_stats_20,f := open(METADATA_PRELOAD_DATA_PATH + "Plauborg_day_stat_20.bin","wb"))
#f.close()

# parameters = ["Time","TM"]
# feature_target = "TJM10"
# min_length = 24 # minimum number of rows used in sequense
# base_model = GridSearchCV(SE.PlauborgRegresson(fit_intercept=False),param_grid=search_area,n_jobs = 3)

# Plauborg_stats_10 = model_traning_testing(
#    datafile = imputed_nibio_data,
#    base_model = base_model,
#    parameters = parameters,
#    feature_target = feature_target,
#    min_length = min_length
# )
# pickle.dump(Plauborg_stats_10,f := open(METADATA_PRELOAD_DATA_PATH + "Plauborg_stat_10.bin","wb"))
# f.close()

#base_model = GridSearchCV(SE.PlauborgRegresson(),param_grid=search_area_day,n_jobs = 3)
#
#Plauborg_daily_stats_10 = model_traning_testing(
#    datafile = augmented_nibio_data,
#    base_model = base_model,
#    parameters = parameters,
#    feature_target = feature_target,
#    min_length = min_length
#)

# In[8]:

#pickle.dump(Plauborg_daily_stats_10,f := open(METADATA_PRELOAD_DATA_PATH + "Plauborg_day_stat_10.bin","wb"))
#f.close()

#del Plauborg_stats_10, Plauborg_stats_20#,Plauborg_daily_stats_10, Plauborg_daily_stats_20 # removes unesseserry data
# del Plauborg_stats_20,Plauborg_stats_10

## -------------------------------------- BiLSTM ---------------------------------------------

def plot_model_performance(
    data,
    available_stat,
    feature = ["Time","TM"],
    probing_year = "2022",
    target = "TJM10",
    figure_alpha = 0.5,
    point_size = 0.5,
    name = "lin_stat_10",
    lead_time = 0,
    back_to_font = True
):
    model2check = available_stat[name]["global"]
    target_slice = slice(lead_time-1,None,None) if back_to_font else slice(0,-lead_time,None)
    for region in nibio_id.keys():
        data_lin_list = dict(data[region,:,probing_year].shave_top_layer().flatten(return_key = True))
        #big_fig = plt.Figure()
        fig,axs = plt.subplots(2,2)
        gridspec = axs[1, 0].get_subplotspec().get_gridspec()
        for a in axs[1,:]:
            a.remove()
        subfig = fig.add_subplot(gridspec[1, :])
        used_indexes = []
        for i,st in data_lin_list.items(): # going over all stations
            data_lin = st.loc[:,[*feature,target]].dropna(how="any")
            data_time = st.loc[data_lin.index,"Time"]
            if data_lin.count().sum() <= 0:
                continue 
            used_indexes.append(i) # appends station number
            #print(data_lin["TM"].to_numpy().reshape(-1, 1)[2500:2520],data_lin[target].to_numpy()[2500:2520],model2check["model"].predict(data_lin["TM"].to_numpy().reshape(-1, 1))[2500:2520])
            axs[0,0].scatter(x = data_lin["TM"].to_numpy().reshape(-1, 1),y = data_lin[target].to_numpy(),
                             s = point_size,
                             linewidth=0,
                             alpha = figure_alpha,
                             label = i)
            axs[0,1].scatter(x = model2check["model"].predict(data_lin[feature]),y = data_lin[target].to_numpy()[target_slice],
                             s = point_size,
                             linewidth=0,
                             alpha = figure_alpha,
                             label = i)
            subfig.plot(data_time[target_slice],model2check["model"].predict(data_lin[feature])-data_lin[target].to_numpy()[target_slice],
                        alpha = figure_alpha,
                        label = i)
            subfig.set_ylim(ymin = -10, ymax = 10)
            axs[0,0].set_xlim(xmin = -10, xmax = 30)
            axs[0,1].set_xlim(xmin = -10, xmax = 30)
            axs[0,0].set_ylim(ymin = -10, ymax = 30)
            axs[0,1].set_ylim(ymin = -10, ymax = 30)

        axs[0,0].set_ylabel("℃ True target ({})".format(target),labelpad = 0.5)
        axs[0,0].set_title("TM vs {}".format(target))
        axs[0,1].set_title("true {tr} vs model {tr}".format(tr = target))
        subfig.set_ylabel("∆℃")
        subfig.set_xlabel("Time [h]")
        subfig.set_xticks(subfig.get_xticks())
        subfig.set_xticklabels(subfig.get_xticklabels(), rotation=45)
        new_Laxis = axs[0,1].secondary_yaxis(location = "right")
        new_Laxis.set_yticks([])
        new_Laxis.set_ylabel("℃ Predicted",loc="bottom",labelpad = 3.0)
        print("{} with data from stations in {} in year {}, target {}".format(name,region,probing_year,target))
        #big_fig.subtitle("{} with data from stations in {} in year {}".format(name,region,probing_year))
        fig.legend(labels=used_indexes,markerscale=10,title="stations")
        plt.savefig(PLOT_PATH+"diffplot_{}_{}_{}_{}.pdf".format(name,region,probing_year,target))
        plt.clf()

def model_prosent_accurasy(y_true,y_pred,epsilon = 0.5):
    """
        calculates the prosentage of the predicted values actually falls within acceptibale range (epsilon)
    """
    diff = (y_pred.flatten() - y_true.flatten()) <= epsilon
    total = diff.sum() / diff.shape[0]
    return total

from numpy.random import uniform
def model_log_condition_number(arr,mod,epsilon = 0.01,iter = 100):
    #data_arr = arr.copy()
    #if "Time" in data_arr.columns and type(data_arr["Time"][0]) is pd.Timestamp:
    #    data_arr["Time"] = arr["Time"].transform({"Time":lambda x: x.day_of_year*24 + x.hour})
    cond_num = 0.000000000001
    base_pred = mod.predict(arr)
    norm_pred = np.linalg.norm(base_pred)
    norm_arr = np.linalg.norm((arr.drop("Time",axis=1) if "Time" in arr.columns else arr).infer_objects(copy = False))
    delta = np.sqrt(epsilon / len(arr))
    for _ in range(iter):
        delta_arr = uniform(low = -delta,high = delta, size = (len(arr),(arr.shape[1]-1 if arr.shape[1] > 1 else 1),*arr.shape[2:])) 
        norm_delta = np.linalg.norm(delta_arr)
        shifted_arr = (arr.drop("Time",axis=1) if "Time" in arr.columns else arr).infer_objects(copy = False) + delta_arr
        new_pred = mod.predict((pd.concat([arr["Time"],shifted_arr],axis=1) if "Time" in arr.columns else shifted_arr))
        new_cond = norm_arr*np.linalg.norm(new_pred - base_pred)/(norm_pred * norm_delta)
        if new_cond > cond_num:
            cond_num = new_cond
    return np.log(cond_num)

def append_stat_from_stat(y_true,y_pred,x_true,model_stat,model):
    model_stat["log_cond"] = model_log_condition_number(x_true,model,iter = 20)  
    model_stat["digit_sense"] = int(model_stat["log_cond"]) + (1 if int(model_stat["log_cond"]) > 0 else -1)
    model_stat["R^2"] = 1 - model_stat["SSE"]/model_stat["SST"]
    model_stat["MSE"] = model_stat["SSE"] / model_stat["n"]
    model_stat["MAE"] = model_stat["SAE"] / model_stat["n"]
    model_stat["pros_acc"] = model_prosent_accurasy(y_true,y_pred,epsilon = 0.5)
    model_stat["RMSE"] = np.sqrt(model_stat["MSE"])
    model_stat["bias"] = model_stat["bias"] / model_stat["n"]
    model_stat["adj R^2"] = 1 - (model_stat["SSE"]/model_stat["SST"] * (model_stat["n"] - 1)/(model_stat["n"] - 1 - x_true.shape[1]) )
    return model_stat

def stat_model(y_true ,y_pred) -> dict[str,float]: 
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
    #print(y_true)
    #print(y_pred)
    stats = {
        "SSE": ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64),
        "SAE": np.abs(y_true - y_pred).sum(axis=0, dtype=np.float64),
        "bias": (y_pred - y_true).sum(axis=0, dtype=np.float64),
        "SST": ((y_true - np.average(y_true, axis=0)) ** 2).sum(axis=0, dtype=np.float64),
        "n": y_true.shape[0]
    }
    return stats

def calc_stat_from_data(y_true ,x_true,model ,s_model_stat = dict()):
        current_stat = stat_model(y_true,model.predict(x_true))
        #print(current_stat)
        for metric in current_stat: 
            s_model_stat.setdefault(metric,0)
            s_model_stat[metric] += current_stat[metric]
        return s_model_stat
    

def recalc_stat(data: DFL.DataFileLoader,features: list[str], target: str, model, exclustion = [], lead_time = 0, back_to_front = True, min_length = 1,old_model_stat: dict = dict()):
    """
        Assume data is grouped as wanted, the new groups are 'global', region (index 0), station (index 1). Index 2 is the year
    """
    master_stat = {
        "global": dict(),
        "region": dict(),
        "local": dict()
    }
    target_indexing = slice(lead_time-1,None,None) if back_to_front else slice(0,-lead_time,None)
    y_pred = np.array([])
    y_ground = np.array([])
    for path in data:
        if any( x in exclustion for x in path[0] ):
            continue
        #print(path)
        df = path[1]
        df.loc[df[[*features,target]].isna().any(axis=1),[*features,target]] = np.nan
        collected_indexes = []
        for dt in find_non_nan_ranges(path[1][target]):
            if dt[1] - dt[0] < min_length:
                continue
            collected_indexes.extend(range(dt[0],dt[1]+1))
            y_pred = np.append(y_pred,model.predict(path[1].loc[dt[0]:dt[1],features]))
            y_ground = np.append(y_ground,path[1].loc[dt[0]:dt[1],target][target_indexing])
            master_stat['global'] = calc_stat_from_data(path[1].loc[dt[0]:dt[1],target].to_numpy().flatten()[target_indexing],
                                                        path[1].loc[dt[0]:dt[1],features],
                                                        model,
                                                        master_stat["global"])
            master_stat['region'][path[0][0]] = calc_stat_from_data(path[1].loc[dt[0]:dt[1],target].to_numpy().flatten()[target_indexing],
                                                                    path[1].loc[dt[0]:dt[1],features],
                                                                    model,
                                                                    master_stat['region'].get(path[0][0],dict()))
            master_stat['local'][path[0][1]] = calc_stat_from_data(path[1].loc[dt[0]:dt[1],target].to_numpy().flatten()[target_indexing],
                                                                   path[1].loc[dt[0]:dt[1],features],
                                                                   model,
                                                                   master_stat['local'].get(path[0][1],dict()))
            #print("input_len :",dt[1]-dt[0],"ground_len :",len(y_ground),"pred_len :",len(y_pred))
    for scope in master_stat.keys():
        if scope == "global":
            master_stat["global"] = append_stat_from_stat(y_ground,
                                                       y_pred,
                                                       path[1].loc[collected_indexes,features],
                                                       master_stat["global"],
                                                       model
                                                      )
            master_stat["global"]["model"] = model
            continue
        for st in master_stat[scope]:
            master_stat[scope][st] = append_stat_from_stat(y_ground,
                                                       y_pred,
                                                       path[1].loc[collected_indexes,features],
                                                       master_stat[scope][st],
                                                       model
                                                      )
    return master_stat

# ------------------------------- BiLSTM ----------------------------------

parameters = ["Time","TM"]
feature_target = "TJM20"
min_length = 2*24 + 10 # minimum number of rows used in sequense
search_area = {"input_shape":[n for n in range(24,7*24+1,24)],"epochs":[6,10]}
base_model = GridSearchCV(SE.l1KerasBiLSTM(),param_grid=search_area,pre_dispatch = 5,n_jobs = -1,cv=2)

l1KerasBiLSTM_stats_20 = model_traning_testing(
     datafile = imputed_nibio_data[:,:,"2014":"2020"],
     base_model = base_model,
     parameters = parameters,
     feature_target = feature_target,
     min_length = min_length
)

l1KerasBiLSTM_stats_20["global"]["model"].best_estimator_.model.save(METADATA_PRELOAD_DATA_PATH + "l1KerasBiLSTM_stat_20.keras")
#open(METADATA_PRELOAD_DATA_PATH + "KerasBiLSTM_stat_20.json","w").write(KerasBiLSTM_stats_20["global"]["model"].best_estimator_.model.to_json()).close()

#print(imputed_nibio_data["Innlandet","11","2021":"2022"].flatten()[0])

# compute stat then save without model
print("Finished training")
l1KerasBiLSTM_stats_20 = recalc_stat(imputed_nibio_data[:,:,"2021":"2022"],
                            ["Time","TM"],
                            feature_target,
                            l1KerasBiLSTM_stats_20["global"]["model"],
                            min_length=min_length,
                            lead_time=l1KerasBiLSTM_stats_20["global"]["model"].best_estimator_.input_shape)
l1KerasBiLSTM_stats_20 = {
    "KerasBiLSTM_stats_20":l1KerasBiLSTM_stats_20
}
plot_model_performance(imputed_nibio_data,
                       l1KerasBiLSTM_stats_20,
                       ["Time","TM"],
                        probing_year="2022",
                        target=feature_target,
                        lead_time=l1KerasBiLSTM_stats_20["l1KerasBiLSTM_stats_20"]["global"]["model"].best_estimator_.input_shape,
                        name="l1KerasBiLSTM_stats_20"
                    )
plot_model_performance(imputed_nibio_data,
                       l1KerasBiLSTM_stats_20,
                       ["Time","TM"],
                        probing_year="2021",
                        target=feature_target,
                        lead_time=l1KerasBiLSTM_stats_20["l1KerasBiLSTM_stats_20"]["global"]["model"].best_estimator_.input_shape,
                        name="l1KerasBiLSTM_stats_20"
                    )
l1KerasBiLSTM_stats_20["l1KerasBiLSTM_stats_20"]["global"]["model"].best_estimator_.model = None

pickle.dump(l1KerasBiLSTM_stats_20,f := open(METADATA_PRELOAD_DATA_PATH + "l1KerasBiLSTM_stat_20.bin","wb"))
f.close()
del l1KerasBiLSTM_stats_20

feature_target = "TJM10"

l1KerasBiLSTM_stats_10 = model_traning_testing(
     datafile = imputed_nibio_data[:,:,"2014":"2020"],
     base_model = base_model,
     parameters = parameters,
     feature_target = feature_target,
     min_length = min_length
)

l1KerasBiLSTM_stats_10["global"]["model"].best_estimator_.model.save(METADATA_PRELOAD_DATA_PATH + "l1KerasBiLSTM_stat_10.keras")
#open(METADATA_PRELOAD_DATA_PATH + "KerasBiLSTM_stat_20.json","w").write(KerasBiLSTM_stats_20["global"]["model"].best_estimator_.model.to_json()).close()

#print(imputed_nibio_data["Innlandet","11","2021":"2022"].flatten()[0])

# compute stat then save without model
print("Finished training")
l1KerasBiLSTM_stats_10 = recalc_stat(imputed_nibio_data[:,:,"2021":"2022"],
                            ["Time","TM"],
                            feature_target,
                            l1KerasBiLSTM_stats_10["global"]["model"],
                            min_length=min_length,
                            lead_time=l1KerasBiLSTM_stats_10["global"]["model"].best_estimator_.input_shape)
l1KerasBiLSTM_stats_10 = {
    "l1KerasBiLSTM_stats_20":l1KerasBiLSTM_stats_10
}
plot_model_performance(imputed_nibio_data,
                       l1KerasBiLSTM_stats_10,
                       ["Time","TM"],
                        probing_year="2022",
                        target=feature_target,
                        lead_time=l1KerasBiLSTM_stats_10["l1KerasBiLSTM_stats_10"]["global"]["model"].best_estimator_.input_shape,
                        name="l1KerasBiLSTM_stats_20"
                    )
plot_model_performance(imputed_nibio_data,
                       l1KerasBiLSTM_stats_10,
                       ["Time","TM"],
                        probing_year="2021",
                        target=feature_target,
                        lead_time=l1KerasBiLSTM_stats_10["l1KerasBiLSTM_stats_10"]["global"]["model"].best_estimator_.input_shape,
                        name="l1KerasBiLSTM_stats_10"
                    )
l1KerasBiLSTM_stats_10["l1KerasBiLSTM_stats_10"]["global"]["model"].best_estimator_.model = None

pickle.dump(l1KerasBiLSTM_stats_10,f := open(METADATA_PRELOAD_DATA_PATH + "l1KerasBiLSTM_stat_10.bin","wb"))
f.close()
del l1KerasBiLSTM_stats_10


# ------------------------------- LSTM --------------------------------

parameters = ["Time","TM"]
feature_target = "TJM20"
min_length = 2*24 + 10 # minimum number of rows used in sequense
search_area = {"input_shape":[n for n in range(24,7*24+1,24)],"epochs":[6,10]}
#search_area = {"input_shape":[n for n in range(12,24,12)],"lstm_units":[2**k for k in range(5,7,2)],"epochs":[n for n in range(1,4,2)]}
base_model = GridSearchCV(SE.l2KerasBiLSTM(),param_grid=search_area,pre_dispatch = 5,n_jobs = -1,cv=2)

l2KerasBiLSTM_stats_20 = model_traning_testing(
     datafile = imputed_nibio_data[:,:,"2014":"2020"],
     base_model = base_model,
     parameters = parameters,
     feature_target = feature_target,
     min_length = min_length
)

l2KerasBiLSTM_stats_20["global"]["model"].best_estimator_.model.save(METADATA_PRELOAD_DATA_PATH + "l2KerasBiLSTM_stat_20.keras")
#open(METADATA_PRELOAD_DATA_PATH + "KerasBiLSTM_stat_20.json","w").write(KerasBiLSTM_stats_20["global"]["model"].best_estimator_.model.to_json()).close()

#print(imputed_nibio_data["Innlandet","11","2021":"2022"].flatten()[0])

# compute stat then save without model
print("Finished training")
l2KerasBiLSTM_stats_20 = recalc_stat(imputed_nibio_data[:,:,"2021":"2022"],
                            ["Time","TM"],
                            feature_target,
                            l2KerasBiLSTM_stats_20["global"]["model"],
                            min_length=min_length,
                            lead_time=l2KerasBiLSTM_stats_20["global"]["model"].best_estimator_.input_shape)
l2KerasBiLSTM_stats_20 = {
    "l2KerasBiLSTM_stats_20":l2KerasBiLSTM_stats_20
}
plot_model_performance(imputed_nibio_data,
                       l2KerasBiLSTM_stats_20,
                       ["Time","TM"],
                        probing_year="2022",
                        target=feature_target,
                        lead_time=l2KerasBiLSTM_stats_20["l2KerasBiLSTM_stats_20"]["global"]["model"].best_estimator_.input_shape,
                        name="l2KerasBiLSTM_stats_20"
                    )
plot_model_performance(imputed_nibio_data,
                       l2KerasBiLSTM_stats_20,
                       ["Time","TM"],
                        probing_year="2021",
                        target=feature_target,
                        lead_time=l2KerasBiLSTM_stats_20["l2KerasBiLSTM_stats_20"]["global"]["model"].best_estimator_.input_shape,
                        name="l2KerasBiLSTM_stats_20"
                    )
l2KerasBiLSTM_stats_20["l2KerasBiLSTM_stats_20"]["global"]["model"].best_estimator_.model = None

pickle.dump(l2KerasBiLSTM_stats_20,f := open(METADATA_PRELOAD_DATA_PATH + "l2KerasBiLSTM_stat_20.bin","wb"))
f.close()
del l2KerasBiLSTM_stats_20

feature_target = "TJM10"

l2KerasBiLSTM_stats_10 = model_traning_testing(
     datafile = imputed_nibio_data[:,:,"2014":"2020"],
     base_model = base_model,
     parameters = parameters,
     feature_target = feature_target,
     min_length = min_length
)

l2KerasBiLSTM_stats_10["global"]["model"].best_estimator_.model.save(METADATA_PRELOAD_DATA_PATH + "l2KerasBiLSTM_stat_10.keras")
#open(METADATA_PRELOAD_DATA_PATH + "KerasBiLSTM_stat_20.json","w").write(KerasBiLSTM_stats_20["global"]["model"].best_estimator_.model.to_json()).close()

#print(imputed_nibio_data["Innlandet","11","2021":"2022"].flatten()[0])

# compute stat then save without model
print("Finished training")
l2KerasBiLSTM_stats_10 = recalc_stat(imputed_nibio_data[:,:,"2021":"2022"],
                            ["Time","TM"],
                            feature_target,
                            l2KerasBiLSTM_stats_10["global"]["model"],
                            min_length=min_length,
                            lead_time=l2KerasBiLSTM_stats_10["global"]["model"].best_estimator_.input_shape)
l2KerasBiLSTM_stats_10 = {
    "l2KerasBiLSTM_stats_20":l2KerasBiLSTM_stats_10
}
plot_model_performance(imputed_nibio_data,
                       l2KerasBiLSTM_stats_10,
                       ["Time","TM"],
                        probing_year="2022",
                        target=feature_target,
                        lead_time=l2KerasBiLSTM_stats_10["l2KerasBiLSTM_stats_10"]["global"]["model"].best_estimator_.input_shape,
                        name="l2KerasBiLSTM_stats_20"
                    )
plot_model_performance(imputed_nibio_data,
                       l2KerasBiLSTM_stats_10,
                       ["Time","TM"],
                        probing_year="2021",
                        target=feature_target,
                        lead_time=l2KerasBiLSTM_stats_10["l2KerasBiLSTM_stats_10"]["global"]["model"].best_estimator_.input_shape,
                        name="l2KerasBiLSTM_stats_10"
                    )
l2KerasBiLSTM_stats_10["l2KerasBiLSTM_stats_10"]["global"]["model"].best_estimator_.model = None

pickle.dump(l2KerasBiLSTM_stats_10,f := open(METADATA_PRELOAD_DATA_PATH + "l2KerasBiLSTM_stat_10.bin","wb"))
f.close()
del l2KerasBiLSTM_stats_10

# _------------------------------- modBiLSTM ----------------------------

parameters = ["Time","TM"]
feature_target = "TJM20"
min_length = 2*24 + 10 # minimum number of rows used in sequense
search_area = {"input_shape":[n for n in range(24,7*24+1,24)],"lstm_units":[2**6,2**7],"epochs":[6,10]}
#search_area = {"input_shape":[n for n in range(12,24,12)],"lstm_units":[2**k for k in range(5,7,2)],"epochs":[n for n in range(1,4,2)]}
base_model = GridSearchCV(SE.KerasBiLSTM(),param_grid=search_area,pre_dispatch = 5,n_jobs = -1,cv=2)

KerasBiLSTM_stats_20 = model_traning_testing(
     datafile = imputed_nibio_data[:,:,"2014":"2020"],
     base_model = base_model,
     parameters = parameters,
     feature_target = feature_target,
     min_length = min_length
)

KerasBiLSTM_stats_20["global"]["model"].best_estimator_.model.save(METADATA_PRELOAD_DATA_PATH + "KerasBiLSTM_stat_20.keras")
#open(METADATA_PRELOAD_DATA_PATH + "KerasBiLSTM_stat_20.json","w").write(KerasBiLSTM_stats_20["global"]["model"].best_estimator_.model.to_json()).close()

#print(imputed_nibio_data["Innlandet","11","2021":"2022"].flatten()[0])

# compute stat then save without model
print("Finished training")
KerasBiLSTM_stats_20 = recalc_stat(imputed_nibio_data[:,:,"2021":"2022"],
                            ["Time","TM"],
                            "TJM20",
                            KerasBiLSTM_stats_20["global"]["model"],
                            min_length=min_length,
                            lead_time=KerasBiLSTM_stats_20["global"]["model"].best_estimator_.input_shape)
KerasBiLSTM_stats_20 = {
    "KerasBiLSTM_stats_20":KerasBiLSTM_stats_20
}
plot_model_performance(imputed_nibio_data,
                       KerasBiLSTM_stats_20,
                       ["Time","TM"],
                        probing_year="2022",
                        target="TJM20",
                        lead_time=KerasBiLSTM_stats_20["KerasBiLSTM_stats_20"]["global"]["model"].best_estimator_.input_shape,
                        name="KerasBiLSTM_stats_20"
                    )
plot_model_performance(imputed_nibio_data,
                       KerasBiLSTM_stats_20,
                       ["Time","TM"],
                        probing_year="2021",
                        target="TJM20",
                        lead_time=KerasBiLSTM_stats_20["KerasBiLSTM_stats_20"]["global"]["model"].best_estimator_.input_shape,
                        name="KerasBiLSTM_stats_20"
                    )
KerasBiLSTM_stats_20["KerasBiLSTM_stats_20"]["global"]["model"].best_estimator_.model = None

pickle.dump(KerasBiLSTM_stats_20,f := open(METADATA_PRELOAD_DATA_PATH + "KerasBiLSTM_stat_20.bin","wb"))
f.close()
del KerasBiLSTM_stats_20

feature_target = "TJM10"

KerasBiLSTM_stats_10 = model_traning_testing(
     datafile = imputed_nibio_data[:,:,"2014":"2020"],
     base_model = base_model,
     parameters = parameters,
     feature_target = feature_target,
     min_length = min_length
)

KerasBiLSTM_stats_10["global"]["model"].best_estimator_.model.save(METADATA_PRELOAD_DATA_PATH + "KerasBiLSTM_stat_10.keras")
#open(METADATA_PRELOAD_DATA_PATH + "KerasBiLSTM_stat_20.json","w").write(KerasBiLSTM_stats_20["global"]["model"].best_estimator_.model.to_json()).close()

#print(imputed_nibio_data["Innlandet","11","2021":"2022"].flatten()[0])

# compute stat then save without model
print("Finished training")
KerasBiLSTM_stats_10 = recalc_stat(imputed_nibio_data[:,:,"2021":"2022"],
                            ["Time","TM"],
                            "TJM20",
                            KerasBiLSTM_stats_10["global"]["model"],
                            min_length=min_length,
                            lead_time=KerasBiLSTM_stats_10["global"]["model"].best_estimator_.input_shape)
KerasBiLSTM_stats_10 = {
    "KerasBiLSTM_stats_20":KerasBiLSTM_stats_10
}
plot_model_performance(imputed_nibio_data,
                       KerasBiLSTM_stats_10,
                       ["Time","TM"],
                        probing_year="2022",
                        target="TJM20",
                        lead_time=KerasBiLSTM_stats_10["KerasBiLSTM_stats_10"]["global"]["model"].best_estimator_.input_shape,
                        name="KerasBiLSTM_stats_20"
                    )
plot_model_performance(imputed_nibio_data,
                       KerasBiLSTM_stats_10,
                       ["Time","TM"],
                        probing_year="2021",
                        target="TJM20",
                        lead_time=KerasBiLSTM_stats_10["KerasBiLSTM_stats_10"]["global"]["model"].best_estimator_.input_shape,
                        name="KerasBiLSTM_stats_10"
                    )
KerasBiLSTM_stats_10["KerasBiLSTM_stats_10"]["global"]["model"].best_estimator_.model = None

pickle.dump(KerasBiLSTM_stats_10,f := open(METADATA_PRELOAD_DATA_PATH + "KerasBiLSTM_stat_10.bin","wb"))
f.close()
del KerasBiLSTM_stats_10

# ## -------------------------------------- modBiLSTM ---------------------------------------------

# feature_target = "TJM20"
# base_model = GridSearchCV(SE.modKerasBiLSTM(),param_grid=search_area,pre_dispatch = 5,n_jobs = -1,cv=2)

# modKerasBiLSTM_stats_20 = model_traning_testing(
#      datafile = imputed_nibio_data["Innlandet","11","2014":"2015"],
#      base_model = base_model,
#      parameters = parameters,
#      feature_target = feature_target,
#      min_length = min_length
# )

# modKerasBiLSTM_stats_20["global"]["model"].best_estimator_.model.save(METADATA_PRELOAD_DATA_PATH + "modKerasBiLSTM_stat_20.keras")
# #open(METADATA_PRELOAD_DATA_PATH + "KerasBiLSTM_stat_20.json","w").write(KerasBiLSTM_stats_20["global"]["model"].best_estimator_.model.to_json()).close()

# #print(imputed_nibio_data["Innlandet","11","2021":"2022"].flatten()[0])

# # compute stat then save without model
# print("Finished training")
# modKerasBiLSTM_stats_20 = recalc_stat(imputed_nibio_data["Innlandet",:,"2021":"2022"],
#                             ["Time","TM"],
#                             "TJM20",
#                             modKerasBiLSTM_stats_20["global"]["model"],
#                             lead_time=modKerasBiLSTM_stats_20["global"]["model"].best_estimator_.input_shape)
# modKerasBiLSTM_stats_20 = {
#     "modKerasBiLSTM_stats_20":modKerasBiLSTM_stats_20
# }
# plot_model_performance(imputed_nibio_data,
#                        modKerasBiLSTM_stats_20,
#                        ["Time","TM"],
#                         probing_year="2022",
#                         target="TJM20",
#                         lead_time=modKerasBiLSTM_stats_20["modKerasBiLSTM_stats_20"]["global"]["model"].best_estimator_.input_shape,
#                         name="modKerasBiLSTM_stats_20"
#                     )

# plot_model_performance(imputed_nibio_data,
#                        modKerasBiLSTM_stats_20,
#                        ["Time","TM"],
#                         probing_year="2021",
#                         target="TJM20",
#                         lead_time=modKerasBiLSTM_stats_20["modKerasBiLSTM_stats_20"]["global"]["model"].best_estimator_.input_shape,
#                         name="modKerasBiLSTM_stats_20"
#                     )

# #KerasBiLSTM_stats_20["KerasBiLSTM_stats_20"]["global"]["model"].best_estimator_.model = None

# #pickle.dump(KerasBiLSTM_stats_20,f := open(METADATA_PRELOAD_DATA_PATH + "KerasBiLSTM_stat_20.bin","wb"))
# #f.close()
# del modKerasBiLSTM_stats_20

# feature_target = "TJM10"
# modKerasBiLSTM_stats_10 = model_traning_testing(
#      datafile = imputed_nibio_data["Innlandet","11","2014":"2015"],
#      base_model = base_model,
#      parameters = parameters,
#      feature_target = feature_target,
#      min_length = min_length
# )
# modKerasBiLSTM_stats_10["global"]["model"].best_estimator_.model.save(METADATA_PRELOAD_DATA_PATH + "modKerasBiLSTM_stat_10.keras")

# modKerasBiLSTM_stats_10 = recalc_stat(imputed_nibio_data["Innlandet",:,"2021":"2022"],
#                             ["Time","TM"],
#                             "TJM10",
#                             modKerasBiLSTM_stats_10["global"]["model"],
#                             lead_time=modKerasBiLSTM_stats_10["global"]["model"].best_estimator_.input_shape)
# modKerasBiLSTM_stats_10 = {
#     "modKerasBiLSTM_stats_10":modKerasBiLSTM_stats_10
# }
# plot_model_performance(imputed_nibio_data,
#                        modKerasBiLSTM_stats_10,
#                        ["Time","TM"],
#                         probing_year="2022",
#                         target="TJM10",
#                         lead_time=modKerasBiLSTM_stats_10["modKerasBiLSTM_stats_10"]["global"]["model"].best_estimator_.input_shape,
#                         name="modKerasBiLSTM_stats_10"
#                     )

# plot_model_performance(imputed_nibio_data,
#                        modKerasBiLSTM_stats_10,
#                        ["Time","TM"],
#                         probing_year="2021",
#                         target="TJM10",
#                         lead_time=modKerasBiLSTM_stats_10["modKerasBiLSTM_stats_10"]["global"]["model"].best_estimator_.input_shape,
#                         name="modKerasBiLSTM_stats_10"
#                     )

# del modKerasBiLSTM_stats_10
# ---------------------------- ILSTM ----------------------------------------
# parameters = ["Time","TM"]
# feature_target = "TJM20"
# min_length = 12*14+2 # minimum number of rows used in sequense
# search_area = {"seq_length":[12*n for n in range(2,7,4)],"hidden_size":[2**k for k in range(3,4,2)],"learning_rate":[0.1,0.5,0.7],"total_epoch":[n for n in range(1,5,3)]}
# base_model = GridSearchCV(SE.ILSTM(),param_grid=search_area,n_jobs = -1,cv=2)

# ILSTM_stats_20 = model_traning_testing(
#    datafile = imputed_nibio_data,
#    base_model = base_model,
#    parameters = parameters,
#    feature_target = feature_target,
#    min_length = min_length
# )
# pickle.dump(ILSTM_stats_20,f := open(METADATA_PRELOAD_DATA_PATH + "ILSTM_stat_20.bin","wb"))
# f.close()

# feature_target = "TJM10"
# ILSTM_stats_10 = model_traning_testing(
#    datafile = imputed_nibio_data,
#    base_model = base_model,
#    parameters = parameters,
#    feature_target = feature_target,
#    min_length = min_length
# )

# pickle.dump(ILSTM_stats_10,f := open(METADATA_PRELOAD_DATA_PATH + "ILSTM_stat_10.bin","wb"))
# f.close()

# del ILSTM_stats_10, ILSTM_stats_20 # removes unesseserry data

# -------------------------- modBiLSTM ---------------------------------------

# parameters = ["Time","TM"]
# feature_target = "TJM20"
# min_length = 200 # minimum number of rows used in sequense
# search_area = {"input_shape":[n for n in range(12,2*7*24,18)],"lstm_units":[2**k for k in range(2,4,2)],"epochs":[n for n in range(1,5,2)]}
# base_model = GridSearchCV(SE.modKerasBiLSTM(),param_grid=search_area,pre_dispatch = 2,n_jobs = -1,cv=2)

# modKerasBiLSTM_stats_20 = model_traning_testing(
#      datafile = imputed_nibio_data,
#      base_model = base_model,
#      parameters = parameters,
#      feature_target = feature_target,
#      min_length = min_length
# )
# modKerasBiLSTM_stats_20["global"]["model"].best_estimator_.model.save(METADATA_PRELOAD_DATA_PATH + "modKerasBiLSTM_stat_20.keras")
# #pickle.dump(modKerasBiLSTM_stats_20,f := open(METADATA_PRELOAD_DATA_PATH + "modKerasBiLSTM_stat_20.bin","wb"))
# #f.close()
# del modKerasBiLSTM_stats_20

# feature_target = "TJM10"
# modKerasBiLSTM_stats_10 = model_traning_testing(
#      datafile = imputed_nibio_data,
#      base_model = base_model,
#      parameters = parameters,
#      feature_target = feature_target,
#      min_length = min_length
# )

# modKerasBiLSTM_stats_20["global"]["model"].best_estimator_.model.save(METADATA_PRELOAD_DATA_PATH + "modKerasBiLSTM_stat_10.keras")
# # In[ ]:

# #pickle.dump(modKerasBiLSTM_stats_10,f := open(METADATA_PRELOAD_DATA_PATH + "modKerasBiLSTM_stat_10.bin","wb"))
# #f.close()

# del modKerasBiLSTM_stats_10 # removes unesseserry data
