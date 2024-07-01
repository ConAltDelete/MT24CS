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

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

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
    start_idx = None

    # Iterate over rows
    for idx, row in df.iterrows():
        if row.notna().all():
            # If the row does not contain NaNs
            if start_idx is None:
                # If this is the start of a new range
                start_idx = idx
        else:
            # If the row contains NaNs
            if start_idx is not None:
                # If this is the end of a range
                yield (start_idx, idx - 1)
                start_idx = None

    # Check if the last range is still open
    if start_idx is not None:
        yield (start_idx, df.index[-1])


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




def model_traning_testing(datafile,base_model,parameters,feature_target,min_length):
    #progess_file = open(OTHER_PATH + "M_{}.txt".format(model_name := type(base_model)),"a+")
    
    base_model_stats = {
        "global":{},
    }

    # --------------- make data -------------------
    #progess_file.write("[{} : {}] Begun training\n".format(datetime.datetime.now(),model_name))
    global_model = copy.deepcopy(base_model)
    all_data_X = []
    all_data_y = []
    for key,yr in datafile:
            #progess_file.write("[{} : {}] started {}\n".format(datetime.datetime.now(),model_name,key))
            print("[{}] started {}\n".format(datetime.datetime.now(),key))

            #if np.nan not in yr and yr.shape[0] >= min_length:
            #    all_data_X.append(yr.loc[:,parameters])
            #    all_data_y.append(yr.loc[:,feature_target])
            #    continue
            #print(pd.concat([yr.loc[:,parameters],yr.loc[:,feature_target]],axis=1,ignore_index=True))
            for dt in find_non_nan_ranges(pd.concat([yr.loc[:,parameters],yr.loc[:,feature_target]],axis=1,ignore_index=True)):
                if dt[1]-dt[0] < min_length:
                    continue
                new_frame = yr.iloc[dt[0]:dt[1],:]
                all_data_X.append(new_frame.loc[:,parameters])
                all_data_y.append(new_frame.loc[:,feature_target])
    
    # --------------- eval parameters -------------------

    #print(all_data_X,all_data_y)
    global_model.fit(all_data_X,all_data_y)
    base_model_stats["global"]["model"] = global_model
    return base_model_stats

def confidence_ellipse(x, y, ax, n_std=1.0,set_label=False, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    source: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor,**kwargs)

    if set_label:
        ellipse.set_label("⋌₀ = {}".format(round(min(ell_radius_x,ell_radius_y),2)))

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

from matplotlib.colors import Colormap
def plot_predground_eclipse(
    data,
    available_stat,
    feature = ["Time","TM"],
    target = "TJM10",
    figure_alpha = 0.5,
    point_size = 0.5,
    name = "lin_stat_10"
):

    model2check = available_stat[name]["global"]
    fig, ax = plt.subplots()
    for i,region in enumerate(nibio_id.keys()):
        all_data = None
        for key,yr in data[region]:
            if all_data is None:
                all_data = yr.dropna(subset = ["TM","TJM10","TJM20"],how = "any")
            else:
                all_data = pd.concat([all_data, yr.dropna(subset = ["TM","TJM10","TJM20"],how = "any")],axis = 0, ignore_index=True)
        data_lin = all_data.loc[:,[*feature,target]].dropna(how="any")
        #data_time = st.loc[data_lin.index,"Time"]
        if data_lin.count().sum() <= 0:
           continue 
        y_pred = model2check["model"].predict(data_lin[feature])
        ax.scatter(x = y_pred[::12],y = data_lin[target].to_numpy()[:y_pred.shape[0]:12],
                             s = point_size,
                             linewidth=0,
                             alpha = figure_alpha,
                             label = region)
    all_data = None
    for key,yr in data:
            if all_data is None:
                all_data = yr.dropna(subset = ["TM","TJM10","TJM20"],how = "any")
            else:
                all_data = pd.concat([all_data, yr.dropna(subset = ["TM","TJM10","TJM20"],how = "any")],axis = 0, ignore_index=True)
    data_lin = all_data.loc[:,[*feature,target]].dropna(how="any")
    y_pred = model2check["model"].predict(data_lin[feature])
    a,b = np.polyfit(x = y_pred, y = data_lin[target].to_numpy()[:y_pred.shape[0]],deg=1)
    ax.plot(np.linspace(-5,25,num=3),a*np.linspace(-5,25,num=3)+b,label="{}x+{}".format(np.round(a,2),np.round(b,2)))
    ax.plot(np.linspace(-5,25,num=3),np.linspace(-5,25,num=3),"--",label = "Symmetry line")
    for k in range(1,4):
        confidence_ellipse(y_pred, data_lin[target].to_numpy()[:y_pred.shape[0]], ax, n_std=float(k),edgecolor=["red","green","blue"][k-1],set_label = (k == 2))
    #plt.title("{} prediction accuracy with\nconfidence eclipses for 68%, 95% and 99%".format(model2name[name]))
    plt.grid(True)
    plt.xlim((-5,25))
    plt.ylim((-5,25))
    plt.xlabel("predicted values [℃]")
    plt.ylabel("true values [℃]")
    plt.legend(markerscale=10)
    plt.savefig(PLOT_PATH + "conf_elips_" + name + ".pdf")
    plt.show()

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
    if lead_time is not None:
            target_slice = slice(lead_time-1,None,None) if back_to_font else slice(0,-lead_time,None)
    else:
            target_slice = slice(0,None,None)
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
        plt.close()

def model_prosent_accurasy(y_true,y_pred,epsilon = 0.5):
    """
        calculates the prosentage of the predicted values actually falls within acceptibale range (epsilon)
    """
    diff = np.abs(y_pred.flatten() - y_true.flatten()) <= epsilon
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
    

def recalc_stat(data: DFL.DataFileLoader,features: list[str], target: str, model, exclustion = [], append_hist = False, lead_time = 0, back_to_front = True, min_length = 1,old_model_stat: dict = dict()):
    """
        Assume data is grouped as wanted, the new groups are 'global', region (index 0), station (index 1). Index 2 is the year
    """
    master_stat = {
        "global": dict(),
        "region": dict(),
        "local": dict()
    }
    if lead_time is not None:
        target_indexing = slice(lead_time-1,None,None) if back_to_front else slice(0,-lead_time,None)
    else:
        target_indexing = slice(0,None,None)
    y_pred = np.array([])
    y_ground = np.array([])
    for path in data:
        if any( x in exclustion for x in path[0] ):
            continue
        #print(path)
        df = path[1]
        df.loc[df[[*features,target]].isna().any(axis=1),[*features,target]] = np.nan
        collected_indexes = []
        for dt in find_non_nan_ranges(pd.concat([path[1].loc[:,features],path[1].loc[:,target]],axis=1,ignore_index=True)):
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
            master_stat["global"]["best_params"] = model.best_params_
            continue
        for st in master_stat[scope]:
            master_stat[scope][st] = append_stat_from_stat(y_ground,
                                                       y_pred,
                                                       path[1].loc[collected_indexes,features],
                                                       master_stat[scope][st],
                                                       model
                                                      )
    if append_hist:
        master_stat["global"]["history"] = model.best_estimator_.history
    return master_stat

# -------------------- Modeling area ----------------------

# -------------------- own messure ------------------------

def own_scorer(estimator,X,y, mean = True):
    r"""Costum scorer that scores with a balance between $R^2$ and MSE.
        $$
            scorer = \left{\frac{MSE/R^2,R^2 > 0}{-\infty, else}\right.
        $$
    """
    score = 0.0

    if isinstance(X,list):
        n_samples = 0
        for TX, Ty in zip(X,y):
            n_samples += TX.shape[0]
            Tscore = own_scorer(estimator=estimator,X=TX,y=Ty, mean = False)
            if np.isinf(Tscore):
                return Tscore
            score -= Tscore
        return -score/n_samples

    if "input_shape" in estimator.__dict__ or "lag_max" in estimator.__dict__: # adjust y
        if "input_shape" in estimator.__dict__:
            new_y = y[:-(estimator.input_shape-1)]
        elif "lag_max" in estimator.__dict__:
            new_y = y[:-(estimator.lag_max-1)]
        else:
            raise ValueError("Can't find relevant shape in estimator.")
        top = mean_squared_error(new_y,estimator.predict(X))
        bot = r2_score(new_y,estimator.predict(X))
        if bot <= 0:
            return np.NINF
        else:
            score += top/bot
        
    else: # don't adjust y
        # accuracy_score, mean_squared_error, r2_score, mean_absolute_error
        top = mean_squared_error(y,estimator.predict(X))
        bot = r2_score(y,estimator.predict(X))
        if bot <= 0:
            return np.NINF
        else:
            score += top/bot

    if not(mean):
        score *= X.shape[0]

    return -1*score

parameters = ["Time","TM"]
feature_target = "TJM20"
min_length = 7*24+1 # minimum number of rows used in sequense
search_area = {"input_shape":[n for n in range(24,7*24+1,24)],"epochs":[20]}
base_model = GridSearchCV(SE.KerasBiGRU(),param_grid=search_area,n_jobs = -1,scoring=own_scorer,pre_dispatch=5)

# base_model = SE.KerasBiLSTM()

KerasBiLSTM = model_traning_testing(
   datafile = imputed_nibio_data[:,:,"2014":"2020"],
   base_model = base_model,
   parameters = parameters,
   feature_target = feature_target,
   min_length = min_length
)

print(KerasBiLSTM["global"]["model"].cv_results_)

KerasBiLSTM = recalc_stat(imputed_nibio_data[:,:,"2021":"2022"],
                            parameters,
                            feature_target,
                            KerasBiLSTM["global"]["model"],
                            min_length=min_length,
                            append_hist = True,
                            lead_time=KerasBiLSTM["global"]["model"].best_estimator_.input_shape,
                            old_model_stat=KerasBiLSTM
                            )
KerasBiLSTM = {
    "jitasdKerasBiLSTM":KerasBiLSTM
}
plot_model_performance(imputed_nibio_data,
                       KerasBiLSTM,
                       parameters,
                        probing_year="2022",
                        target=feature_target,
                        lead_time=KerasBiLSTM["jitasdKerasBiLSTM"]["global"]["model"].best_estimator_.input_shape,
                        name="jitasdKerasBiLSTM"
                    )
plot_model_performance(imputed_nibio_data,
                       KerasBiLSTM,
                       parameters,
                        probing_year="2021",
                        target=feature_target,
                        lead_time=KerasBiLSTM["jitasdKerasBiLSTM"]["global"]["model"].best_estimator_.input_shape,
                        name="jitasdKerasBiLSTM"
                    )
plot_predground_eclipse(
    imputed_nibio_data[:,:,"2021":"2022"],
    KerasBiLSTM,
    feature = parameters,
    target = "TJM20",
    figure_alpha = 0.5,
    point_size = 0.5,
    name = "jitasdKerasBiLSTM"
)
# -------------------- R^2 messure ------------------------

def own_scorer(estimator,X,y, raw = False):
    r"""Costum scorer that scores with a balance between $R^2$ and MSE.
        $$
            scorer = \left{\frac{MSE/R^2,R^2 > 0}{\infty, else}\right.
        $$
    """
    score = 0.0

    if isinstance(X,list):
        #n_samples = 0
        scale_down = 0.0
        for TX, Ty in zip(X,y):
            #n_samples += TX.shape[0]
            Tscore = own_scorer(estimator=estimator,X=TX,y=Ty,raw = True)
            #if np.isinf(Tscore):
            #    return Tscore
            scale_up = ((Ty-np.mean(Ty))**2).sum()
            scale_down += scale_up
            score += (1-Tscore)*scale_up
        return 1-(score/scale_down)

    if "input_shape" in estimator.__dict__ or "lag_max" in estimator.__dict__: # adjust y
        if "input_shape" in estimator.__dict__:
            new_y = y[:-(estimator.input_shape-1)]
        elif "lag_max" in estimator.__dict__:
            new_y = y[:-(estimator.lag_max-1)]
        else:
            raise ValueError("Can't find relevant shape in estimator.")
        #top = mean_squared_error(new_y,estimator.predict(X))
        bot = r2_score(new_y,estimator.predict(X))
        #if bot <= 0:
        #    return np.NINF
        score += bot
        
    else: # don't adjust y
        # accuracy_score, mean_squared_error, r2_score, mean_absolute_error
        #top = mean_squared_error(y,estimator.predict(X))
        bot = r2_score(y,estimator.predict(X))
        #if bot <= 0:
        #    return np.NINF
        score += bot

    #if not(raw):
    #    score *= X.shape[0]

    return score

parameters = ["Time","TM"]
feature_target = "TJM20"
min_length = 7*24+1 # minimum number of rows used in sequense
search_area = {"input_shape":[n for n in range(24,7*24+1,24)],"epochs":[20]}
base_model = GridSearchCV(SE.KerasBiGRU(),param_grid=search_area,n_jobs = -1,scoring=own_scorer,pre_dispatch=5)

# base_model = SE.KerasBiLSTM()

KerasBiLSTM = model_traning_testing(
   datafile = imputed_nibio_data[:,:,"2014":"2020"],
   base_model = base_model,
   parameters = parameters,
   feature_target = feature_target,
   min_length = min_length
)

print(KerasBiLSTM["global"]["model"].cv_results_)

KerasBiLSTM = recalc_stat(imputed_nibio_data[:,:,"2021":"2022"],
                            parameters,
                            feature_target,
                            KerasBiLSTM["global"]["model"],
                            min_length=min_length,
                            append_hist = True,
                            lead_time=KerasBiLSTM["global"]["model"].best_estimator_.input_shape,
                            old_model_stat=KerasBiLSTM
                            )
KerasBiLSTM = {
    "jitR2KerasBiLSTM":KerasBiLSTM
}
plot_model_performance(imputed_nibio_data,
                       KerasBiLSTM,
                       parameters,
                        probing_year="2022",
                        target=feature_target,
                        lead_time=KerasBiLSTM["jitR2KerasBiLSTM"]["global"]["model"].best_estimator_.input_shape,
                        name="jitR2KerasBiLSTM"
                    )
plot_model_performance(imputed_nibio_data,
                       KerasBiLSTM,
                       parameters,
                        probing_year="2021",
                        target=feature_target,
                        lead_time=KerasBiLSTM["jitR2KerasBiLSTM"]["global"]["model"].best_estimator_.input_shape,
                        name="jitR2KerasBiLSTM"
                    )
plot_predground_eclipse(
    imputed_nibio_data[:,:,"2021":"2022"],
    KerasBiLSTM,
    feature = parameters,
    target = "TJM20",
    figure_alpha = 0.5,
    point_size = 0.5,
    name = "jitR2KerasBiLSTM"
)

# -------------------- MSE messure ------------------------

def own_scorer(estimator,X,y, mean = True):
    r"""Costum scorer that scores with a balance between $R^2$ and MSE.
        $$
            scorer = \left{\frac{MSE/R^2,R^2 > 0}{\infty, else}\right.
        $$
    """
    score = 0.0

    if isinstance(X,list):
        n_samples = 0
        #scale_down = 0.0
        for TX, Ty in zip(X,y):
            n_samples += TX.shape[0]
            Tscore = own_scorer(estimator=estimator,X=TX,y=Ty, mean = False)
            #if np.isinf(Tscore):
            #    return Tscore
            #scale_up = ((Ty-np.mean(Ty))**2).sum()
            #scale_down += scale_up
            score -= Tscore#*TX.shape[0] #(1-Tscore)*scale_up
        return -score/n_samples

    if "input_shape" in estimator.__dict__ or "lag_max" in estimator.__dict__: # adjust y
        if "input_shape" in estimator.__dict__:
            new_y = y[:-(estimator.input_shape-1)]
        elif "lag_max" in estimator.__dict__:
            new_y = y[:-(estimator.lag_max-1)]
        else:
            raise ValueError("Can't find relevant shape in estimator.")
        top = mean_squared_error(new_y,estimator.predict(X))
        #bot = r2_score(new_y,estimator.predict(X))
        #if bot <= 0:
        #    return np.NINF
        score += top
        
    else: # don't adjust y
        # accuracy_score, mean_squared_error, r2_score, mean_absolute_error
        top = mean_squared_error(y,estimator.predict(X))
        #bot = r2_score(y,estimator.predict(X))
        #if bot <= 0:
        #    return np.NINF
        score += top

    if not(mean):
        score *= X.shape[0]

    return -1*score

parameters = ["Time","TM"]
feature_target = "TJM20"
min_length = 7*24+1 # minimum number of rows used in sequense
search_area = {"input_shape":[n for n in range(24,7*24+1,24)],"epochs":[20]}
base_model = GridSearchCV(SE.KerasBiGRU(),param_grid=search_area,n_jobs = -1,scoring=own_scorer,pre_dispatch=5)

# base_model = SE.KerasBiLSTM()

KerasBiLSTM = model_traning_testing(
   datafile = imputed_nibio_data[:,:,"2014":"2020"],
   base_model = base_model,
   parameters = parameters,
   feature_target = feature_target,
   min_length = min_length
)

print(KerasBiLSTM["global"]["model"].cv_results_)

KerasBiLSTM = recalc_stat(imputed_nibio_data[:,:,"2021":"2022"],
                            parameters,
                            feature_target,
                            KerasBiLSTM["global"]["model"],
                            min_length=min_length,
                            append_hist = True,
                            lead_time=KerasBiLSTM["global"]["model"].best_estimator_.input_shape,
                            old_model_stat=KerasBiLSTM
                            )
KerasBiLSTM = {
    "jitMSEKerasBiLSTM":KerasBiLSTM
}
plot_model_performance(imputed_nibio_data,
                       KerasBiLSTM,
                       parameters,
                        probing_year="2022",
                        target=feature_target,
                        lead_time=KerasBiLSTM["jitMSEKerasBiLSTM"]["global"]["model"].best_estimator_.input_shape,
                        name="jitMSEKerasBiLSTM"
                    )
plot_model_performance(imputed_nibio_data,
                       KerasBiLSTM,
                       parameters,
                        probing_year="2021",
                        target=feature_target,
                        lead_time=KerasBiLSTM["jitMSEKerasBiLSTM"]["global"]["model"].best_estimator_.input_shape,
                        name="jitMSEKerasBiLSTM"
                    )
plot_predground_eclipse(
    imputed_nibio_data[:,:,"2021":"2022"],
    KerasBiLSTM,
    feature = parameters,
    target = "TJM20",
    figure_alpha = 0.5,
    point_size = 0.5,
    name = "jitMSEKerasBiLSTM"
)