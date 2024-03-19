#!/usr/bin/env python
# coding: utf-8

# # Data visualisation
# 
# We start by importing the data

# In[1]:


import sklearn
import datetime

import os
import pickle
import copy

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import statsmodels as sm
import torch.utils.data as Data

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation 
from tensorflow.keras.layers import MaxPooling2D, Dropout, Conv2DTranspose
from tensorflow.keras.layers import concatenate, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics

#sklearn → model trening
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics         import accuracy_score, mean_squared_error, r2_score

#sklearn → data treatment
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.models import LinearRegression

#from ILSTM_Soil_model_main import lstm_interprety_soil_moisture as ILSTM
from My_tools import DataFileLoader as DFL # min egen
from My_tools import StudyEstimators as SE
# path definitions

ROOT = "../../"

PLOT_PATH = ROOT + "plots/"

DATA_PATH = ROOT + "data/"

METADATA_PRELOAD_DATA = ROOT + "PRIVATE_FILES/weatherdata.bin"

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
    "Innlandet" : ["11","18","26","27"],
    "Trøndelag" : ["15","57","34","39"],
    "Østfold" : ["37","41","52","118"],
    "Vestfold" : ["30","38","42","50"] # Fjern "50" for å se om bedre resultat
}


# Loading data from folders

# ## Function definitions

# In[2]:


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

def table2Latex(table, dir_path, file_name, header = "", append = False):
    if os.path.isfile(dir_path + file_name + ".tex") & append:
        file = open(dir_path+file_name+".tex","a", encoding="utf-8")
    else:
        file = open(dir_path+file_name+".tex","w", encoding="utf-8")

    file.write(r"\begin{tabular}{|l|" + ["c" for _ in range(len(header))].join("|") + "|}")
    if header != "":
        file.write(header.join("&") + r"\\\hline")
    for row in table:
        file.write(row.join("&") + r"\\\hline")
    file.write(r"\end{tabular}")


# In[3]:
if os.path.exists(METADATA_PRELOAD_DATA):
    imputed_nibio_data = pickle.load(open(METADATA_PRELOAD_DATA,"rb"))
    print("Fetched data from", METADATA_PRELOAD_DATA)
else:
    nibio_data_ungroup = DFL.DataFileLoader(DATA_COLLECTION_NIBIO,r"weather_data_hour_stID(\d{1,3})_y(\d{4}).csv")
    nibio_data_ungroup.load_data(names = ["Time","TM","RR","TJM10","TJM20"])
    nibio_data = nibio_data_ungroup.group_layer(nibio_id)

    nibio_data_raw_ungroup = DFL.DataFileLoader(DATA_COLLECTION_NIBIO,r"weather_data_raw_hour_stID(\d{1,3})_y(\d{4}).csv")
    nibio_data_raw_ungroup.load_data(names = ["Time","TM","RR","TJM10","TJM20"])
    nibio_data_raw = nibio_data_raw_ungroup.group_layer(nibio_id)

    def dataframe_merge_func(x,y):
        y.iloc[y.iloc[:,1].notna() & (y.iloc[:,1] <= 0),2] = pd.NA
        x.iloc[0:y.shape[0],2] = y.iloc[0:y.shape[0],2]
        return x

    imputed_nibio_data = nibio_data.combine(nibio_data_raw,merge_func = dataframe_merge_func)
    pickle.dump(imputed_nibio_data,open(METADATA_PRELOAD_DATA,"wb"))
    print("dumped data to",METADATA_PRELOAD_DATA)


def attempt_fitting(base_model, param_area, nibio_data):
    best_plauborg = {
    "Score":np.inf,
    "mse":0,
    "r2":0,
    "year":0,
    "model":None
    }

    worst_plauborg = {
    "Score":-np.inf,
    "mse":0,
    "r2":0,
    "year":0,
    "model":None,
    }

    all_plauborg = []

    search_area = param_area

    base_model = base_model()

    for regi in nibio_id.keys():
        for i in range(2014,2022):
            # First we fetch region (regi), all stations (:), then relevant years ("2014":str(i)). Since we only look at one region at the time
            # we remove the root group (shave_top_layer()), then we merge the years (merge_layer(level = 1), level 1 since level 0 would be the stations at this point)
            # then make a list (flatten(), the default handeling is to put leafs in a list)
            
            data = nibio_data[regi,:,"2014":str(i)].shave_top_layer().merge_layer(level = 1).flatten(return_key = True) # shape [(key, value)] ; looks at all previus years including this year
            test = nibio_data[regi,:,str(i+1)].shave_top_layer().merge_layer(level = 1).flatten(return_key = True) # shape [(key, value)] ; looks at the next year

            data = [(k,v.infer_objects(copy=False).fillna(0)) for k,v in data] # Removes nan in a quick manner
            test = [(k,v.infer_objects(copy=False).fillna(0)) for k,v in test] # but will be reviced.
            
            model = GridSearchCV(copy.deepcopy(base_model),param_grid=search_area,pre_dispatch=20, n_jobs = -1)
            n = 0
            overall_r2 = None # approximates a average r2
            overall_mse = None
            for d,t in zip(data,test): # fitting model with all stations
                model.fit(d[1].loc[:,["Time","TM"]],d[1].loc[:,["TJM20"]]) # regions model
                s_model = GridSearchCV(copy.deepcopy(base_model),param_grid=search_area, n_jobs = -1).fit(d[1].loc[:,["Time","TM"]],d[1].loc[:,["TJM20"]])# Station model
                s_pred = s_model.predict(t[1].loc[:,["Time","TM"]])
                
                if overall_r2 is not None:
                    overall_r2 = 1-mediant(1-overall_r2, 1-r2_score(t[1].loc[t[1].shape[0]-s_pred.shape[0]:,"TJM20"].to_numpy(),s_pred))
                else:
                    overall_r2 = r2_score(t[1].loc[t[1].shape[0]-s_pred.shape[0]:,"TJM20"].to_numpy(),s_pred)
                
                if overall_mse is not None:
                    overall_mse = (t[1].loc[t[1].shape[0]-s_pred.shape[0]:,"TJM20"].to_numpy()-s_pred)**2 + (overall_mse*n)
                else:
                    overall_mse = (t[1].loc[t[1].shape[0]-s_pred.shape[0]:,"TJM20"].to_numpy()-s_pred)**2
                n += len(s_pred)
                overall_mse /= n

                show_plot([
                    pd.DataFrame({"Time":t[1].loc[t[1].shape[0]-s_pred.shape[0]:,"Time"].to_numpy().ravel(),"TJM20":s_pred.ravel() - t[1].loc[t[1].shape[0]-s_pred.shape[0]:,["TJM20"]].to_numpy().ravel()})
                ],{0:{"label":"spesial"}})
            
            show_plot([
                pd.DataFrame({"Time":t[1].loc[t[1].shape[0]-s_pred.shape[0]:,"Time"],"TJM20":model.predict(t[1].loc[:,["Time","TM"]]).ravel() - t[1].loc[t[1].shape[0]-s_pred.shape[0]:,["TJM20"]].to_numpy().ravel()})
            ],{0:{"label":"global"}})

            plt.savefig(PLOT_PATH + base_model.__name__ +"_" + regi + "_y"+ str(i) + ".pdf")
            plt.clf()

            mse = {k:mean_squared_error(t.loc[t.shape[0]-s_pred.shape[0]:,"TJM20"].to_numpy().ravel(),model.predict(t.loc[:,["Time","TM"]])) for k,t in test}
            r2 = {k:r2_score(t.loc[t.shape[0]-s_pred.shape[0]:,"TJM20"].to_numpy().ravel(),model.predict(t.loc[:,["Time","TM"]])) for k,t in test}
                
            print(base_model.__name__,":",regi,"from year 2014 to year",i,":\n",
                  "\tMSE:", mse,
                  "\n\tR2:",r2,
                  "\n\tparams:",model.best_params_)
            score = max(m/r if r != 0 else np.inf for m,r in zip(mse.values(),r2.values()))
            model_info = {
                    "Name":base_model.__name__,
                    "Score":score, 
                    "params":model.best_params_,
                    "mse":mse, 
                    "r2": r2,
                    "r2_spes": overall_r2,
                    "mse_spes": overall_mse,
                    "year_max": str(i+1), 
                    "region": regi, 
                    "model": model
                }
            all_plauborg.append(model_info)
            if score < best_plauborg["Score"]:
                best_plauborg = model_info
            elif score > worst_plauborg["Score"]:
                worst_plauborg = model_info

    return {"all":all_plauborg,"best":best_plauborg,"worst":worst_plauborg}
# In[5]:


for regi in nibio_id.keys(): 
    show_plot([station.loc[:,["Time","TJM20"]] for station in imputed_nibio_data[regi,:].shave_top_layer().merge_layer(level=1).flatten()],{})
    plt.legend(nibio_id[regi])
    plt.title("Område: {}, feature: {}".format(regi,"TJM20"))
    plt.savefig(PLOT_PATH + regi + '.pdf', bbox_inches='tight') # pdf for vectorised grafics.
    plt.clf() # clear current figure for the next figure


# The data is splitted among two collections of data, one is a pdf and the other is a `.xlsx` format. We start by collecting the data from the hourly data collection.


# ## Linear regression function 
# 
# This function does a transformation of the $m\times n$ matrix (our dataframe) to a $m \times p$ matrix. This can be seen as a kernel trick where we transform the data to a more seperable state to improve prediction. The scema for this model is
# $$
#     (\vec{F}\circ \mathbf{A})\vec{\beta}=\vec{y}+\vec{\varepsilon}
# $$

# In[ ]:


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

# Linear Regression

result_fitting = attempt_fitting(LinearRegression,{"fit_intercept":[True,False],"Positive":[True,False]},imputed_nibio_data)

print("Linear Regresson best:",result_fitting["best"])
print("Linear Regresson worst:",result_fitting["worst"])
print("Linear Regresson median:",sorted(result_fitting["all"],key = lambda x: x["Score"])[int(len(result_fitting["all"])/2)])

# ### Plauborg regression
# 
# Author Plauborg used the above model to predict soil temperature, but used previus time to make the model more time dependent and fourier terms to reflect changes during the year.

# In[ ]:


# In[ ]:

#! Need to adjust following code
result_fitting = attempt_fitting(SE.PlauborgRegresson,{"lag_max":range(2,8),"fourier_sin_length":range(2,10),"fourier_cos_length":range(2,10)},imputed_nibio_data)

print("best:",result_fitting["best"])
print("worst:",result_fitting["worst"])
print("median:",sorted(result_fitting["all"],key = lambda x: x["Score"])[int(len(result_fitting["all"])/2)])

best_data = imputed_nibio_data[best_plauborg := result_fitting["best"],:,best_plauborg["year_max"]].merge_layer(level = 0)
worst_data = imputed_nibio_data[worst_plauborg := result_fitting["worst"],:,worst_plauborg["year_max"]].merge_layer(level = 0)

show_plot([
    pd.DataFrame({
        "Time":best_data.Time.iloc[:5879].to_numpy().ravel(),
        "TJM20":best_plauborg["model"].predict(best_data.loc[:5878,["Time","TM"]]).ravel() - best_data.loc[:5878,["TJM20"]].to_numpy().ravel()
    }) ],
    {})
plt.title("Y_pred - Y_truth")
plt.savefig(PLOT_PATH + "Plauborg_plot_best.pdf")
show_plot([
    pd.DataFrame({
        "Time":worst_data.Time.iloc[:5879].to_numpy().ravel(),
        "TJM20":worst_plauborg["model"].predict(worst_data.loc[:5878,["Time","TM"]]).ravel() - worst_data.loc[:5878,["TJM20"]].to_numpy().ravel()
    }) ],
    {})
plt.title("Y_pred - Y_truth")
plt.savefig(PLOT_PATH + "Plauborg_plot_worst.pdf")


# In[ ]:


# imputed_nibio_data["Vestfold",:,"2019"].DictData


# ### Rankin regression
# 
# This regression tries to solve the following integreal using an FDM.
# 
# $$
# T = \int_{t_0}^{t_{max}} \frac{1}{C_{A}} \frac{\partial}{\partial z}\left(K_T \frac{\partial T}{\partial z}\right) dt
# $$
# 
# Where T is temperature, z is depth, and t is time. In this study we will approximate several thing including
# 
# - $K_T / C_A \approx \partial_tT/\partial^2_zT$
# - $f_S \approx -0.5\ln(T^{t+1}/T_*^{t})/D_t$

# best_rankin = {
#     "Score":np.inf,
#     "mse":0,
#     "r2":0,
#     "year":0,
#     "model":None
# }
# 
# worst_rankin = {
#     "Score":-np.inf,
#     "mse":0,
#     "r2":0,
#     "year":0,
#     "model":None,
# }
# 
# base_model = SE.RankinRegresson()
# 
# for regi in nibio_id.keys():
#     for i in range(2014,2022):
#         # First we fetch region (regi), all stations (:), then relevant years ("2014":str(i)). Since we only look at one region at the time
#         # we remove the root group (shave_top_layer()), then we merge the years (merge_layer(level = 1), level 1 since level 0 would be the stations at this point)
#         # then make a list (flatten(), the default handeling is to put leafs in a list)
#         
#         data = imputed_nibio_data[regi,:,"2014":str(i)].shave_top_layer().merge_layer(level = 1).flatten() # looks at all previus years including this year
#         test = imputed_nibio_data[regi,:,str(i+1)].shave_top_layer().merge_layer(level = 1).flatten() # looks at the next year
# 
#         data = [d.infer_objects(copy=False).fillna(0) for d in data] # Removes nan in a quick manner
#         test = [d.infer_objects(copy=False).fillna(0) for d in test] # but will be reviced.
#         
#         model = copy.deepcopy(base_model)
#         overall_r2 = None
#         for d,t in zip(data,test): # fitting model with all stations
#             model.fit(d,d.loc[:,["TJM20"]]) # regions model
#             s_model = copy.deepcopy(base_model).fit(d,d.loc[:,["TJM20"]]) # Station model
#             if overall_r2 is not None:
#                 overall_r2 = 1-mediant(1-overall_r2, 1-r2_score(t["TJM20"].to_numpy(),s_model.predict(t)))
#             else:
#                 overall_r2 = r2_score(t["TJM20"].to_numpy(),s_model.predict(t))
#             
#             
#         print(regi,"from year 2014 to year",i,":\n",
#               "\tMSE:",
#               mae := [mean_squared_error(t["TJM20"].to_numpy(),model.predict(t)) for t in test],
#               "\n\tR2:",
#               r2 := [r2_score(t["TJM20"].to_numpy(),model.predict(t)) for t in test])
#         score = max(m/r for m,r in zip(mae,r2))
#         model_info = {
#                 "Score":score, 
#                 "mse":mae, 
#                 "r2": r2,
#                 "r2_spes": overall_r2,
#                 "year_max": i, 
#                 "region": regi, 
#                 "model": model
#             }
#         if score < best_rankin["Score"]:
#             best_rankin = model_info
#         elif score > worst_rankin["Score"]:
#             worst_rankin = model_info
# 
# print(best_rankin)
# print(worst_rankin)

# ## LSTM
# 
# This is a base model for testing ILSTM in the next section.

# In[ ]:

result_fitting = attempt_fitting(SE.KerasBiLSTM,{"input_shape":[24*n for n in range(1,7)],"lstm_units":[2*k for k in range(20,25)],"epochs":[4*n for n in range(30,50)]},imputed_nibio_data)

print("best:",result_fitting["best"])
print("worst:",result_fitting["worst"])
print("median:",sorted(result_fitting["all"],key = lambda x: x["Score"])[int(len(result_fitting["all"])/2)])

# In[10]:


# ------------------------------------------------------------------------------------
all_data_daily = data_t.set_index("Time").resample("D").mean().dropna().reset_index()

p_data = F_plauborg(all_data_daily)
ridge = LinearRegression().fit(p_data[50:], all_data_daily.iloc[50:,[-1]])
y_pred = ridge.predict(p_data[50:])
display = PredictionErrorDisplay(y_true=all_data_daily.iloc[50:,[-1]], y_pred=y_pred)
display.plot(kind = "actual_vs_predicted",scatter_kwargs = {
    "c": np.linspace(0,1,num = all_data_daily.iloc[50:,[-1]].shape[0]),
    "color": None
})
plt.show()

all_data_daily = all_data_daily.reset_index().loc[50:]

Y = pd.DataFrame(
    zip(all_data_daily["Time"].to_numpy().tolist(), y_pred.flatten()),
    columns=["Time","Y_pred"])

show_plot([all_data_daily.loc[:,["Time","TJM20"]],Y,all_data_daily.loc[:,["Time","RR"]]],{1:{"alpha":0.5}} )
plt.legend(["Y","Y_pred"])
plt.ylim(-5,25)
plt.ylabel("℃")
plt.show()


# # ILSTM training
# 
# Here we will be training a version of LSTM

# In[30]:


import copy

def ILSTM_train(raw_data, target_label,total_epoch = 50,hidden_size=16,lerningrate=1e-3, lead_time=1, seq_length=24, batch_size=16):
    data,scaler,scaler1 = ILSTM.nibio_data_transform(raw_data, target_label)
    data = scaler1.transform(data)

    # TODO: Generate the tensor for lstm model

    [data_x, data_y,data_z] = ILSTM.LSTMDataGenerator(data, lead_time, batch_size, seq_length)

       # concat all variables.
    # TODO: Flexible valid split
    data_train_x=data_x[:int((data_x.shape[0])-400*24)]
    data_train_y = data_y[:int(data_x.shape[0]-400*24)]

    train_data = Data.TensorDataset(data_train_x, data_train_y)
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    data_valid_x=data_x[int(data_x.shape[0]-400*24):int(data_x.shape[0]-365*24)] # -> trener 35 dager
    data_valid_y=data_y[int(data_x.shape[0]-400*24):int(data_x.shape[0]-365*24)] # -> tester 35 dager
    data_test_x=data_x[int(data_x.shape[0]-365*24):int(1.0 * data_x.shape[0])] # -> validerer på resterende
    data_testd_z=data_z[int(data_x.shape[0]-365*24):int(1.0 * data_x.shape[0])] # -> stat på rest

    # TODO: Flexible input shapes and optimizer
    # IMVTensorLSTM,IMVFullLSTM
    model = ILSTM.ILSTM_SV(data_x.shape[2],data_x.shape[1], 1, hidden_size).cuda()
    # TODO: Trian LSTM based on the training and validation sets
    model,predicor_import,temporal_import=ILSTM.train_lstm(model,lerningrate,total_epoch,train_loader,data_valid_x,data_valid_y,"./saved_models/lstm_1d.h5")

    # TODO: Create predictions based on the test sets
    pred, mulit_FV_aten, predicor_import,temporal_import = ILSTM.create_predictions(model, data_test_x,scaler)
    # TODO: Computer score of R2 and RMSE

    data_testd_z=data_testd_z.reshape(-1,1)
    data_testd_z=data_testd_z.cpu()
    data_testd_z=data_testd_z.detach().numpy()
    # Unnormalize
    data_testd_z=scaler.inverse_transform(data_testd_z)
    ILSTM.compute_rmse_r2(data_testd_z,pred,modelname)

    print(pred)
    


# Need to transform the data first to fit the model.

# In[26]:


def datetime2string(x):
    x["Time"] = x["Time"].apply(lambda y: y.strftime("%Y-%m-%d %X"))
    return x
station_data = imputed_nibio_data.data_transform(datetime2string).merge_layer(level = 1)


# In[31]:


ILSTM_train(copy.deepcopy(station_data["11"]),"TJM20",batch_size = 8,total_epoch = 20)

