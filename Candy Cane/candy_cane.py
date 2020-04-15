# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:43:19 2020

@author: Dustin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
from plotly.offline import plot
import plotly.graph_objs as go

##########################   Format Data    ###################################
#Note: Skip to part 2 if this is already done
###############################################################################

data = pd.read_excel('CANDY CANE 101 - CLASS.xlsx')

#Drop the No Data Available cases
NoData_index = data[data['Flowline Temperature'] == 'No Available Data'].index
data.drop(index = NoData_index, inplace = True)

#Look at data
description = data.describe(include = 'all')
data.dtypes
# Out[17]: 
# Type of Facility                            object
# Name of Facility                            object
# Timestamp                           datetime64[ns]
# Wellhead Casing "A" - Pressure              object
# Wellhead Casing "B" - Pressure              object
# Flowline Pressure                           object
# Flowline Temperature                        object
# Volume - Calendar Day Production            object
# Wellhead Tubing - Pressure                  object
# dtype: object

#Change type for relative columns to values
int_columns = {'Wellhead Casing "A" - Pressure': 'float32',
               'Wellhead Casing "B" - Pressure': 'float32',
               'Flowline Pressure': 'float32',
               'Flowline Temperature' : 'float32',
               'Volume - Calendar Day Production': 'float32',
               'Wellhead Tubing - Pressure' : 'float32'
    }


#Did not work - why?
data.astype(int_columns).dtypes
#There is a value of 'The time is invalid.'
TimeInvalid_idx = data[data['Wellhead Tubing - Pressure'] == 'The time is invalid.'].index
data.drop(index = TimeInvalid_idx, inplace = True)

data = data.astype(int_columns)
data.dtypes
# Out[70]: 
# Type of Facility                            object
# Name of Facility                            object
# Timestamp                           datetime64[ns]
# Wellhead Casing "A" - Pressure             float32
# Wellhead Casing "B" - Pressure             float32
# Flowline Pressure                          float32
# Flowline Temperature                       float32
# Volume - Calendar Day Production           float32
# Wellhead Tubing - Pressure                 float32
# dtype: object
#Do it via loop and pd.to_numeric

# negative values are erroneous. How many are there?

float_idx = data.dtypes[data.dtypes == 'float32'].index

data[data[float_idx[0]]<0].index

def find_neg(x):
    count = (x < 0).sum()
    idx = x[x<0].index
        
    return count, idx

neg_values = data[float_idx].apply(find_neg)
# Wellhead Casing "A" - Pressure      92                                                (0, [])
# Flowline Pressure                   194                                         (0, [])
# Wellhead Tubing - Pressure          5320
# dtype: object

#count the unique values
neg_values_unique = []
for i in neg_values.index:
    neg_values_unique += list(neg_values.loc[i][1])
neg_values_unique = np.unique(neg_values_unique)
#they potentially lost power
#A sensor that is off
#For time series - fill it out
print('% of cases where a reading has erroneous recordings: {}'.format((len(neg_values_unique)/len(data))*100))
# % of cases where a reading has erroneous recordings: 1.316982946243129

#GO AHEAD AND DROP THESE CASES - ask if this is coshure in class
data.drop(index = neg_values_unique, inplace = True)

#Calculate the 'Wellhead Casing "B" - Pressure' via sliding average
#do it in 15 minute increments
data['Wellhead Casing "B" - Pressure'] = data['Wellhead Casing "A" - Pressure'].rolling(window = 15).mean()
data['Wellhead Casing "B" - Pressure'].fillna(data['Wellhead Casing "A" - Pressure'], inplace = True)

#look at only data with volume/day present
data_volume = data.iloc[56120:,:]
#Looks like @ index 56120 is when the well kicks off
dv_index = data_volume.index

data.to_csv('data_nonulls.csv')
data_volume.to_csv('data_nonull_afterstart.csv')

##########################   Import Refinded Data    ######################
data_volume = pd.read_csv('data_nonull_afterstart.csv')
data_volume.drop(columns = 'Unnamed: 0', inplace = True)
dv_index = data_volume.index
#data_volume.dtypes
#Type of Facility                     object
#Name of Facility                     object
#Timestamp                            object
#Wellhead Casing "A" - Pressure      float64
#Wellhead Casing "B" - Pressure      float64
#Flowline Pressure                   float64
#Flowline Temperature                float64
#Volume - Calendar Day Production    float64
#Wellhead Tubing - Pressure          float64
#dtype: object

#Convert to datetime
data_volume['Timestamp'] = pd.to_datetime(data_volume['Timestamp'])
#data_volume.dtypes

##########################   Plot Data    ###################################
fig = go.Figure()
fig.add_trace(go.Scatter(x=data_volume.Timestamp, y=data_volume['Volume - Calendar Day Production'], name="Volume",
                         line_color='deepskyblue'))
fig.update_layout(title_text='Time Series with Rangeslider',
                  xaxis_rangeslider_visible=True)

plot(fig, auto_open=True)

##########################   Moving Averages    ##############################
volume = data_volume['Volume - Calendar Day Production']

#I looked at a handfull of exponential windows to see which was better, but I
#did not like them. I put the coding in the bottom for future references if needed

#look at other types of averages at 1hr increments
df_avg = {#'exp_avg' : volume.ewm(span = 60).mean(),
          'exp_avg' : volume.ewm(span = 720).mean(), #12hr increments
          #'rolling' : volume.rolling(60, min_periods = 1).mean(),
          #'rolling' : volume.rolling(1440, min_periods = 1).mean(), #day average
          'rolling' : volume.rolling(4320, min_periods = 1).mean(), #3day average
          'too_smooth' : volume.rolling(10080, min_periods = 1).mean(), #7day average
          #'expanding' : volume.expanding(60).mean()
        }
#Dont include it in LSTM
df_avg['True_Volume'] = volume
df_avg['Timestamp'] = data_volume['Timestamp']
df_avg = pd.DataFrame(df_avg)
#df_avg['expanding'][df_avg['expanding'].isnull()] = df_avg['True_Volume'][df_avg['expanding'].isnull()].mean()

"""
Notes on plotting:
    Looked @ 60 minute averages. Liked the exponential for mimic of real and
        like the expanding for a an averaged line
    After looking at the cumulative for the entire graph, it did not follow
        the trend downwards at all, will look at rolling w/ bigger average (day)
    Finally I used an exponential window of 12hrs and rolling window of 24 hrs
        now I can do a formula of when exponential decay < 10% size of rolling window
        to start a downtime and track the time
"""
#Plot the traces against eachother
#colors = ['black', 'deepskyblue', 'green', 'gold']#, 'red']#, 'orange']
#columns = list(df_avg.columns)[1:]

colors = ['black', 
          'deepskyblue', 
          'red']
columns = ['too_smooth', 
           'exp_avg', 
           'rolling']
fig = go.Figure()
for i in range(len(colors)):
    fig.add_trace(go.Scatter(x=df_avg.Timestamp, y=df_avg[columns[i]], name=columns[i],
                         line_color=colors[i]))
    
fig.update_layout(title_text='Time Series With 60 min Window Averages',
                  xaxis_rangeslider_visible=True)

plot(fig, auto_open=True)

##########################   Label Data    ###################################
"""
1st method was anything less than 6000 (Pasted below)
2nd anything less than 4000 (Pasted Below)
3rd compare the exponential window and rolling window for when they cross paths
"""

#Track downtime - format in a way to add to the dataframe for learning
status = 'end'
down = 0
track = {'start_down':[],
         'end_down': [],
         'down': []}
for i in df_avg.index:
    previous_status = status
    diff  = df_avg.loc[i,'exp_avg'] - df_avg.loc[i,'rolling']
    if diff < .10 * -df_avg.loc[i, 'rolling']:
        if status != 'start':
            print('down @ {}'.format(i))
            status = 'start'
            down = 1
    elif diff > 0 and status == 'start':
        status = 'end'
        down = 0
        print('done @ {}'.format(i))
    #Store Everything
    if status == 'start' and previous_status == 'end':
        track['start_down'] += [1]
    else:
        track['start_down'] += [0]
    if status == 'end' and previous_status == 'start':
        track['end_down'] += [1]
    else:
        track['end_down'] += [0]
    track['down'] += [down]

sum(track['start_down'])
sum(track['end_down'])
#sum(track['down'])
    
#Add downtime to True Volume
df_avg['Down'] = track['down']
df_avg['start_down'] = track['start_down']
df_avg['end_down'] = track['end_down']

#Plot it on the Graph
fig = go.Figure()

fig.add_trace(go.Scatter(x=df_avg.Timestamp, y=df_avg.True_Volume, name='Volume',
                         line_color='green'))

fig.add_trace(go.Scatter(x = df_avg.Timestamp[df_avg['start_down']==1],
                         y = df_avg.True_Volume[df_avg['start_down']==1],
                         mode = 'markers',
                         marker = dict(color = 'red',
                                       size = 10
                                       ),
                         name = 'start'
                         ))

fig.add_trace(go.Scatter(x = df_avg.Timestamp[df_avg['end_down']==1],
                         y = df_avg.True_Volume[df_avg['end_down']==1],
                         mode = 'markers', 
                         marker = dict(color = 'black',
                                       size = 10
                                       ),
                         name = 'end'))
    
fig.update_layout(title_text='Time Series with downtime start/stop')

plot(fig, auto_open=True)

"""
If you can estimate the total deferred volume by getting a better estimate of the
mean values (try a week at a time for windows of mean with exponential and see how it tracks) then
look at difference

If you can estimate whether planned or unplanned, then this may help in estimating
the planned ones
"""

"""
Just run an LSTM on it and see how it does now
1) All of the data
2) Exclude those from the training and the test
                         Frequency of down time
                         Average number of days between maintenance           
3) Multivariate - use production and a pressure

Filters:
    hodrick prescott filter
    -growth estimates
    Common Filter
"""
############################   Look at Lengths of Downtime   ##################
"""
Getting downtime information
note: some of the downtime is longer than 24 hrs. so you will have to split by
day somehow to get it even more similar to the other data set
"""
#downtime information
count = 0
index = []
status = 0 #0 is not down, 1 is down
tracker = {
        'Date' : [],
        'length_down' : [],
        'deferred_volume' : [],
        'produced_volume': []
        }
for i in range(len(df_avg['Timestamp'])):
    prev_status = status
    #change the status
    if df_avg.loc[i,'start_down'] == 1:
        status = 1
        date = df_avg.loc[i, 'Timestamp']
    elif df_avg.loc[i,'end_down'] == 1:
        status = 0
    #do calculation if status is down
    if status == 1:
        count += 1    
        index += [i]
    #do final calculations for down time and store the data
    if prev_status == 1 and status ==0:
        hrs = count/60
        tracker['length_down'] += [hrs] #number of hours down
        #track estimated volume lost
        expected_volume = df_avg.loc[index, 'too_smooth'].mean()*hrs/24
        actual_volume = df_avg.loc[index, 'True_Volume'].sum()/(1440)
        tracker['Date'] += [date]
        tracker['deferred_volume'] += [expected_volume - actual_volume]
        tracker['produced_volume'] += [actual_volume]
        index = []
        count = 0

df_downtime = pd.DataFrame(tracker)      
        
df_downtime['Date'] = pd.to_datetime(df_downtime['Date'])
df_downtime.dtypes

#https://stackoverflow.com/questions/26105804/extract-month-from-date-in-python/26105855
############################   Add winter/not winter   ########################



############################   LSTM Univariate   #########################################
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

"""
https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

1) Use data that doesnt include the PM's
2) Use data that does include PM's

Structure Data [ Normalize between (0,1), lookback = 1hr, Regression with TimeSteps ]
Create structure of LSTM Model
Train the Model
Scores of the Model (Make sure to inverse transform for RMSE)
Explore/Examine
"""
#######   Structure the Data
df_avg.columns
data_volume.columns

data = pd.concat([data_volume.iloc[:,2:], df_avg.iloc[:,2:]], axis = 1)
data['Volume - wo dt'] = np.where(data['Down']==0, data['Volume - Calendar Day Production'], data['too_smooth'])
data.columns
#Index(['Timestamp', 'Wellhead Casing "A" - Pressure',
#       'Wellhead Casing "B" - Pressure', 'Flowline Pressure',
#       'Flowline Temperature', 'Volume - Calendar Day Production',
#       'Wellhead Tubing - Pressure', 'exp_avg', 'rolling', 'too_smooth',
#       'Down', 'start_down', 'end_down', 'Volume - wo dt'],
#      dtype='object')

#lookback
def look_back(df, lookback = 1):
    #new data will have to start at lookback because it will be using last 15
    #minutes for decision
    data2 = df.loc[lookback:,:]
    data2.rename(columns = {0:'y'}, copy = False, inplace = True)
    
    n = len(data2['y'])
    
    for i in range(1,lookback + 1):
        #Do this to start at the furthest lookback
        vec = df.loc[lookback-i:]
        #Using the [:n,:] caps the vector off at actual length
        data2['X{}'.format(i)] = vec.loc[:n,:]
        print('step 3.{} done'.format(i))

    return data2

lookback = 15 #15 minutes. I tried 60 minutes but took too long

#######   Structure the LSTM Model  **Refer to LSTM for regression with TimeSteps
#LSTM Model
model = Sequential()
model.add(LSTM(45, input_shape = (lookback, 1)))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')

#######   WITHOUT DOWNTIME
X_ndt = data['Volume - wo dt']
# NOTE TO TRY THIS WITHOUT ANY VALUES IN THOSE SPOTS

#Scale
scaler = MinMaxScaler(feature_range=(0, 1))
X_ndt_scaled = scaler.fit_transform(np.array(X_ndt).reshape(-1,1))

df = look_back(pd.DataFrame(X_ndt_scaled), lookback = lookback)
X, y = df.iloc[:,1:], df.iloc[:,0]
shape = X.shape

#Structure the Data
X = np.reshape(np.array(X), (shape[0], shape[1], 1))
X.shape

#Train and Test
train_X, train_y = X[:int(.8*len(X)),:], y[:int(.8*len(X))]
test_X, test_y = X[int(.8*len(X)):len(X)-lookback,:], y[int(.8*len(X)):len(X)-lookback]

# fit network #note to reset model if doing another go
history = model.fit(train_X, train_y, epochs=15, batch_size = 1000, validation_data=(test_X, test_y), verbose=2, shuffle=False)
#about 35 sec/epoch

plt.figure(figsize=(12,8))
plt.title('Loss Function', fontsize=21)
plt.xlabel('\nNumber of Epochs')
plt.ylabel('Loss Value for Train and Test Datasets\n')
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Test')
plt.legend()
plt.show()

#######    WITH DOWNTIME
X_dt = data['Volume - Calendar Day Production']

#Scale
scaler = MinMaxScaler(feature_range=(0, 1))
X_dt_scaled = scaler.fit_transform(np.array(X_dt).reshape(-1,1))

df = look_back(pd.DataFrame(X_dt_scaled), lookback = lookback)
X, y = df.iloc[:,1:], df.iloc[:,0]
shape = X.shape

#Structure the Data
X = np.reshape(np.array(X), (shape[0], shape[1], 1))
X.shape

#Train and Test (use 20% as test set)
train_X, train_y = X[:int(.8*len(X)),:], y[:int(.8*len(X))]
test_X, test_y = X[int(.8*len(X)):len(X)-lookback,:], y[int(.8*len(X)):len(X)-lookback]
train_index = list(train_y.index); test_index = list(test_y.index)

# fit network #note to reset model if doing another go
history = model.fit(train_X, train_y, epochs=15, batch_size = 1000, validation_data=(test_X, test_y), verbose=2, shuffle=False)
#about 35 sec/epoch

plt.figure(figsize=(12,8))
plt.title('Loss Function', fontsize=21)
plt.xlabel('\nNumber of Epochs')
plt.ylabel('Loss Value for Train and Test Datasets\n')
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Test')
plt.legend()
plt.show()

#######    Using Train, Validation, Test
X_dt = data['Volume - Calendar Day Production']

#Scale
scaler = MinMaxScaler(feature_range=(0, 1))
X_dt_scaled = scaler.fit_transform(np.array(X_dt).reshape(-1,1))

df = look_back(pd.DataFrame(X_dt_scaled), lookback = lookback)
X, y = df.iloc[:,1:], df.iloc[:,0]
shape = X.shape

#Structure the Data
X = np.reshape(np.array(X), (shape[0], shape[1], 1))
X.shape

#Train and Test (use 20% as test set)
train_X, train_y = X[:int(.7*len(X)),:], y[:int(.7*len(X))]
val_X, val_y = X[int(.7*len(X)):int(.85*len(X)),:], y[int(.7*len(X)):int(.85*len(X))]
test_X, test_y = X[int(.85*len(X)):len(X)-lookback,:], y[int(.85*len(X)):len(X)-lookback]
train_index = list(train_y.index); val_index = list(val_y.index); test_index = list(test_y.index)

# fit network #note to reset model if doing another go
history = model.fit(train_X, train_y, epochs=15, batch_size = 1000, validation_data=(val_X, val_y), verbose=2, shuffle=False)
#about 35 sec/epoch

plt.figure(figsize=(12,8))
plt.title('Loss Function', fontsize=21)
plt.xlabel('\nNumber of Epochs')
plt.ylabel('Loss Value for Train and Test Datasets\n')
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Test')
plt.legend()
plt.show()

#######    MAKE PREDICTIONS
trainPredict = model.predict(train_X)
#note to only use on last method
valPredict = model.predict(val_X) 
testPredict = model.predict(test_X)
# invert predictions
trainPredict = pd.DataFrame(scaler.inverse_transform(trainPredict))
train_y = scaler.inverse_transform(np.reshape(np.matrix(train_y), (-1,1)))
#note to only use on last method
valPredict = pd.DataFrame(scaler.inverse_transform(valPredict)) #note to only use on last method
val_y = scaler.inverse_transform(np.reshape(np.matrix(val_y), (-1,1))) #note to only use on last method
testPredict = pd.DataFrame(scaler.inverse_transform(testPredict))
test_y = scaler.inverse_transform(np.reshape(np.matrix(test_y), (-1,1)))
# calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(train_y, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
##
valScore = np.sqrt(mean_squared_error(val_y, valPredict))
print('Validation Score: %.2f RMSE' % (valScore))
testScore = np.sqrt(mean_squared_error(test_y, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

#######    PLOT THE PREDICTIONS
# Format true y values
y_true = pd.DataFrame(data['Volume - Calendar Day Production'])
y_true.reset_index(inplace = True)

# shift train predictions for plotting by assigning index
trainPredict['index'] = train_index

# shift validatin predictions for plotting by assigning index
#note to only use on last method
valPredict['index'] = val_index

# shift test predictions for plotting by assigning index
testPredict['index'] = test_index

#Merge DataFrames on column index
y2 = pd.merge_ordered(y_true, trainPredict, on = 'index')
y2 = pd.merge_ordered(y2, testPredict, on = 'index')
#note to only use on last method
y2 = pd.merge_ordered(y2, valPredict, on = 'index')

# plot baseline and predictions
fig = go.Figure()

fig.add_trace(go.Scatter(x=y2['index'], y=y2['Volume - Calendar Day Production'],
                         line_color = 'black',
                         name = 'Baseline'))

fig.add_trace(go.Scatter(x=y2['index'], y=y2['0_x'], name='Train Prediction',
                         line_color='green'))

fig.add_trace(go.Scatter(x=y2['index'], y=y2['0_y'],
                         line_color = 'red',
                         name = 'Test Prediction'
                         ))
#note to only use on last method
fig.add_trace(go.Scatter(x=y2['index'], y=y2[0],
                         line_color = 'blue',
                         name = 'Validation Prediction'
                         ))
    
fig.update_layout(title_text='Predictions - W/ Downtime',
                  annotations=[dict(x=250000, y=11000, xref="x", yref="y",
                                    text='Train RMSE: {:.2f}, Test RMSE:{:.2f}'.format(trainScore, testScore))])
#note to use with last method
fig.update_layout(title_text='Predictions - W/ Downtime [Train, Validation, Test]',
                  annotations=[dict(x=250000, y=11000, xref="x", yref="y",
                                    text='Train RMSE: {:.2f}, Val. RMSE:{:.2f}, Test RMSE:{:.2f}'.format(trainScore, valScore, testScore))])

    
plot(fig, auto_open=True)


"""
Discuss Plotting:
1) Plotting w/o downtime (I replaced the downtime values with too smooth values)
    - Follows the model very closly, sits slightly below true values
    - Can see where downtimes were originally kicked off - is this okay?
    - Seems too accurate to me, not sure what is going on
    - Train RMSE = 90.30, Test RMSE = 19.17
2) Plotting w/ downtime (Using the true volume)
    - Follows the model very closly, sits slightly above true values
    - RMSE is a little worse on the test set this go than with the dropped downtime
    - Train RMSE = 82.55, Test RMSE = 90.03
3) Plotting w/ downtime (70% train, 15% Validate, 15% Test)
    - Follows the model very closely. Still too closely for me to be comfortable with
    - Sits slightly below
    - Ask Luis about this?
    - Train RMSE = 69.12, Val RMSE = 69.12, Test RMSE = 42.22
"""




# Proba predict 
# We predict the probability of failure as 1 - Pr of output
PrtestPredict =  1 - model.predict_proba(test_X)


############################ Not used / before refining code  #################

#method to change to numeric if too many string variables.
for i in int_columns.keys():
    i_s = '{}'.format(i)
    data[i_s] = pd.to_numeric(data[i_s], errors = 'coerce', downcast = 'float')

data.dtypes
# Out[38]: 
# Type of Facility                            object
# Name of Facility                            object
# Timestamp                           datetime64[ns]
# Wellhead Casing "A" - Pressure             float32
# Wellhead Casing "B" - Pressure             float32
# Flowline Pressure                          float32
# Flowline Temperature                       float32
# Volume - Calendar Day Production           float32
# Wellhead Tubing - Pressure                 float32

#count the null values
data.isnull().sum() #60 columns have null values

null_idx = data[data['Wellhead Tubing - Pressure'].isnull()].index
data.loc[null_idx, :]
#In looking at the data - null_idx == 'Time is invalid'

#plot data with Matplot
fig, ax = plt.subplots(figsize = (15, 10))
ax.plot(data_volume['Timestamp'], data_volume['Volume - Calendar Day Production'])
ax.set_title('Time Series of Volume per Day')
ax.set_xlabel('Time')
ax.set_ylabel('Volume/Day')

#look at span for exponential weighted average
best = {}
for i in [15,30,45, 60]:
    RMSE = np.sqrt(mean_squared_error(volume, volume.ewm(span = i).mean()))
    error_size = RMSE/volume.mean()
    best[error_size] = (i, RMSE)

"""
Looking at exponential time averages
"""
#Look at the lowest relative size errors to mean of all the values
use = list(best.keys())
df_exp_avg = {}
for i in use:
    t = best[i][0]
    df_exp_avg['exp_span_{}'.format(t)] = volume.ewm(span = t).mean()

df_exp_avg['True_Volume'] = volume
df_exp_avg['Timestamp'] = data_volume['Timestamp']
df_exp_avg = pd.DataFrame(df_exp_avg)

#Plot the exponential traces against eachother
colors = ['black', 'deepskyblue', 'green', 'gold', 'red']#, 'orange']
columns = list(df_exp_avg.columns)[1:]
fig = go.Figure()
for i in range(len(colors)):
    fig.add_trace(go.Scatter(x=df_exp_avg.Timestamp[:10000], y=df_exp_avg[columns[i]][:10000], name="Volume",
                         line_color=colors[i]))
    
fig.update_layout(title_text='Time Series With Exponential Averages',
                  xaxis_rangeslider_visible=True)

plot(fig, auto_open=True)

"""
Simple tracker of downtime - got 90 of them
"""
#Track the down times
#look at length of downtime
tracker_drop = {}
#tracker_drop { start_idx { length: val , avg_value: val, min_vol: val}}

count = 0
end_index = None
cv_tracker = []
for i in dv_index:
    cv = data_volume.loc[i,'Volume - Calendar Day Production']

    #search for a value under 6000
    if cv < 4000:
        if count == 0:
            start_index = i
        count += 1
        cv_tracker += [cv]
    elif cv > 6000:
        #save the values and reset
        if count != 0:
            tracker_drop[start_index] = {'length (hr)': len(cv_tracker)/60,
                                         'avg_val Production': np.mean(cv_tracker),
                                         'min_val Production': min(cv_tracker)}
            if end_index == None:
                tracker_drop[start_index]['Since Previous'] = 0
            else:
                tracker_drop[start_index]['Since Previous'] = (start_index - end_index)/60
            end_index = start_index + count
            cv_tracker = []    
            count = 0

downtime_tracker = pd.DataFrame.from_dict(tracker_drop, orient = 'index')
downtime_tracker.describe()
#        length (hr)  avg_val Production  min_val Production  Since Previous
# count    90.000000           90.000000           90.000000       89.000000
# mean      2.130000         3180.478211         2620.469717       63.360300
# std       2.963396         2176.943975         2484.686450       90.506214
# min       0.233333            0.010214            0.000000        0.233333
# 25%       0.250000          728.298482            0.000000        1.766667
# 50%       1.008333         3555.247161         3238.122650       26.000000
# 75%       2.500000         5134.879650         5113.535875       92.000000
# max      14.516667         5977.735827         5976.704600      488.500000

