# Analysis and Forecasting on Boston Data
We do an exploratory data analysis on a Boston crime dataset from Kaggle, focusing on crime over time. Then we merge this with
a Boston weather dataset and try to forecast crimes per hour.


## Set up dependencies
Use Anaconda to install
* `conda install "python=3.6.6" "scipy=1.1" "h5py<3" "tensorflow=1.14" "Keras=2.2" "scikit-learn=0.20"`

* `pip install mlflow[extras]==1.2.0 "mflux-ai>=0.3.0"`


## Download datasets
At your preferred location, create a folder called `boston_crime`.

From [Kaggle](https://www.kaggle.com), download the datasets [crime.csv and offense_codes.csv](https://www.kaggle.com/AnalyzeBoston/crimes-in-boston)
concerning crimes in Boston between 2015 and 2018, as well as the dataset [Boston weather](https://www.kaggle.com/jqpeng/boston-weather-data-jan-2013-apr-2018).

Extract/paste these sets into files
* `raw_crime_data.csv`
* `offense_codes.csv`
* `raw_weather_data.csv`
in the folder `boston_crime`. The file `offense_codes.csv` is not strictly necessary, but is nice to have for context.


## Prepare data
Make a file called `prepare_data.py`. Firstly, we will use the following imports.
```python
import joblib
import mflux_ai
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# load raw crime and weather data
crime_data = pd.read_csv('raw_crime_data.csv', encoding='windows-1252')
weather_data = pd.read_csv('raw_weather_data.csv')
```

The raw crime data is a bit unwieldy. Let's clean it up a little and save a prettier version in case we want to look at it.
```python
# select only necessary columns from crime data and do some relabeling
crime_data = crime_data[['OFFENSE_CODE_GROUP', 'SHOOTING', 'OCCURRED_ON_DATE', 'DAY_OF_WEEK', 'HOUR', 'UCR_PART']]
crime_data.UCR_PART = crime_data.UCR_PART.replace(['Part One', 'Part Two', 'Part Three', 'Other', np.nan],
                                                  ['One', 'Two', 'Three', 'Other', 'NR'])
crime_data.SHOOTING = crime_data.SHOOTING.replace([np.nan, 'Y'], [0, 1])
crime_data = crime_data.rename(columns={'OFFENSE_CODE_GROUP': 'Type', 'SHOOTING': 'Shooting', 'OCCURRED_ON_DATE': 'Date',
                                        'DAY_OF_WEEK': 'Weekday', 'HOUR': 'Hour', 'UCR_PART': 'UCR'})
# save the new crime dataframe
crime_data.to_csv(r'C:/Users/Erlend/Documents/GitHub/erlendAI/boston_crime/tutorial/crime_data.csv',
                  index=False)
```

And similarly for the weather data.
```python
# restructure and relabel the weather data
weather_data = weather_data.rename(columns={'Year': 'year', 'Month': 'month', 'Day': 'day'})
weather_data['Date'] = pd.to_datetime(weather_data[['year', 'month', 'day']])
weather_columns = ['Date', 'Avg Temp (F)', 'Avg Sea Level Press (in)', 'Avg Visibility (mi)',
                   'Low Visibility (mi)', 'High Wind (mph)', 'High Wind Gust (mph)']
weather_data = weather_data[weather_columns]
weather_data = weather_data.rename(columns={'Avg Temp (F)': 'Temp', 'Avg Sea Level Press (in)': 'Pressure',
                                            'Avg Visibility (mi)': 'Avg Vis', 'Low Visibility (mi)': 'Low Vis',
                                            'High Wind (mph)': 'Wind', 'High Wind Gust (mph)': 'Wind Gust'})
weather_data = weather_data.set_index('Date')
# save tidy weather data
crime_data.to_csv(r'C:/Users/Erlend/Documents/GitHub/erlendAI/boston_crime/tutorial/weather_data.csv',
                  index=False)
```

We have two dataframes. The crime data has individual crimes as rows and various information as columns. 'Date' is a column and
the various types of crime are entries in the column 'Type'. We want the input data for a network in a shape more like the
weather data, with time as the row index and each type of crime as its own column. The entries will be number of occurrences 
per timestep. To make this dataframe, we will prepare the row index and column names, make an empty dataframe, and then fill 
it in.

Prepare a list of datetimes spanning the crime data. We will make one frame with intervals of one hour, and one with intervals of
one day.
```python
crime_data.Date = pd.to_datetime(crime_data.Date)
crime_data.Date = crime_data.Date.dt.floor('1H')
# prepare row index of time series dataframe
start_date = crime_data.Date.sort_values().iloc[0]
weather_data = weather_data.loc[start_date:]
end_date = crime_data.Date.sort_values().iloc[-1]
hours = pd.date_range(start_date, end_date, freq='H')
days = pd.date_range(start_date, end_date, freq='D')
```

We assume that less common crimes will impact our predictions less. To not get to many columns, we define an arbitrary cutoff 
for what crimes to add. How to choose these might be worth examining more later.
```python
# select most common crimes to index time series columns
cutoff = 1000
types = crime_data.Type.value_counts()
columns = []
for e in types.index:
    if types[e] > cutoff:
        columns.append(e)
```

We create the hourly and daily dataframes.
```python
# crete time series dataframe skeletons
crime_by_hour = pd.DataFrame(index=hours, columns=columns)
crime_by_day = pd.DataFrame(index=days, columns=columns)

# find and add nonzero cells in time series dataframes
for col in columns:
    crime_by_hour[col] = crime_data[crime_data.Type == col].Date.value_counts()
    crime_by_day[col] = crime_data[crime_data.Type == col].Date.dt.floor('1D').value_counts()
crime_by_hour = crime_by_hour.fillna(0)
crime_by_day = crime_by_day.fillna(0)
```

We add the 'Total' column of the sum of all crimes and some data about hour, day and month to let our model capture time-cycles 
more easily.
```python
# add some useful columns
crime_by_hour['Total'] = crime_by_hour.sum(axis=1)
crime_by_hour['Hour'] = crime_by_hour.index.hour
crime_by_hour['Weekday'] = crime_by_hour.index.weekday
crime_by_hour['Monthday'] = crime_by_hour.index.day
crime_by_hour['Month'] = crime_by_hour.index.month
crime_by_day['Total'] = crime_by_day.sum(axis=1)
crime_by_day['Hour'] = crime_by_day.index.hour
crime_by_day['Weekday'] = crime_by_day.index.weekday
crime_by_day['Monthday'] = crime_by_day.index.day
crime_by_day['Month'] = crime_by_day.index.month
```

The dataframe `hourly_input` only has weather for the first hour per day, and both dataframes have data for some months past
the last weather data. We fix this with
```python
# finish updating dataframes
hourly_input = crime_by_hour.join(weather_data).fillna(method='ffill')
daily_input = crime_by_day.join(weather_data).dropna()
last_weather_hour = daily_input.iloc[-1].name + pd.to_timedelta('23 hours')
hourly_input = hourly_input[:last_weather_hour]
```

Save these dataframes, both to file (if we want to look at them quickly) and to Mflux (if we want to share them easily)
```python
# save time series dataframes
mflux_ai.init("hkPqTCtv-cUsROULi0Aizg")
hourly_input.to_csv(r'C:/Users/Erlend/Documents/GitHub/erlendAI/boston_crime/tutorial/hourly_data.csv', index=False)
mflux_ai.put_dataset(hourly_input, "hourly_input.pkl")
daily_input.to_csv(r'C:/Users/Erlend/Documents/GitHub/erlendAI/boston_crime/tutorial/daily_data.csv', index=False)
mflux_ai.put_dataset(daily_input, "daily_input.pkl")
```

Currently, our data is of various different sizes, for example 'verbal disputes' usually has the values 0 or 1 and 'Avg Temp'
often has values over 50. We want to standardize the input so that every column has mean 0 and variance 1.
```python
# Standardize dataframes
hourly_scaler = StandardScaler()
hourly_scaler.fit(hourly_input)
hourly_standardized = hourly_scaler.transform(hourly_input)
hourly_standardized = pd.DataFrame(data=hourly_standardized, index=hourly_input.index, columns=hourly_input.columns)

daily_scaler = StandardScaler()
daily_scaler.fit(daily_input)
daily_standardized = daily_scaler.transform(daily_input)
daily_standardized = pd.DataFrame(data=daily_standardized, index=daily_input.index, columns=daily_input.columns)
```

Finally, we save the prepared dataframes to our folder as well as to Mflux. We also save the scaler used to standardize the data, 
so we can invert this operation after we have made our prediction.
```python
# save standardized dataframes
hourly_standardized.to_csv(r'C:/Users/Erlend/Documents/GitHub/erlendAI/boston_crime/tutorial/hourly_standardized.csv',
                           index=False)
mflux_ai.put_dataset(hourly_standardized, "hourly_standardized.pkl")
mflux_ai.put_dataset(hourly_scaler, "hourly_scaler.pkl")
daily_standardized.to_csv(r'C:/Users/Erlend/Documents/GitHub/erlendAI/boston_crime/tutorial/daily_standardized.csv',
                          index=False)
daily_standardized_filename = "my-dataset.pkl"
mflux_ai.put_dataset(daily_scaler, "daily_scaler.pkl")
mflux_ai.put_dataset(daily_standardized, "daily_standardized.pkl")
```

















## Some Data Exploration
Create a file called `explore_data.py`. The following code shows many bar graphs of crime broken down over time, such as 
crime by hour of the day or day of the week. You may want to comment out large parts if you want to investigate one case in 
particular. Use ctrl + C if you accidentally start a long queue of displays. Notice the strong daily cycle, and the interesting 
monthly behaviour (possibly a lot of paperwork gets filed the first day every month).
```python
import pandas as pd
from matplotlib import pyplot

crime_data = pd.read_csv('crime_data.csv', encoding='windows-1252')
crime_and_weather_data = pd.read_csv('crime_and_weather_data.csv', encoding='windows-1252')
count_cutoff = 5000


print('HOURLY COUNT')
print('Total crime')
hourly_count = crime_data.Hour.value_counts()
pyplot.bar(hourly_count.index, hourly_count)
pyplot.show()
type_count = crime_data.Type.value_counts()
for e in type_count.index:
    if type_count[e] > count_cutoff:
        print(e)
        specific_hourly_count = crime_data[crime_data.Type == e].Hour.value_counts()
        pyplot.bar(specific_hourly_count.index, specific_hourly_count)
        pyplot.show()


print(' ')
print('WEEKLY COUNT')
print('Total crime')
crime_data.Weekday = crime_data.Weekday.replace(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
                                                 'Sunday'], [1, 2, 3, 4, 5, 6, 7])
daily_count = crime_data.Weekday.value_counts()
pyplot.bar(daily_count.index, daily_count)
pyplot.show()
type_count = crime_data.Type.value_counts()
for e in type_count.index:
    if type_count[e] > count_cutoff:
        print(e)
        specific_daily_count = crime_data[crime_data.Type == e].Weekday.value_counts()
        pyplot.bar(specific_daily_count.index, specific_daily_count)
        pyplot.show()

print(' ')
print('MONTHLY COUNT')
print('Total crime')
crime_data['Date'] = pd.to_datetime(crime_data['Date'])
crime_data['Monthday'] = crime_data.Date.dt.day
monthday_count = crime_data.Monthday.value_counts()
pyplot.bar(monthday_count.index, monthday_count)
pyplot.show()
type_count = crime_data.Type.value_counts()
for e in type_count.index:
    if type_count[e] > count_cutoff:
        print(e)
        specific_monthday_count = crime_data[crime_data.Type == e].Monthday.value_counts()
        pyplot.bar(specific_monthday_count.index, specific_monthday_count)
        pyplot.show()

print(' ')
print('YEARLY COUNT')
print('Total crime')
crime_data['Date'] = pd.to_datetime(crime_data['Date'])
crime_data['Month'] = crime_data.Date.dt.month
monthly_count = crime_data.Month.value_counts()
pyplot.bar(monthly_count.index, monthly_count)
pyplot.show()
type_count = crime_data.Type.value_counts()
for e in type_count.index:
    if type_count[e] > count_cutoff:
        print(e)
        specific_monthly_count = crime_data[crime_data.Type == e].Month.value_counts()
        pyplot.bar(specific_monthly_count.index, specific_monthly_count)
        pyplot.show()
```

This code draws a correlation grid of all the columns. _IMPROVE THIS PART_
```python
corr = crime_and_weather_data.corr()
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(crime_and_weather_data.columns),1)
ax.set_xticks(ticks)
pyplot.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(crime_and_weather_data.columns)
ax.set_yticklabels(crime_and_weather_data.columns)
pyplot.show()
```

## Foracasting with RNN
Now we build and train a small recurrent neural network. Make a file called `hourly_model.py` where we will use the hourly
data we have prepared. There are a few dependencies to import. Keras helps us build the network, sklearn (scikitlearn) 
is used to measure error, and of course mflux_ai to log data and models over several runs.
```python
import mflux_ai
import mlflow.sklearn
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, CuDNNLSTM
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error as MSE
```

Initalize Mflux and load the dataset
```python
mflux_ai.init("hkPqTCtv-cUsROULi0Aizg")
data = mflux_ai.get_dataset("hourly_standardized.pkl")
```

Allocate memory. _EXAMINE THIS PART, IS MEMORY REALLY THE PROBLEM?_
```python
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
sess = tf.Session(config=config)
```

We create some variables to make the code flexible. The column "Total" will be our target to predict. We can change the size
of the validation and test data here
```python
columns = list(data.columns)
target_pos = data.columns.get_loc("Total")
data = np.asarray(data)
input_width = data.shape[1]
output_width = 1
data_size = len(data)
val_size = 4000
test_size = 4000
train_size = data_size - val_size - test_size
shape = (1, -1, input_width)
```

Split the data into training, validation and test data with input and target
```python
x_train = data[:train_size, :]
x_train = np.reshape(x_train, shape)
y_train = x_train[:, 1:, target_pos:target_pos + 1]
x_train = x_train[:, :-1]

x_val = data[train_size:train_size + val_size, :]
x_val = np.reshape(x_val, shape)
y_val = x_val[:, 1:, target_pos:target_pos + 1]
x_val = x_val[:, :-1]

x_test = data[train_size + val_size:train_size + val_size + test_size, :]
x_test = np.reshape(x_test, shape)
y_test = x_test[:, 1:, target_pos:target_pos + 1]
x_test = x_test[:, :-1]
```

We will create a loop that tests several variations for some networks. Therefore it is good to standardize how we build them.
Here `no2, no3` are unused variables, but we will want all networks to take the same amount of inputs. These unused variables 
could of course be used to set some hyperparameter, like learning rate (`optimizer = Adam(lr=0.003)`). Here is a network with one 
hidden layer of fully connected nodes.
```python
def DENSE1(nodes1, no2, no3):
    optimizer = Adam(lr=0.003)
    model = Sequential()
    model.add(Dense(nodes1, input_shape=(None, input_width)))
    model.add(Dense(output_width))
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()
    stop = EarlyStopping(patience=20)
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
    model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val), callbacks=[stop, checkpoint])
    return model, nodes1, 0, 0
```

And here are a few more options to try.
```python
def DENSE2(nodes1, nodes2, no3):
    optimizer = Adam(lr=0.003)
    model = Sequential()
    model.add(Dense(nodes1, input_shape=(None, input_width)))
    model.add(Dense(nodes2))
    model.add(Dense(output_width))
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()
    stop = EarlyStopping(patience=20)
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
    model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val), callbacks=[stop, checkpoint])
    return model, nodes1, nodes2, 0


def LSTM1(nodes1, no2, no3):
    optimizer = Adam(lr=0.003)
    model = Sequential()
    model.add(CuDNNLSTM(nodes1, input_shape=(None, input_width), return_sequences=True))
    model.add(Dense(output_width))
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()
    stop = EarlyStopping(patience=20)
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
    model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val), callbacks=[stop, checkpoint])
    return model, nodes1, 0, 0


def LSTM2(nodes1, nodes2, no3):
    optimizer = Adam(lr=0.003)
    model = Sequential()
    model.add(Dense(nodes1, input_shape=(None, input_width)))
    model.add(CuDNNLSTM(nodes2, return_sequences=True))
    model.add(Dense(output_width))
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()
    stop = EarlyStopping(patience=20)
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
    model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val), callbacks=[stop, checkpoint])
    return model, nodes1, nodes2, 0


def LSTM3(nodes1, nodes2, no3):
    optimizer = Adam(lr=0.003)
    model = Sequential()
    model.add(CuDNNLSTM(nodes1, input_shape=(None, input_width), return_sequences=True))
    model.add(Dense(nodes2))
    model.add(Dense(output_width))
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()
    stop = EarlyStopping(patience=20)
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
    model.fit(x_train, y_train, epochs=300, validation_data=(x_val, y_val), callbacks=[stop, checkpoint])
    return model, nodes1, nodes2, 0


def LSTM4(nodes1, nodes2, nodes3):
    optimizer = Adam(lr=0.003)
    model = Sequential()
    model.add(Dense(nodes1, input_shape=(None, input_width)))
    model.add(CuDNNLSTM(nodes2, return_sequences=True))
    model.add(Dense(nodes3))
    model.add(Dense(output_width))
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()
    stop = EarlyStopping(patience=20)
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
    model.fit(x_train, y_train, epochs=300, validation_data=(x_val, y_val), callbacks=[stop, checkpoint])
    return model, nodes1, nodes2, nodes3
```
_FIGURE OUT HOW TO FIX OR HANDLE THE CRASHING ON DEEPER LSTM NETWORKS!_

The following loop lets us try a few configurations of a certain network in one run, and log all resulting models to Mflux
```python
experiment_id = mlflow.set_experiment("Boston_Hourly")
for i in range(3):
    for j in range(1):
        for k in range(1):
            K.clear_session()
            nodes1, nodes2, nodes3 = 10 + 10*i, 10 + 10*j, 10 + 10*k
            model, nodes1, nodes2, nodes3 = DENSE1(nodes1, nodes2, nodes3)
            # print some information about each model
            network_prediction = model.predict(x_test)
            network_prediction = network_prediction.reshape((-1))
            y_test = y_test.reshape(-1)
            network_score = MSE(network_prediction, y_test)
            print('Our prediction has MSE: ', network_score)
            print('With hidden layer nodes ', nodes1, nodes2, nodes3)
            # log model to mflux
            y_test = y_test.reshape(1, -1, 1)
            model_score = model.evaluate(x_test, y_test, batch_size=2)
            run_number = str(nodes1) + ', ' + str(nodes2) + ', ' + str(nodes3)
            with mlflow.start_run(experiment_id=experiment_id, run_name=run_number):
                mlflow.log_metric("mse", model_score)
                mlflow.log_param("model_type", model.__class__.__name__)
                mlflow.sklearn.log_model(model, "model")
```

Using the above models we got mean squared errors of around 0.43 in our best test runs.


## Retrieving and applying the best models
_Automate finding the best x models and display their score and a graph against reality_
<!---
```python
mflux_ai.init("hkPqTCtv-cUsROULi0Aizg")
data = mflux_ai.get_dataset("hourly_standardized.pkl")
```








## Fetching and Examining the Best Model
When want to fecth and examine the best model from mflux. We recycle some code to handle the standardization and its 
inverse, and then make some simple predictions to evaluate against.
```python
import mlflow.sklearn
import mflux_ai
import pandas as pd
from sklearn.preprocessing import StandardScaler
from math import sqrt
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from matplotlib import pyplot

mflux_ai.init("your_project_token_goes_here")

model1 = mlflow.sklearn.load_model("s3://mlflow/0/your_MODEL_ID_GOED_HERE/artifacts/model")
data = pd.read_csv('hourly_input.csv', encoding='windows-1252')

print(data.shape)
columns = list(data.columns)
target_pos = data.columns.get_loc("Total")
data = np.asarray(data)
input_width = data.shape[1]
output_width = 1
data_size = len(data)
val_size = 3000
test_size = 3000
train_size = data_size - val_size - test_size
shape = (1, -1, input_width)

scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)

x_test = data[train_size + val_size:train_size + val_size + test_size, :]
x_test = np.reshape(x_test, shape)
y_test = x_test[:, 1:, target_pos:target_pos + 1]
x_test = x_test[:, :-1]

y_mean = scaler.mean_[target_pos]
y_var = scaler.var_[target_pos]
y_std = sqrt(y_var)

# Make various baseline predictions to evaluate
network_prediction = model1.predict(x_test)
network_prediction = network_prediction.reshape((-1)) * y_std + y_mean
y_test_post = y_test * y_std + y_mean
y_test_post = y_test_post.reshape(-1)
network_score = MSE(network_prediction, y_test_post)
x_test_post = scaler.inverse_transform(x_test)
data = scaler.inverse_transform(data)

repeat_hour_prediction = x_test_post[:, :, target_pos:target_pos + 1].reshape(-1)
repeat_day_prediction = data[train_size + val_size - 23:train_size + val_size + test_size - 24,
                        target_pos:target_pos + 1].reshape(-1)
repeat_week_prediction = data[train_size + val_size - 167:train_size + val_size + test_size - 168,
                        target_pos:target_pos + 1].reshape(-1)
avg_prediction = np.asarray([np.mean(y_test_post, axis=0)] * len(y_test_post))

repeat_yesterday = MSE(repeat_day_prediction, y_test_post)
repeat_hour_score = MSE(repeat_hour_prediction, y_test_post)
repeat_week_score = MSE(repeat_week_prediction, y_test_post)
avg_score = MSE(avg_prediction, y_test_post)

print('Our prediction has MSE: ', network_score)
print('Predicting the average value for each type has MSE: ', avg_score)
print('Predicting no change from the same time last week has MSE: ', repeat_week_score)
print('Predicting no change from the same time yesterday has MSE: ', repeat_yesterday)
print('Predicting no change has MSE: ', repeat_hour_score)

pyplot.plot(y_test_post[-50:], color='red')
pyplot.plot(network_prediction[-50:], color='green')
pyplot.show()
```
--->

## Final thoughts
We started with a dataset and no set goal other than to do something useful with it. There are many paths to take right 
from the start. For example
* Try adding rarer crimes to dataset. Maybe rare, but serious, crimes have higher predictive power
* Try removing columns as well, aim for the smallest version that has full predictive power
* Forecast further ahead with iterative forecasting
* Improve the baseline predictions by making new ones and doing weighted averages of old ones
* Add new layers to the network
* Try other types of networks, like temporal convolutional networks
* Test other hyperparameters; optimizer, activation function
* Attempt forecasting on the daily_input. It is denser, so should lend itself to better predicting types of crime
* Give the model access to the weather data for the timestep it is predicting
* "Time to event" forecasting of more rare occurrences
* Pay more attention to the location data
* Find the most crime ridden areas
* Does area correlate with type of crime?
* Find ways to preprocess the data so learning can be done more effectively on certain types of crime
* Forecast a danger rating for the various areas, maybe even broken down by important crime categories
* What types of crimes correlate most with each other? Grouping them will help learning
* Kaggle has many datasets of Boston; Housing, Airbnb, Public Schools and more. What useful ways can they be combined?
