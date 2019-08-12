# Time series forecasting tutorial

In this tutorial we will get started with time series forecasting using autoregression (AR) models.

_To begin with, it should be pointed out that with time series forecasting it is often very difficult to
achieve relative errors as low as could be expected on many other machine learning tasks._

#### Setting it up

To manage our dependencies we will use Anaconda. You can get it here:

https://www.anaconda.com/download/      (make sure to get the python 3 version)

Open an Anaconda Prompt

We will need:

python

Pandas

Matplotlib

scikit-learn

_You may need administrator rights to do this_
```
conda install python pandas mathplotlib scikit-learn
```
Clone or download mlflow from https://github.com/mlflow/mlflow

In mlflow/examples create a folder called time_series_tutorial.
In this create a file called sunspots.csv and a python file called tutorial.py

Open Anaconda Prompt and cd into mlflow/examples.

#### Download the dataset
Data sets for time series are typically a lot smaller than for other ML methods.  
To download this one, you can simply go to
https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv,
select all, copy the whole thing and paste it into sunspots.csv.

#### Beginning to code
Now to import the data into our python file, type or paste this into tutorial.py:
```
from pandas import Series
series = Series.from_csv('time_series_tutorial/sunspots.csv', header=0)
```
In a time series the distance between two neighbouring points is always the same,
so we can drop the time index
```
X = series.values
```
Now X is just an array of observations, in chronological order. Run the code by opening
an Anaconda Prompt and cd into
```
mlflow/examples
```
and type
```
python time_series_tutorial/tutorial.py
```
For instance, adding
```
print(X[100:110])
```
to the bottom of the python file gives the output
```
[38.1 12.8 25.  51.3 39.7 32.5 64.7 33.5 37.6 52. ]
```
showing what X looks like without the time index

Now we want to split the data into a training set and a test set. For a time series,
the test set will be the last part of the series. Lets make a variable to control where to split the data,
and then split it
```
test_n = 500
train = X[:-test_n]
test = X[-test_n:]
```
Lets add a variable to control how many steps ahead we are to forecast.
```
forecast_steps = 10
```
To keep this tutorial simple, we will not support cases where
```
test_n / forecast_steps
```
is not an integer.

As a baseline to compare our results against, we make a naive_forecast method.
This method will always predict that the next value is the same as the last,
for as long in to the future as we ask it.
```
def naive_forecast():
	history = [e for e in train]
	predictions = []
	for i in range(0, test_n, forecast_steps):
	    forecast = [history[-1]] * forecast_steps
	    predictions.extend(forecast)
	    last_obs = test[i + forecast_steps -1]
	    history.append(last_obs)
	return predictions
```
Now, lets run a naive forecast and inspect it.
We will want pyplot to visualise the output and use mean squared error (MSE) as a measure of accuracy.
```
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
```
Then to do the forecast, give it a score with MSE and plot it
```
predictions = naive_forecast()
error = mean_squared_error(predictions, test)
print('This forecast has MSE: ', error)
pyplot.plot(test)
pyplot.plot(predictions, color="red")
pyplot.show()
```
When we now run python time_series_tutorial/tutorial.py from mlflow/examples in an Anaconda Prompt,
we should get and MSE of 1023 and this plot

![alt text](time-series-forecasting/images/SS_naive_500_10.png "this comment voted most likely to be forgotten")

Now, lets add a simple AR-model (autoregression) to forecast our series.
```
from statsmodels.tsa.ar_model import AR
```
This method implements a first stab at AR-modeling:
```
def AR_forecast():
    history = [e for e in train]
    predictions = []
    model = AR(history)
    model_fit = model.fit()
    coefs = model_fit.params
    for i in range(0, test_n, forecast_steps):
        history_this_far = history[:]
        for j in range(forecast_steps):
            forecast_value = coefs[0]
            for k in range(1, len(coefs)):
                forecast_value += coefs[k] * history_this_far[-k]
            predictions.append(forecast_value)
            history_this_far.append(forecast_value)
        history.extend(test[i:i+forecast_steps])
    return predictions
```
Lets go through it.
```
history = [e for e in train]
predictions = []
```
The history variable starts as the training data, but as we make predictions and move forward in time,
it will get appended with data from the test data that has "just become available" to us.
The predictions variable will store all our predictions.
```
model = AR(history)
model_fit = model.fit()
coefs = model_fit.params
```
Here we specify that we are using the AR model on the training data and fit it.
The variable coefs holds the coefficients calculated by the model.
These are weights to be applied to the last few elements of our time series in order to forecast the next element.
```
for i in range(0, test_n, forecast_steps):
    history_this_far = history[:]
```
We have chosen to do one forecast every few steps, and no forcasting in between.
Here we loop over the relevant time steps to do forecasting. In order to forecast more than one step,
we will use the output of one forecast as input for the next one.
Because we will be appending forecasts to the actual historical data, we make an other variable for this.
```
for j in range(forecast_steps):
    forecast_value = coefs[0]
    for k in range(1, len(coefs)):
        forecast_value += coefs[k] * history_this_far[-k]
    predictions.append(forecast_value)
    history_this_far.append(forecast_value)
```
This loop calculates all forecasts made at one point in time.
The formula is

forecast = coefs[0] + coefs[1] * history[-1] + ... + coefs[n] * history[-n]

We save the forecast and append it to history_this_far to be used as our best guess to calculate the next forecast.
```
history.extend(test[i:i+forecast_steps])
return predictions
```
Before the next forecast we add data that will by then have been observed.
Finally, to run the correct model and evaluate it we change
```
predictions = naive_forecast()
```
to
```
predictions = AR_forecast()
```
When we run the code now, we get an MSE of 638 and this plot

![alt text](time-series-forecasting/images/SS_AR_500_10.png "this comment voted most spikey")

#### The whole code:
```
from pandas import Series
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AR

series = Series.from_csv('time_series_tutorial/sunspots.csv', header=0)
X = series.values

test_n = 500
train = X[:-test_n]
test = X[-test_n:]
forecast_steps = 10


def naive_forecast():
    history = [e for e in train]
    predictions = []
    for i in range(0, test_n, forecast_steps):
        forecast = [history[-1]] * forecast_steps
        predictions.extend(forecast)
        last_obs = test[i + forecast_steps -1]
        history.append(last_obs)
    return predictions


def AR_forecast():
    history = [e for e in train]
    predictions = []
    model = AR(train)
    model_fit = model.fit()
    coefs = model_fit.params
    for i in range(0, test_n, forecast_steps):
        history_this_far = history[:]
        for j in range(forecast_steps):
            forecast_value = coefs[0]
            for k in range(1, len(coefs)):
                forecast_value += coefs[k] * history_this_far[-k]
            predictions.append(forecast_value)
            history_this_far.append(forecast_value)
        history.extend(test[i:i+forecast_steps])
    return predictions


predictions = AR_forecast()
error = mean_squared_error(predictions, test)
print('This forecast has MSE: ', error)
pyplot.plot(test)
pyplot.plot(predictions, color="red")
pyplot.show()
```


	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

