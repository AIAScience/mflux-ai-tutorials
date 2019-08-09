# Time series forecasting tutorial

_Add description, or link to one, of how to set up software_

In this tutorial we will get started with time series forecasting using autoregression (AR) models.
 
_To begin with, it should be pointed out that with time series forecasting it is often very difficult to
achieve relative errors as low as could be expected in many other machine learning tasks._

#### Setting it up
_Download Anaconda and use it to get python, numpy, sklearn, pandas. Install mlflow.._

_In mlflow/examples create a folder called time_series_tutorial._
_In this create a file called sunspots.csv and a python file called tutorial.py_

_Open Anaconda Prompt and cd into mlflow/examples_

#### Download the dataset
Data sets for time series are typically a lot smaller than for other ML methods.  
To download this one, you can simply go to
_https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv,
select all, copy the whole thing and paste it into sunspots.csv._

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
for as long in to the future as we ask of it.
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
We will want pyplot to visualise the output and MSE as a measure of accuracy.
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

_plutti plutti SS_naive_500_10_
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
gimmi speis
