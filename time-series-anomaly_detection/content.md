# Unsupervised anomaly detection in time series


## Install Anaconda on your computer

Download and install Anaconda. Select the Python 3.* version):
https://www.anaconda.com/download/

When Anaconda is installed, open "Anaconda Prompt" or any other terminal where you have ```conda``` available now.

## Make an isolated Python environment
Run ```conda create --name anomaly-detection python=3.6``` in your terminal.
Then, to activate your new environment, run ```conda activate anomaly-detection```.


##  Install the required packages

Run ```pip install mlflow[extras]==1.1.0 mflux-ai matplotlib==3.0.3  fbprophet==0.3.post2```  in your terminal.

## Tutorial

In this tutorial, we will create an unsupervised anomaly detection model for time series. Anomaly detection problem for time series can be formulated as finding outlier data points relative to some standard or usual signal.
It will input a time series and detect abnormal time series points. We will focus on unsupervised learning, i.e we do not have labeled data points. We will use
the prophet library for modeling our time series.

## The data set

The dataset
The data set contains  energy consumption readings in kWh (per half hour) for a sample of London Households that took part in that took part in the
UK Power Networks led Low Carbon London project between November 2011 and February 2014. More information about the project can be found
 [here](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households?resource=3527bf39-d93e-4071-8451-df2ade1ea4f2)

The data set is publicly available at https://data.london.gov.uk/download/smartmeter-energy-use-data-in-london-households/04feba67-f1a3-4563-98d0-f3071e3d56d1/Power-Networks-LCL-June2015(withAcornGps).csv_Pieces.zip

## Library imports
```python
import fbprophet
import matplotlib.pyplot as plt
import pandas as pd
```

## Import data
Import the data set into a pandas dataframe
```python
df = pd.read_csv('Power-Networks-LCL-June2015(withAcornGps)v2_1.csv', header=0)

```

## Data pre-processing

Convert the date string field into a pandas datetime type
```python
df['date'] = pd.to_datetime(df['DateTime'])
```

Extract date and consumption columns and set the date as the index.


```python
data = df.loc[:, ['KWH/hh (per half hour) ']]
data = data.set_index(df.date)
```

Next, we will resample the time series. Resampling involves changing the frequency of your time series observations.


```python
data['KWH/hh (per half hour) '] = pd.to_numeric(data['KWH/hh (per half hour) '], downcast='float', errors='coerce')
daily = data.resample('D').sum()
daily.reset_index(inplace=True)
daily = daily.rename(columns={'date': 'ds', 'KWH/hh (per half hour) ': 'y'})
daily['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))```

```



### The whole code:
```python
import fbprophet
import matplotlib.pyplot as plt
import pandas as pd


def detect_anomalies(forecast):
    forecasted = forecast[['ds', 'trend', 'yhat', 'yhat_lower', 'yhat_upper', 'actual']].copy()

    forecasted['anomaly'] = 0
    forecasted.loc[forecasted['actual'] > forecasted['yhat_upper'], 'anomaly'] = 1
    forecasted.loc[forecasted['actual'] < forecasted['yhat_lower'], 'anomaly'] = 1

    return forecasted


def plot_anomalies(forecasted):
    ax = plt.gca()

    ax.plot(forecasted['ds'].values, forecasted['actual'].values, 'b-')

    ax.scatter(forecasted[forecasted['anomaly'] == 1]['ds'].values,
               forecasted[forecasted['anomaly'] == 1]['actual'].values, color='red')

    ax.fill_between(forecasted['ds'].values, forecasted['yhat_lower'].values, forecasted['yhat_upper'].values,
                    alpha=0.3, facecolor='r')

    plt.show()



df = pd.read_csv('Power-Networks-LCL-June2015(withAcornGps)v2_1.csv', header=0)
df['date'] = pd.to_datetime(df['DateTime'])

data = df.loc[:, ['KWH/hh (per half hour) ']]
data = data.set_index(df.date)
data['KWH/hh (per half hour) '] = pd.to_numeric(data['KWH/hh (per half hour) '], downcast='float', errors='coerce')
daily = data.resample('D').sum()
daily.reset_index(inplace=True)

daily = daily.rename(columns={'date': 'ds', 'KWH/hh (per half hour) ': 'y'})
daily['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))

# Make prohet model and fit on the data
model = fbprophet.Prophet(changepoint_prior_scale=0.1)
model.fit(daily)

forecast = model.predict(daily)

forecast['actual'] = daily['y'].reset_index(drop=True)

forecast = detect_anomalies(forecast)
print("Anomalies")
n_obs = forecast.shape[0]
n_anomalies = forecast['anomaly'].sum()
n_normal = n_obs - n_anomalies
print("There are {} anomalies and {} normal data points. {} % of the data points are anomalies.".format(n_anomalies,
                                                                                                        n_normal,
                                                                                                        int((n_anomalies / n_normal) * 100)))

plot_anomalies(forecast)

```
