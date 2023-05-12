from __future__ import absolute_import, division, print_function, unicode_literals

from prophet.plot import plot_plotly

from prophet.serialize import model_to_json
from prophet.plot import plot_plotly
import pandas as pd  
from prophet import *
import numpy as np
import plotly.express as px



forecast = pd.read_csv('forecast.csv')
df = pd.read_csv('model/london_merged.csv')
df['timestamp'] =  pd.to_datetime(df['timestamp'])
col = 'cnt'

# df = df[[col , 'timestamp']]
df['timestamp'] =  pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')
df = df.resample('d').max()
df=df.dropna(axis=0)
df = df.reset_index()
df.drop_duplicates(inplace=True)
df['ds'] = df['timestamp']
# df['y'] = df['t1']
df = df.rename({col : 'y'}, axis = 'columns')
df=df[['ds', 'y']]
forecast=forecast[(forecast['ds']< '2017-01-04')]
forecast['ds'] =  pd.to_datetime(forecast['ds'])
results=pd.concat([df.set_index('ds')['y'],forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']]],axis=1)
results['error'] = results['y'] - results['yhat']
results["uncertainty"] = results['yhat_upper'] - results['yhat_lower']
results['anomaly'] = results.apply(lambda x: 'Yes' if(np.abs(x['error']) >  1.5*x['uncertainty']) else 'No', axis=1)
results.to_csv('model/anomaly.csv', index=False)




