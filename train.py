
# importing necessary libraries
from __future__ import absolute_import, division, print_function, unicode_literals

from prophet.plot import plot_plotly

from prophet.serialize import model_to_json
from prophet.plot import plot_plotly
import pandas as pd  
from prophet import *
# from 

# Loading the dataset
df = pd.read_csv('london_merged.csv')

# Creating the training and testing datasets
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

# Convert 'season' column to one-hot-encoded columns
seasons = pd.get_dummies(df['season'], prefix='season')
df = pd.concat([df, seasons], axis=1)

# Drop the original 'season' column
df.drop('season', axis=1, inplace=True)
df

# Creating the Prophet model with seasonal regressors
model = Prophet(seasonality_mode='multiplicative', 
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05)

# Adding the seasonal regressors based on the categorical features
for col in seasons.columns:
    if col != 'season_1':
        model.add_seasonality(name=col, period=365/4, fourier_order=5, condition_name=col)
model.add_seasonality(name='holiday', period=365, fourier_order=10, condition_name='is_holiday')
model.add_seasonality(name='weekend', period=7, fourier_order=10, condition_name='is_weekend')


# Adding the continuous regressors
model.add_regressor('t1')
model.add_regressor('t2')



# Fitting the model to the training data
model.fit(df)

# Making predictions on the testing data
future = model.make_future_dataframe(periods=365, freq='D')
future['t1'] = df['t1']
future['t2'] = df['t2']
future['is_holiday'] = df['is_holiday']
future['is_weekend'] = df['is_weekend']
future['season_0.0'] = df['season_0.0']
future['season_1.0'] = df['season_1.0']
future['season_2.0'] = df['season_2.0']
future['season_3.0'] = df['season_3.0']
future=pd.concat([future.iloc[:730].dropna(axis=0),future.iloc[730:]])
f1['ds']=(future.iloc[730:])['ds'].copy()
# f1=future.iloc()
future.iloc[730:]=future.iloc[:365]
(future.iloc[730:])['ds']=f1['ds'].copy()
forecast = model.predict(future)
with open('model/model.json', 'w') as fout:
  fout.write(model_to_json(model))  # Save model
plot_plotly(model,forecast)


