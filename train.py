


def train_model():
  
  df = pd.read_csv('london_merged.csv') #latest data

  df['timestamp'] =  pd.to_datetime(df['timestamp'])
  col = 'cnt'

  df = df[[col , 'timestamp']]
  df['timestamp'] =  pd.to_datetime(df['timestamp'])
  #-------

  df = df.set_index('timestamp')
  df = df.resample('d').max()
  df=df.dropna(axis=0)
  df = df.reset_index()
  #---------

  df['ds'] = df['timestamp']
  df = df.rename({col : 'y'}, axis = 'columns')


  m = Prophet()

  # m.add_seasonality(name='yearly', period=365, fourier_order=20)

  m.fit(df)

  future = m.make_future_dataframe(periods=365)

  future['yearly'] = future['ds'].apply(lambda x: x.year - 1)
  rmse=get_perf(m , df)
  print('rmse',rmse)
  forecast = m.predict(future)
  fig=plot_plotly(m,forecast)
  fig.show()
  # plot_plotly(m,forecast) 
  # with open('model.json', 'w') as fout:
  #   fout.write(model_to_json(m))  # Save model
  # print("model saved")
  # forecast.to_csv("forecast.csv", index=False)
  # print("prediction file saved")


train_model()
