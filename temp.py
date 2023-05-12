
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from matplotlib.animation import FuncAnimation

df = pd.read_csv('london_merged.csv')

df['timestamp'] = pd.to_datetime(df['timestamp'])

col = 'cnt'


# df = df[[col , 'timestamp']]

df['timestamp'] =pd.to_datetime(df['timestamp'])

df = df.set_index('timestamp')

df = df.resample('d').max()

df=df.dropna(axis=0)

df = df.reset_index()

df.drop_duplicates(inplace=True)

df['ds'] = df['timestamp']

# df['y'] = df['t1']

df = df.rename({col : 'y'}, axis = 'columns')



df=df[['ds','y']]
df['year'] = df['ds'].dt.year
grouped_df = df.groupby('year')

# Calculate the angle in radians for each date
df['angle'] = np.linspace(0, 2 * np.pi, len(df))

# Create the polar plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)

# Plot the data for each year
for year, data in grouped_df:
    ax.plot(data['angle'], data['y'], label=str(year))

# Customize the plot
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)
ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 6))
ax.set_xticklabels(['0', '30', '60', '90', '120', '150', '180', '210', '240', '270', '300', '330'])
ax.set_title('Polar Plot', pad=20)
ax.legend()

# Animation for continuous movement
def update_plot(frame):
    ax.clear()
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 6))
    ax.set_xticklabels(['0', '30', '60', '90', '120', '150', '180', '210', '240', '270', '300', '330'])
    ax.set_title('Polar Plot', pad=20)
    
    for year, data in grouped_df:
        ax.plot(data['angle'][:frame], data['y'][:frame], label=str(year))
    
    ax.legend()

ani = FuncAnimation(fig, update_plot, frames=len(df), interval=200)

# Show the plot
plt.show()
