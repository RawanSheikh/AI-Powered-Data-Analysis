import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('weather.csv')
df['Average Temperature Celsius'] = (df['Data.Temperature.Avg Temp'] - 32) / 1.8
df['Precipitation mm'] = df['Data.Precipitation'] * 25.4
df

# df_sf` contains observations only from San Francisco
df_sf = df[df['Station.City'] == 'San Francisco'].copy()

plt.figure(figsize=(10, 5))
plt.plot(pd.to_datetime(df_sf['Date.Full'], errors='coerce'), df_sf['Average Temperature Celsius'], linestyle='-', marker='o')

plt.title('Daily Average Temperature in San Francisco')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)

plt.show()