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

# change data from long to wide format for *Boston, MA*, *Anchorage, AK*, and *Dallas-Fort Worth, TX* by temp and precipitation

df_aktxma = df[df['Station.Location'].isin(['Anchorage, AK', 'Boston, MA', 'Dallas-Fort Worth, TX'])]

df_aktxma_temperature = df_aktxma.pivot(index='Date.Full', columns='Station.Location', values='Average Temperature Celsius')
df_aktxma_temperature.index = pd.to_datetime(df_aktxma_temperature.index)
df_aktxma_temperature.columns = df_aktxma_temperature.columns.get_level_values(0)


df_aktxma_precipitation = df_aktxma.pivot(index='Date.Full', columns='Station.Location', values='Precipitation mm')
df_aktxma_precipitation.index = pd.to_datetime(df_aktxma_precipitation.index)
df_aktxma_precipitation.columns = df_aktxma_precipitation.columns.get_level_values(0)

plt.figure(figsize=(10, 6))

plt.plot(df_aktxma_temperature.index, df_aktxma_temperature['Anchorage, AK'], label='Anchorage, AK')
plt.plot(df_aktxma_temperature.index, df_aktxma_temperature['Boston, MA'], label='Boston, MA')
plt.plot(df_aktxma_temperature.index, df_aktxma_temperature['Dallas-Fort Worth, TX'], label='Dallas-Fort Worth, TX')

plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Daily Average Temperature - Anchorage, Boston, and Dallas-Fort Worth')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(df_aktxma_precipitation.index, df_aktxma_precipitation['Anchorage, AK'], label='Anchorage, AK')
plt.plot(df_aktxma_precipitation.index, df_aktxma_precipitation['Boston, MA'], label='Boston, MA')
plt.plot(df_aktxma_precipitation.index, df_aktxma_precipitation['Dallas-Fort Worth, TX'], label='Dallas-Fort Worth, TX')
plt.xlabel('Date')
plt.ylabel('Precipitation (mm)')
plt.title('Daily Precipitation (mm) - Anchorage, Boston, and Dallas-Fort Worth')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
