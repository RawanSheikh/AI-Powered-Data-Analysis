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
 # Codegrade Tag Question2
# Do *not* remove the tag above
import pandas as pd
import re
from bs4 import BeautifulSoup

def extract_announcements(page):
    """Extract apartment sales data from an HTML page and return it as a Pandas DataFrame.

    Parameters:
    page (str): The HTML content as a string.

    Returns:
    pd.DataFrame: A dataframe containing extracted apartment sales data.
    """
    months_in_swedish = {
        "januari": "01", "februari": "02", "mars": "03", "april": "04", "maj": "05",
        "juni": "06", "juli": "07", "augusti": "08", "september": "09",
        "oktober": "10", "november": "11", "december": "12"
    }

    def convert_swedish_date(sv_date):
        """Convert Swedish date (e.g., '18 juni 2023') to 'YYYY-MM-DD'."""
        try:
            day, month, year = sv_date.lower().split()
            return f"{year}-{months_in_swedish.get(month, '00')}-{day.zfill(2)}"
        except ValueError:
            return None

    def extract_number(text):
        """Extract numeric values, handling fractions like '½'."""
        match = re.search(r'\d+[½,\.\d]*', text)
        return float(match.group().replace('½', '.5').replace(',', '.')) if match else None

    soup = BeautifulSoup(page, 'html.parser')
    property_cards = soup.find_all('div', class_='property-card')

    apartment_data = []  # Corrected variable name here

    for property_card in property_cards:
        card_soup = BeautifulSoup(str(property_card), 'html.parser')
        address = card_soup.find('h3', class_='property-title')
        address = address.text.strip() if address else None

        sales_info = card_soup.find(class_='property-details')
        if not sales_info:
            continue

        details_data = {
            'date': None, 'district': None, 'municipality': None,
            'price': None, 'area': None, 'rooms': None, 'floor': None
        }

        #  fields to extract
        details_map = {
            'datum:': ('date', lambda x: pd.to_datetime(convert_swedish_date(x.strip()))),
            'område:': ('location_info', lambda x: [loc.strip() for loc in x.strip().split('·')]),
            'pris:': ('price', lambda x: int(re.sub(r'\D', '', x)) if re.sub(r'\D', '', x).isdigit() else None),
            'storlek:': ('area', extract_number),
            'rum:': ('rooms', extract_number),
            'våning:': ('floor', lambda x: 0.0 if 'BV' in x else extract_number(x))
        }

        # Extract fields from sales info
        for paragraph in sales_info.find_all('p'):
            text_content = paragraph.text.strip()
            for key, (field, extractor) in details_map.items():
                if text_content.startswith(key):
                    details_data[field] = extractor(text_content.replace(key, '').strip())

        location_prop = details_data.pop('location_info', [])  # Fixed pop key here
        if len(location_prop) == 2:
            details_data['district'], details_data['municipality'] = "", location_prop[1]
        elif len(location_prop) == 3:
            details_data['district'], details_data['municipality'] = location_prop[1], location_prop[2]

        apartment_data.append({  # Corrected variable name here
            'address': address,
            'date': details_data['date'],
            'district': details_data['district'],
            'municipality': details_data['municipality'],
            'price': details_data['price'],
            'area': details_data['area'],
            'rooms': details_data['rooms'],
            'floor': details_data['floor']
        })

    df = pd.DataFrame(apartment_data, columns=['address', 'date', 'district', 'municipality', 'price', 'area', 'rooms', 'floor'])
    return df
