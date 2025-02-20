import math
from pathlib import Path
import zipfile
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup

def extract_files(zip_file_name): #read and decode yearly HTML files
    zip_file_path = Path(zip_file_name)
    yearly_html_content = {}

    if not zip_file_path.exists():
        raise FileNotFoundError(f"Error: The file '{zip_file_path}' isn't present.")

    with zipfile.ZipFile(zip_file_path, 'r') as zip_archive:

        for html_file_name in filter(lambda f: f.endswith(".html"), zip_archive.namelist()):
            try:
                year_extracted = html_file_name.rsplit('_', 1)[-1].split('.')[0]
                year_extracted = int(year_extracted)
                if 2013 <= year_extracted <= 2024:
                    with zip_archive.open(html_file_name) as html_file:
                        yearly_html_content[year_extracted] = html_file.read().decode('utf-8')

            except (ValueError, IndexError):
                pass

    return yearly_html_content
data = extract_files("gothenburg_sold_apartments.zip")



def extract_announcements(page): #parse html real estate property data

    months_in_swedish = {
        "januari": "01", "februari": "02", "mars": "03", "april": "04", "maj": "05",
        "juni": "06", "juli": "07", "augusti": "08", "september": "09",
        "oktober": "10", "november": "11", "december": "12"
    }

    def convert_swedish_date(date_text):
        """Converts Swedish date (e.g., '18 juni 2023') to 'YYYY-MM-DD'."""
        try:
            day, month, year = date_text.lower().split()
            return f"{year}-{months_in_swedish.get(month, '00')}-{day.zfill(2)}"
        except ValueError:
            return None

    def extract_number(text):
        """Extract numeric values, handling fractions like '½'."""
        match = re.search(r'-?\d+[½,\.\d]*', text)
        return float(match.group().replace('½', '.5').replace(',', '.')) if match else None

    soup = BeautifulSoup(page, 'html.parser')
    property_cards = soup.find_all('div', class_='property-card')

    apartment = []

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

        details_map = {
            'datum:': ('date', lambda x: pd.to_datetime(convert_swedish_date(x.strip()))),
            'område:': ('location_info', lambda x: [loc.strip() for loc in x.strip().split('·')]),
            'pris:': ('price', lambda x: int(re.sub(r'\D', '', x)) if re.sub(r'\D', '', x).isdigit() else None),
            'storlek:': ('area', extract_number),
            'rum:': ('rooms', extract_number),
            'våning:': ('floor', lambda x: 0.0 if 'BV' in x else extract_number(x))
        }

        for paragraph in sales_info.find_all('p'):
            text_content = paragraph.text.strip()
            for key, (field, extractor) in details_map.items():
                if text_content.startswith(key):
                    details_data[field] = extractor(text_content.replace(key, '').strip())

        location_info = details_data.pop('location_info', [])
        if len(location_info) == 2:
            details_data['district'], details_data['municipality'] = None, location_info[1]
        elif len(location_info) == 3:
            details_data['district'], details_data['municipality'] = location_info[1], location_info[2]

        if details_data['rooms'] and (details_data['rooms'] <= 0 or details_data['rooms'] > 10):
            details_data['rooms'] = None

        if details_data['floor']:
            if details_data['floor'] > 30:
                details_data['floor'] = None

        for key in ["area", "rooms", "floor"]:
            details_data[key] = pd.to_numeric(details_data[key], errors="coerce")

        apartment.append({
            'address': address,
            'date': details_data['date'],
            'district': details_data['district'],
            'municipality': details_data['municipality'],
            'price': details_data['price'],
            'area': details_data['area'],
            'rooms': details_data['rooms'],
            'floor': details_data['floor']
        })

    return pd.DataFrame(apartment)


yr2023_data = extract_files("gothenburg_sold_apartments.zip").get(2023, "")

if yr2023_data:
    df_2023 = extract_announcements(yr2023_data)
else:
    df_2023 = pd.DataFrame(columns=["address", "date", "district", "municipality", "price", "area", "rooms", "floor"])

if not df_2023.empty:
    df_2023["municipality"] = df_2023["municipality"].astype(str).str.strip()

    for col in ["area", "rooms", "floor"]:
        df_2023[col] = pd.to_numeric(df_2023[col], errors="coerce")
    df_2023.loc[df_2023["floor"] > 100, "floor"] = pd.NA
    df_2023.loc[df_2023["floor"] < -5, "floor"] = pd.NA
    df_2023.loc[df_2023["floor"].isna(), "floor"] = None

    df_2023_gothenburg = df_2023[df_2023["municipality"] == "Göteborg"].copy()
else:
    df_2023_gothenburg = df_2023.copy()


def five_point_summary(s):
    s = pd.to_numeric(s, errors="coerce").dropna()

    return pd.Series({
        'min': s.min(),
        'Q1': s.quantile(0.25),
        'median': s.median(),
        'Q3': s.quantile(0.75),
        'max': s.max()
    })
price_2023_gothenburg_fps = five_point_summary(df_2023_gothenburg['price'])


if not df_2023_gothenburg.empty and "price" in df_2023_gothenburg.columns:
    SEK_prices = df_2023_gothenburg["price"]

    fig, ax = plt.subplots(figsize=(12, 9))
    nr_bins = int(round(math.log2(len(SEK_prices)) + 1))
    counts, bins, _ = ax.hist(SEK_prices, bins=nr_bins, edgecolor="brown")
    if len(bins) != len(counts) + 1:
        raise ValueError(f"Required {len(counts) + 1} bins, but obtained {len(bins)}.")

    tick_SEK = np.arange(0, 2.05e7 + 1, 2.5e6)
    ax.set_xticks(tick_SEK)
    ax.set_xticklabels([f"{t / 1e6:.1f}" for t in tick_SEK], fontsize=7)
    ax.set_title("Apartment Prices in GBG in the year 2023", fontsize=14)
    ax.set_xlabel("Price (MSEK)", fontsize=7)
    ax.set_ylabel("Frequency", fontsize=7)
    plt.show()