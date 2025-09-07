import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datacleaning import clean_cab_data

# -------------------------------
# TIME FEATURES
# -------------------------------
def add_time_features(df):
    df = df.copy()
    df['datetime']=pd.to_datetime(df['datetime'],unit='s',errors='coerce')
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df['season'] = df['datetime'].dt.month % 12 // 3 + 1  # 1:Winter,2:Spring...
    df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
    return df


# -------------------------------
# LOCATION FEATURES
# -------------------------------
def add_location_features(df):
    df = df.copy()
    # Example encoding for source/destination
    for col in ['source', 'destination']:
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
    return df  

    


# -------------------------------
# CAB / PRODUCT FEATURES
# -------------------------------
# -------------------------------
# CAB / PRODUCT FEATURES
# -------------------------------
def add_cab_features(df):
    df = df.copy()
    df['cab_type_encoded'] = df['cab_type'].astype('category').cat.codes
    df['surge_flag'] = (df['surge_multiplier'] > 1).astype(int)
    def categorize_product(name):
        name = str(name).lower()
        if "pool" in name or "shared" in name:
            return "Shared"
        elif "black" in name or "lux" in name:
            return "Premium"
        else:
            return "Standard"
    df['product_group'] = df['name'].apply(categorize_product)
    PRODUCT_GROUP_MAP = {"Shared": 0, "Standard": 1, "Premium": 2}
    df['product_group_encoded'] = df['product_group'].map(PRODUCT_GROUP_MAP)
    return df


# -------------------------------
# PRICE FEATURES
# -------------------------------
def add_price_features(df):
    df = df.copy()
    df=df.dropna(subset=['price'])
    if {'price', 'distance'}.issubset(df.columns):
        df['price_per_km'] = df['price'] / df['distance'].replace(0, np.nan)
        df['log_price'] = np.log1p(df['price'])
    df.drop('price',axis=1,inplace=True)    
    return df


# -------------------------------
# WEATHER FEATURES
# -------------------------------
def add_weather_features(df):
    df = df.copy()
    if 'temperature' in df.columns and 'apparentTemperature' in df.columns:
        df['feels_like'] = df['apparentTemperature'] - df['temperature']
    if 'precipIntensity' in df.columns:
        df['precip_flag'] = (df['precipIntensity'] > 0).astype(int)
    if 'windSpeed' in df.columns:
        df['wind_stress'] = df['windSpeed']**2
    if 'visibility' in df.columns:
        df['visibility_flag'] = (df['visibility'] < 5).astype(int)
    if 'summary' in df.columns:
        df['is_rain'] = df['summary'].str.contains("rain", case=False).astype(int)
        df['is_clear'] = df['summary'].str.contains("clear", case=False).astype(int)
        df['is_cloudy'] = df['summary'].str.contains("cloud", case=False).astype(int)
    return df


# -------------------------------
# SUN / MOON FEATURES
# -------------------------------
def add_sun_moon_features(df):
    df = df.copy()
    if 'hour' not in df.columns:
        df['hour'] = df['datetime'].dt.hour
    df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
    # Placeholder for moon brightness (if you have moon phase data)
    df['moon_brightness'] = np.random.rand(len(df))  # dummy, replace with real
    return df

#--------------------------------
# EXTRACTING THE HOURS 
#--------------------------------
def extract_hour(df:pd.DataFrame)-> pd.DataFrame:
    df=df.copy()
    for col in df.columns:
        if 'Time' in col:
            df[col + "_hour"]=df[col].dt.hour
    return df


#--------------------------------
# DROPING UNNECESSORY COLUMNS
#--------------------------------

def remove_unneccosry_columns(df:pd.DataFrame)-> pd.DataFrame:
    columns=["id", "timezone","source" ,"destination" ,"cab_type" ,"product_id" ,"name" , "short_summary" ,"long_summary", "icon","product_group","windGustTime","temperatureHighTime","temperatureLowTime","apparentTemperatureHighTime","apparentTemperatureLowTime","sunriseTime","sunsetTime","uvIndexTime","temperatureMinTime","temperatureMaxTime","apparentTemperatureMinTime","apparentTemperatureMaxTime","datetime"]
    df.drop(columns, axis=1, inplace=True)
    return df

# -------------------------------
# MASTER PIPELINE 
# -------------------------------
def engineer_features(raw_data:pd.DataFrame)-> pd.DataFrame:
    """Main function to generate all features step by step."""
    cleaned_data=clean_cab_data(raw_data)
    df1 = add_time_features(cleaned_data)
    df2 = add_location_features(df1)
    df3 = add_cab_features(df2)
    df4 = add_price_features(df3)
    df5 = add_weather_features(df4)
    df6 = add_sun_moon_features(df5)
    df7=remove_unneccosry_columns(df6)
    return df7


