# src/inputpipeline.py
import numpy as np
import pandas as pd
from datetime import datetime

# ---- IMPORTANT: adjust these if your training label encodings were different ----
CAB_TYPE_MAP = {
    "Uber": 0,
    "Lyft": 1,
}


# The exact feature set (order matters) used at train time
FEATURE_COLUMNS = [
    "hour","day","month","distance","surge_multiplier","latitude","longitude",
    "temperature","apparentTemperature","precipIntensity","precipProbability",
    "humidity","windSpeed","windGust","visibility","temperatureHigh","temperatureLow",
    "apparentTemperatureHigh","apparentTemperatureLow","dewPoint","pressure",
    "windBearing","cloudCover","uvIndex","visibility.1","ozone","moonPhase",
    "precipIntensityMax","temperatureMin","temperatureMax","apparentTemperatureMin",
    "apparentTemperatureMax","day_of_week","is_weekend","rush_hour","season",
    "is_daytime","source_encoded","destination_encoded","cab_type_encoded",
    "surge_flag","product_group_encoded","price_per_km","feels_like","precip_flag",
    "wind_stress","visibility_flag","moon_brightness"
]

def _now_time_features():
    now = datetime.now()
    hour = now.hour
    day = now.day
    month = now.month
    day_of_week = now.weekday()
    is_weekend = 1 if day_of_week in (5, 6) else 0
    rush_hour = 1 if hour in (7, 8, 9, 17, 18, 19) else 0
    season = (month % 12) // 3 + 1  # 1..4
    is_daytime = 1 if 6 <= hour <= 18 else 0
    return hour, day, month, day_of_week, is_weekend, rush_hour, season, is_daytime

def _rand(a, b):  # float uniform
    return float(np.random.uniform(a, b))

def _randi(a, b):  # int inclusive
    return int(np.random.randint(a, b + 1))

def create_features(
    distance: float,
    cab_type_label: str,
    product_group_label: str,
    surge_flag: int = 0,
    dynamic: bool = True,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame containing ALL 47 features expected by the model.

    - distance: user input (float, km)
    - cab_type_label: user selection string (mapped via CAB_TYPE_MAP)
    - product_group_label: user selection string (mapped via PRODUCT_GROUP_MAP)
    - surge_flag: 0/1 toggle
    - dynamic: if True, generate environment features randomly (to mimic live conditions);
               if False, use stable mid values.

    Returns: pd.DataFrame with columns exactly as in FEATURE_COLUMNS
    """

    # time-derived features from current time
    hour, day, month, day_of_week, is_weekend, rush_hour, season, is_daytime = _now_time_features()

    # encodings (fallback to 0 if not found)
    cab_type_encoded = CAB_TYPE_MAP.get(cab_type_label, 0)
    product_group_encoded = PRODUCT_GROUP_MAP.get(product_group_label, 0)

    # surge multiplier policy
    if surge_flag:
        surge_multiplier = np.random.choice([1.25, 1.5, 1.75, 2.0]) if dynamic else 1.5
    else:
        surge_multiplier = 1.0

    # choose generation mode: dynamic (random) vs static (stable mid values)
    if dynamic:
        temperature = _rand(10, 35)
        apparentTemperature = temperature + _rand(-2, 2)
        humidity = _rand(0.3, 0.9)
        windSpeed = _rand(0, 20)
        windGust = windSpeed + _rand(0, 10)
        visibility = _rand(5, 15)
        dewPoint = _rand(0, 25)
        pressure = _rand(990, 1025)
        cloudCover = _rand(0, 1)
        uvIndex = _randi(0, 10)
        ozone = _rand(250, 350)
        moonPhase = _rand(0, 1)
        precipIntensity = _rand(0, 0.4)
        precipProbability = _rand(0, 1)
        precipIntensityMax = _rand(0, 0.6)
        temperatureHigh = temperature + _rand(1, 4)
        temperatureLow = temperature - _rand(1, 4)
        apparentTemperatureHigh = apparentTemperature + _rand(0.5, 2)
        apparentTemperatureLow = apparentTemperature - _rand(0.5, 2)
        temperatureMin = temperature - _rand(1, 5)
        temperatureMax = temperature + _rand(1, 5)
        apparentTemperatureMin = apparentTemperature - _rand(1, 3)
        apparentTemperatureMax = apparentTemperature + _rand(1, 3)
        latitude = _rand(-90, 90)    # placeholder if you don't have geo
        longitude = _rand(-180, 180) # placeholder
        windBearing = _randi(0, 360)
        price_per_km = _rand(8, 20)
        feels_like = apparentTemperature
        precip_flag = 1 if precipIntensity > 0.01 or precipProbability > 0.4 else 0
        wind_stress = windSpeed ** 2 / 400.0
        visibility_flag = 1 if visibility >= 5 else 0
        moon_brightness = 1 - abs(0.5 - moonPhase) * 2  # 0..1, bright near 0.5
        source_encoded = _randi(0, 10)
        destination_encoded = _randi(0, 10)
    else:
        # Stable mid-values for a calm, reproducible prediction
        temperature = 22.0
        apparentTemperature = 22.5
        humidity = 0.6
        windSpeed = 5.0
        windGust = 8.0
        visibility = 10.0
        dewPoint = 15.0
        pressure = 1012.0
        cloudCover = 0.3
        uvIndex = 5
        ozone = 300.0
        moonPhase = 0.5
        precipIntensity = 0.0
        precipProbability = 0.1
        precipIntensityMax = 0.0
        temperatureHigh = 25.0
        temperatureLow = 20.0
        apparentTemperatureHigh = 26.0
        apparentTemperatureLow = 21.0
        temperatureMin = 20.0
        temperatureMax = 25.0
        apparentTemperatureMin = 21.0
        apparentTemperatureMax = 26.0
        latitude = 0.0
        longitude = 0.0
        windBearing = 180
        price_per_km = 12.0
        feels_like = apparentTemperature
        precip_flag = 0
        wind_stress = windSpeed ** 2 / 400.0
        visibility_flag = 1
        moon_brightness = 1.0
        source_encoded = 0
        destination_encoded = 0

    # assemble full row
    row = {
        "hour": hour,
        "day": day,
        "month": month,
        "distance": float(distance),
        "surge_multiplier": float(surge_multiplier),
        "latitude": float(latitude),
        "longitude": float(longitude),
        "temperature": float(temperature),
        "apparentTemperature": float(apparentTemperature),
        "precipIntensity": float(precipIntensity),
        "precipProbability": float(precipProbability),
        "humidity": float(humidity),
        "windSpeed": float(windSpeed),
        "windGust": float(windGust),
        "visibility": float(visibility),
        "temperatureHigh": float(temperatureHigh),
        "temperatureLow": float(temperatureLow),
        "apparentTemperatureHigh": float(apparentTemperatureHigh),
        "apparentTemperatureLow": float(apparentTemperatureLow),
        "dewPoint": float(dewPoint),
        "pressure": float(pressure),
        "windBearing": int(windBearing),
        "cloudCover": float(cloudCover),
        "uvIndex": int(uvIndex),
        "visibility.1": float(visibility),
        "ozone": float(ozone),
        "moonPhase": float(moonPhase),
        "precipIntensityMax": float(precipIntensityMax),
        "temperatureMin": float(temperatureMin),
        "temperatureMax": float(temperatureMax),
        "apparentTemperatureMin": float(apparentTemperatureMin),
        "apparentTemperatureMax": float(apparentTemperatureMax),
        "day_of_week": int(day_of_week),
        "is_weekend": int(is_weekend),
        "rush_hour": int(rush_hour),
        "season": int(season),
        "is_daytime": int(is_daytime),
        "source_encoded": int(source_encoded),
        "destination_encoded": int(destination_encoded),
        "cab_type_encoded": int(cab_type_encoded),
        "surge_flag": int(surge_flag),
        "product_group_encoded": int(product_group_encoded),
        "price_per_km": float(price_per_km),
        "feels_like": float(feels_like),
        "precip_flag": int(precip_flag),
        "wind_stress": float(wind_stress),
        "visibility_flag": int(visibility_flag),
        "moon_brightness": float(moon_brightness),
    }

    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)  # enforce order
    return df
