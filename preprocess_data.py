import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Load data
data = pd.read_csv('traffic_data.csv')

# Drop rows with missing target (traffic_volume)
data = data.dropna(subset=['traffic_volume'])

# Fill missing temp with median of the hour
data['temp'] = data.groupby(data['Time'].str.split(':').str[0])['temp'].transform(lambda x: x.fillna(x.median()))

# Convert date to datetime and extract features
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['hour'] = data['Time'].str.split(':').str[0].astype(int)
data['minute'] = data['Time'].str.split(':').str[1].astype(int)
data['second'] = data['Time'].str.split(':').str[2].astype(int)

# Drop original date/time columns
data = data.drop(['date', 'Time'], axis=1)

# Map holiday names to numerical values (as in your HTML form)
holiday_map = {
    'None': 7,
    'Columbus Day': 1,
    'Veterans Day': 10,
    'Thanksgiving Day': 9,
    'Christmas Day': 0,
    'New Years Day': 6,
    'Washingtons Birthday': 11,
    'Memorial Day': 5,
    'Independence Day': 2,
    'State Fair': 8,
    'Labor Day': 3,
    'Martin Luther King Jr Day': 4
}
data['holiday'] = data['holiday'].map(holiday_map).fillna(7)  # Default to 'None'

# Map weather to numerical values
weather_map = {
    'Clear': 8,
    'Clouds': 1,
    'Rain': 6,
    'Drizzle': 2,
    'Mist': 5,
    'Haze': 4,
    'Fog': 3,
    'Thunderstorm': 10,
    'Snow': 8,  # Same as Clear (you had duplicate 8 in your HTML)
    'Squall': 9,
    'Smoke': 7
}
data['weather'] = data['weather'].map(weather_map).fillna(1)  # Default to 'Clouds'

# Save processed data
data.to_csv('processed_traffic_data.csv', index=False)

print("✅ Data preprocessing complete. Saved as processed_traffic_data.csv")