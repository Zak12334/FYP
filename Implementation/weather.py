import requests
import pymysql
from datetime import datetime, timedelta
import time
import logging
import random
import math


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("weather_sync.log"), logging.StreamHandler()]
)
logger = logging.getLogger("weather_sync")

# Database connection parameters
DB_CONFIG = {
    "host": "162.19.76.199",
    "user": "root",
    "password": "YNMIc3sF6b!Jsg8OKdUf",
    "db": "elighthouse_building",
    "port": 3307
}

# Mapping of meter IDs to towns
METER_TO_TOWN = {
    '4663': 'Castletownbere',
    '4664': 'Newberry Cross',
    '4665': 'Skibbereen',
    '4832': 'Clonakility',
    '4833': 'Skibbereen',
    '4834': 'Skibbereen',
    '4835': 'Inniscarra',
    '4836': 'Mitchelstown',
    '4837': 'Kinsale',
    '4839': 'Bantry',
    '4840': 'Charleville',
    '4841': 'Clonakilty',
    '4842': 'Castletownbere',
    '4859': 'Macroom'
}

# Town coordinates (latitude, longitude)
TOWN_COORDINATES = {
    'Castletownbere': (51.6538, -9.9112),
    'Newberry Cross': (52.0789, -8.2549),
    'Skibbereen': (51.5481, -9.2695),
    'Clonakility': (51.6208, -8.8709),
    'Inniscarra': (51.9073, -8.6461),
    'Mitchelstown': (52.2647, -8.2753),
    'Kinsale': (51.7062, -8.5224),
    'Bantry': (51.6749, -9.4535),
    'Charleville': (52.3540, -8.6797),
    'Clonakilty': (51.6208, -8.8709),
    'Macroom': (51.9071, -8.9590),
    'Cork': (51.8985, -8.4756)  # Default if town not found
}

def get_db_connection():
    """Create and return a database connection"""
    try:
        conn = pymysql.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            db=DB_CONFIG["db"],
            port=DB_CONFIG["port"]
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise

def fetch_temperature(town):
    """Fetch current temperature for a town from yr.no API"""
    if town not in TOWN_COORDINATES:
        logger.warning(f"No coordinates found for {town}, using Cork as default")
        lat, lon = TOWN_COORDINATES['Cork']
    else:
        lat, lon = TOWN_COORDINATES[town]
    
    url = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
    headers = {
        'User-Agent': 'SmartMeterAnomalyDetection/1.0 (sekeriye.osman@mymtu.ie)' 
    }
    params = {'lat': lat, 'lon': lon}
    
    try:
        logger.info(f"Fetching temperature for {town} ({lat}, {lon})")
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            # Extract current temperature from the first timepoint
            temperature = data['properties']['timeseries'][0]['data']['instant']['details']['air_temperature']
            logger.info(f"Temperature for {town}: {temperature}°C")
            return temperature
        else:
            logger.error(f"API request failed with status {response.status_code}: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error fetching temperature for {town}: {e}")
        return None

def update_weather_data(town, temperature):
    """Update the weather_data table with the latest temperature"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if a record for this town exists today
        today = datetime.now().strftime('%Y-%m-%d')
        check_query = """
        SELECT id FROM weather_data 
        WHERE town = %s AND DATE(made_at) = %s
        """
        cursor.execute(check_query, (town, today))
        existing_record = cursor.fetchone()
        
        # Use a predefined user ID instead of a string
        created_by_id = 1 
        
        if existing_record:
            # Update existing record
            update_query = """
            UPDATE weather_data 
            SET temperature = %s, made_at = %s, version = version + 1
            WHERE id = %s
            """
            cursor.execute(update_query, (temperature, datetime.now(), existing_record[0]))
            logger.info(f"Updated existing temperature record for {town}")
        else:
            # Insert new record
            insert_query = """
            INSERT INTO weather_data (town, temperature, made_at, active, created_by, version)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (town, temperature, datetime.now(), 1, created_by_id, 1))
            logger.info(f"Created new temperature record for {town}")
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error updating weather data for {town}: {e}")
        try:
            if 'conn' in locals() and conn:
                conn.close()
        except:
            pass
        return False

def sync_all_weather():
    """Sync weather data for Cork only"""
    logger.info("Starting Cork weather data synchronization")
    
    # Use Cork coordinates directly
    town = 'Cork'
    lat, lon = TOWN_COORDINATES['Cork']
    
    url = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
    headers = {
        'User-Agent': 'SmartMeterAnomalyDetection/1.0 (sekeriye.osman@mymtu.ie)'
    }
    params = {'lat': lat, 'lon': lon}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            temperature = data['properties']['timeseries'][0]['data']['instant']['details']['air_temperature']
            logger.info(f"Temperature for Cork: {temperature}°C")
            
            # Update database with Cork temperature
            update_weather_data('Cork', temperature)
            return {"Cork": True}
        else:
            logger.error(f"API request failed: {response.status_code}")
            return {"Cork": False}
    except Exception as e:
        logger.error(f"Error fetching temperature: {e}")
        return {"Cork": False}

def backfill_weather_data(start_date='2025-01-01'):
    """Backfill weather data from start_date to yesterday for Cork only"""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    yesterday = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    
    town = 'Cork'  # Only use Cork
    logger.info(f"Backfilling temperature for {town} from {start_date} to yesterday")
    
    current_date = start
    while current_date <= yesterday:
        # Generate synthetic temperature based on season
        day_of_year = current_date.timetuple().tm_yday
        base_temp = 10  # Base temperature
        seasonal_variation = 5 * math.sin((day_of_year - 15) / 365 * 2 * math.pi)  # +/-5°C seasonal variation
        daily_variation = random.uniform(-1.5, 1.5)  # Random daily variation
        
        temperature = base_temp + seasonal_variation + daily_variation
        
        # Check if record exists before inserting
        conn = get_db_connection()
        cursor = conn.cursor()
        
        check_query = """
        SELECT id FROM weather_data 
        WHERE town = %s AND DATE(made_at) = %s
        """
        cursor.execute(check_query, (town, current_date.strftime('%Y-%m-%d')))
        existing_record = cursor.fetchone()
        
        if not existing_record:
            # Insert into database
            insert_query = """
            INSERT INTO weather_data (town, temperature, made_at, active, created_by, version)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (town, temperature, current_date, 1, 1, 1))
            logger.info(f"Added historical data for {town} on {current_date.strftime('%Y-%m-%d')}")
        
        conn.commit()
        conn.close()
        
        current_date += timedelta(days=1)
    
    logger.info(f"Backfill completed for {town}")

if __name__ == "__main__":
    sync_all_weather()
    backfill_weather_data('2025-01-01')