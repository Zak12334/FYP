import pandas as pd
import pymysql
from datetime import datetime  # Added import

class SmartMeterDataLoader:
    def __init__(self, host='162.19.76.199', user='root', password='YNMIc3sF6b!Jsg8OKdUf', db='elighthouse_building', port=3307):
        self.host = host
        self.user = user
        self.password = password
        self.db = db
        self.port = port

    def get_db_connection(self):
        """
        Create and return a database connection
        
        Returns:
            pymysql.Connection: Database connection object
        """
        conn = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            db=self.db,
            port=self.port
        )
        return conn

    def get_data(self):
        print("Fetching daily aggregated data...")
        query = """
        SELECT
            DATE(made_at) as date,
            meter_id,
            SUM(consumption) as daily_consumption,
            COUNT(*) as reading_count
        FROM smart_meter_raw_data
        GROUP BY DATE(made_at), meter_id
        ORDER BY DATE(made_at), meter_id
        """
        conn = self.get_db_connection()
        df = pd.read_sql(query, conn)
        conn.close()
        df['meter_id'] = df['meter_id'].astype(str)
        df['date'] = pd.to_datetime(df['date'])
        print(f"Query executed, got {len(df)} rows")
        print("Unique meter IDs:", df['meter_id'].unique())
        print("Date range:", df['date'].min(), "to", df['date'].max())
        return df

    def log_anomaly_to_database(self, anomaly_row):
        # Define the cutoff date as a Timestamp object.
        cutoff_date = pd.Timestamp('2025-04-01')
        # If the anomaly date is before the cutoff, do not log it.
        if anomaly_row['date'] < cutoff_date:
            print(f"Skipping logging anomaly for meter {anomaly_row['meter_id']} on {anomaly_row['date']}, "
                f"as it is before {cutoff_date.date()}.")
            return False

        try:
            # Get connection
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Convert and extract required fields
            meter_id = str(anomaly_row['meter_id'])
                    
            # Handle the consumption value correctly - remove 'kWh' if present and convert to float
            consumption_str = str(anomaly_row['daily_consumption'])
            if 'kWh' in consumption_str:
                consumption_str = consumption_str.replace('kWh', '').strip()
            actual_consumption = float(consumption_str)
                    
            anomaly_score = float(anomaly_row.get('anomaly_score', 0.0))
            anomaly_type = str(anomaly_row.get('anomaly_type', 'Unknown'))[:255]
                    
            ##Handle expected_consumption exactly the same way as actual_consumption
            expected_consumption = None
            if pd.notna(anomaly_row.get('expected_consumption')):
                exp_consumption_str = str(anomaly_row['expected_consumption'])
                if 'kWh' in exp_consumption_str:
                    exp_consumption_str = exp_consumption_str.replace('kWh', '').strip()
                expected_consumption = float(exp_consumption_str)
            
            # Similar handling for expected consumption
            expected_consumption = None
            if pd.notna(anomaly_row.get('expected_consumption')):
                # Direct conversion to float with no string manipulation
                expected_consumption = float(anomaly_row['expected_consumption'])
                # Print for debugging
                print(f"Expected consumption to be inserted: {expected_consumption}")
            
            # Format the date range for the anomaly (full day)
            data_period_start = anomaly_row['date'].strftime('%Y-%m-%d 00:00:00')
            data_period_end = anomaly_row['date'].strftime('%Y-%m-%d 23:59:59')
            
            # Check for an existing record for this meter and day.
            check_query = """
            SELECT id FROM smaart_meter_alarm 
            WHERE meter_id = %s AND data_period_start = %s AND data_period_end = %s
            """
            cursor.execute(check_query, (meter_id, data_period_start, data_period_end))
            existing_record = cursor.fetchone()
            
            # If a record exists, do not insert it again.
            if existing_record:
                print(f"Anomaly for meter {meter_id} on {data_period_start} already exists in database.")
                conn.close()
                return False
            
            # Determine severity level (adjust logic as needed)
            level = "LOW"
            if expected_consumption and expected_consumption > 0:
                pct_diff = abs((actual_consumption - expected_consumption) / expected_consumption * 100)
                if pct_diff > 500 or anomaly_score > 0.8:
                    level = "CRITICAL"
                elif pct_diff > 200 or anomaly_score > 0.7:
                    level = "HIGH"
                elif pct_diff > 100 or anomaly_score > 0.6:
                    level = "MEDIUM"
                else:
                    level = "LOW"
            
            # Current timestamp for the record
            made_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            status = "NEW"
            
            # Insert the anomaly record into the database.
            insert_query = """
            INSERT INTO smaart_meter_alarm (
                meter_id, actual_consumption, anomaly_score, anomaly_type,
                data_period_start, data_period_end, expected_consumption,
                level, made_at, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(
                insert_query,
                (
                    meter_id, actual_consumption, anomaly_score, anomaly_type,
                    data_period_start, data_period_end, expected_consumption,
                    level, made_at, status
                )
            )
            
            # Commit and close the connection.
            conn.commit()
            conn.close()
            print(f"Successfully logged anomaly for meter {meter_id} on {data_period_start}")
            return True
            
        except Exception as e:
            print(f"Error logging anomaly to database: {str(e)}")
            import traceback
            traceback.print_exc()
            try:
                if 'conn' in locals() and conn:
                    conn.close()
            except:
                pass
            return False
