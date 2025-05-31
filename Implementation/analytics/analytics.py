import os
import pickle
import numpy as np
import pandas as pd
import holidays
import traceback
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose



# Global mapping for meter IDs to building names
METER_TO_BUILDING = {
    '4663': 'Castletownbere Area Engineers Office',
    '4664': 'Newberry Cross Machinery Depot',
    '4665': 'Skibbereen MD',
    '4832': 'Clonakility Library & Offices',
    '4833': 'Skibbereen Library',
    '4834': 'Skibbereen Heritage Centre',
    '4835': 'Environmental Office, Inniscarra',
    '4836': 'Mitcheltstown Area Engineers Office',
    '4837': 'Kinsale MD Office',
    '4839': 'Bantry Fire Station',
    '4840': 'Charleville Area Engineers Office',
    '4841': 'Clonakilty MD Office - Town Hall',
    '4842': 'Castletownbere Area Engineers Office',
    '4859': 'Macroom MD, AEO and Community Hall'
}

class AdvancedAnomalyDetection:
    def __init__(self, sequence_length=14, stride=1, batch_size=32, model_dir='saved_models'):
        self.lstm_model = None
        self.scaler = RobustScaler()
        self.threshold = None
        self.sequence_length = sequence_length
        self.stride = stride
        self.batch_size = batch_size
        self.model_dir = model_dir

    
    def create_lstm_autoencoder(self, input_shape):
        """Enhanced LSTM architecture with BatchNorm and Dropout"""
        inputs = tf.keras.layers.Input(shape=(input_shape[1], input_shape[2]))

        # Encoder
        x = tf.keras.layers.BatchNormalization()(inputs)
        x = tf.keras.layers.LSTM(64, activation='relu', return_sequences=True)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LSTM(32, activation='relu', return_sequences=True)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        encoded = tf.keras.layers.LSTM(16, activation='relu')(x)

        # Decoder
        x = tf.keras.layers.RepeatVector(input_shape[1])(encoded)
        x = tf.keras.layers.LSTM(16, activation='relu', return_sequences=True)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LSTM(32, activation='relu', return_sequences=True)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LSTM(64, activation='relu', return_sequences=True)(x)
        decoded = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_shape[2]))(x)

        model = tf.keras.Model(inputs, decoded)
        model.compile(optimizer='adam',
                     loss='huber',
                     metrics=['mae'])
        return model
    
    def sequence_generator(self, data):
        """Generate sequences for LSTM"""
        sequences = []
        for i in range(0, len(data) - self.sequence_length + 1, self.stride):
            sequence = data[i:i + self.sequence_length]
            sequences.append(sequence)
        return np.array(sequences)

    def fit(self, X, epochs=10):
        """Train the model on the data"""
        print("Scaling data...")
        X_scaled = self.scaler.fit_transform(X)

        # Reshape data into sequences
        print("Creating sequences...")
        X_sequences = self.sequence_generator(X_scaled)
        if len(X_sequences) == 0:
            raise ValueError("Not enough data points to create sequences")

        # Reshape sequences for LSTM input
        X_sequences = X_sequences.reshape((len(X_sequences), self.sequence_length, X.shape[1]))

        print("Creating model...")
        self.lstm_model = self.create_lstm_autoencoder((len(X_sequences), self.sequence_length, X.shape[1]))

        # Add early stopping with shorter patience
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True,
            mode='min'
        )

        print("Training model...")
        try:
            history = self.lstm_model.fit(
                X_sequences, X_sequences,
                epochs=epochs,
                batch_size=min(self.batch_size, len(X_sequences)),
                verbose=1,
                validation_split=0.2,
                callbacks=[early_stopping]
            )
        except Exception as e:
            print(f"Training interrupted: {str(e)}")
            if self.lstm_model is None:
                raise ValueError("Model training failed")

        # Calculate threshold
        print("Calculating threshold...")
        predictions = self.lstm_model.predict(X_sequences, verbose=0)
        reconstruction_errors = np.mean(np.square(X_sequences - predictions), axis=(1, 2))
        self.threshold = np.percentile(reconstruction_errors, 95)

        return history

    def predict(self, X):
        """Predict anomalies in the data"""
        if self.lstm_model is None:
            raise ValueError("Model must be trained before making predictions")

        print("Predicting anomalies...")
        X_scaled = self.scaler.transform(X)
        X_sequences = self.sequence_generator(X_scaled)

        if len(X_sequences) == 0:
            return np.array([False] * len(X)), np.array([0.0] * len(X))

        X_sequences = X_sequences.reshape((len(X_sequences), self.sequence_length, X.shape[1]))

        predictions = self.lstm_model.predict(X_sequences, verbose=0)
        reconstruction_errors = np.mean(np.square(X_sequences - predictions), axis=(1, 2))

        # Calculate anomaly scores
        anomaly_scores = reconstruction_errors / self.threshold
        anomalies = reconstruction_errors > self.threshold

        # Pad results to match input length
        padding_length = len(X) - len(anomalies)
        if padding_length > 0:
            anomalies = np.pad(anomalies, (0, padding_length), 'edge')
            anomaly_scores = np.pad(anomaly_scores, (0, padding_length), 'edge')

        return anomalies, anomaly_scores

    def save_model(self, meter_id):
        """Save the trained model and scaler to disk"""
        if self.lstm_model is None:
            print(f"No model to save for meter {meter_id}")
            return False

        model_path = os.path.join(self.model_dir, f'lstm_model_{meter_id}.h5')
        scaler_path = os.path.join(self.model_dir, f'scaler_{meter_id}.pkl')
        threshold_path = os.path.join(self.model_dir, f'threshold_{meter_id}.pkl')

        try:
            # Save LSTM model
            self.lstm_model.save(model_path)

            # Save scaler
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)

            # Save threshold
            with open(threshold_path, 'wb') as f:
                pickle.dump(self.threshold, f)

            print(f"Successfully saved model for meter {meter_id}")
            return True
        except Exception as e:
            print(f"Error saving model for meter {meter_id}: {str(e)}")
            return False

    def load_model(self, meter_id):
        """Load trained model and scaler from disk if available"""
        model_path = os.path.join(self.model_dir, f'lstm_model_{meter_id}.h5')
        scaler_path = os.path.join(self.model_dir, f'scaler_{meter_id}.pkl')
        threshold_path = os.path.join(self.model_dir, f'threshold_{meter_id}.pkl')

        if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(threshold_path)):
            print(f"No saved model found for meter {meter_id}")
            return False

        try:
            # Load LSTM model
            self.lstm_model = load_model(model_path)

            # Load scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

            # Load threshold
            with open(threshold_path, 'rb') as f:
                self.threshold = pickle.load(f)

            print(f"Successfully loaded model for meter {meter_id}")
            return True
        except Exception as e:
            print(f"Error loading model for meter {meter_id}: {str(e)}")
            return False

class EnhancedSmartMeterAnalytics:
    def __init__(self, model_dir='saved_models', data_loader=None):
        self.scaler = StandardScaler()
        self.lstm_detector = AdvancedAnomalyDetection(model_dir=model_dir)
        self.isolation_forest = IsolationForest(contamination=0.05, random_state=42)
        self.irish_holidays = holidays.Ireland()
        self.seasonal_patterns = {}
        self.thresholds = {}
        self.model_dir = model_dir
        self.data_loader = data_loader

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
    def extract_features(self, df):
        print("Extracting enhanced features for improved anomaly detection...")

        # Process datetime features with more detail
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week

        # cyclical encoding of time features
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)

        # Enhanced season feature
        df['season'] = pd.cut(df['month'],
                            bins=[0, 3, 6, 9, 12],
                            labels=['Winter', 'Spring', 'Summer', 'Fall'],
                            include_lowest=True)

        # One-hot encode season for better model usage
        season_dummies = pd.get_dummies(df['season'], prefix='season')
        df = pd.concat([df, season_dummies], axis=1)

        # Add tank-specific features refill detection
        meter_groups = df.groupby('meter_id')

        # Calculate rolling statistics per meter
        df['rolling_mean_7d'] = meter_groups['daily_consumption'].transform(
            lambda x: x.rolling(window=7, min_periods=2).mean()
        )
        df['rolling_mean_14d'] = meter_groups['daily_consumption'].transform(
            lambda x: x.rolling(window=14, min_periods=3).mean()
        )
        df['rolling_mean_30d'] = meter_groups['daily_consumption'].transform(
            lambda x: x.rolling(window=30, min_periods=5).mean()
        )

        # Use median
        df['rolling_median_7d'] = meter_groups['daily_consumption'].transform(
            lambda x: x.rolling(window=7, min_periods=2).median()
        )

        # Standard deviation at different windows
        df['rolling_std_7d'] = meter_groups['daily_consumption'].transform(
            lambda x: x.rolling(window=7, min_periods=2).std()
        )
        df['rolling_std_14d'] = meter_groups['daily_consumption'].transform(
            lambda x: x.rolling(window=14, min_periods=3).std()
        )

        # Add median absolute deviation
        df['rolling_mad_7d'] = meter_groups['daily_consumption'].transform(
            lambda x: (x - x.rolling(window=7, min_periods=2).median()).abs().rolling(window=7, min_periods=2).median()
        )

        # consumption change metrics
        df['consumption_diff'] = meter_groups['daily_consumption'].diff()
        df['consumption_pct_change'] = meter_groups['daily_consumption'].pct_change() * 100

        # Detect refills
        df['is_refill'] = False
        for meter_id, group in meter_groups:
            # Skip if not enough data
            if len(group) < 7:
                continue

            # Get indices for this meter
            indices = group.index

            for i in range(1, len(indices)):
                idx = indices[i]
                prev_idx = indices[i-1]

                current = df.loc[idx, 'daily_consumption']
                previous = df.loc[prev_idx, 'daily_consumption']

                # Use rolling median and MAD for outlier detection
                rolling_median = df.loc[idx, 'rolling_median_7d'] if pd.notna(df.loc[idx, 'rolling_median_7d']) else previous
                rolling_mad = df.loc[idx, 'rolling_mad_7d'] if pd.notna(df.loc[idx, 'rolling_mad_7d']) else (current * 0.1)  # Default to 10% if unknown

                # Skip if missing values or zero values
                if pd.isna(current) or pd.isna(previous) or previous == 0 or rolling_mad == 0:
                    continue

                # Enhanced refill detection logic:
                if (current > previous * 2.0 and
                    current - previous > 20 and
                    current > rolling_median + 4 * rolling_mad):
                    df.loc[idx, 'is_refill'] = True

        # Track days since last refill
        df['days_since_refill'] = df.groupby('meter_id').cumcount()
        # Reset counter on refill days
        df.loc[df['is_refill'], 'days_since_refill'] = 0
        # Forward fill the refill days for each meter
        df['days_since_refill'] = df.groupby('meter_id')['days_since_refill'].transform(
            lambda x: x.replace(to_replace=0, method='ffill').cumsum()
        )

        # Track when next refill occurs
        df['days_to_next_refill'] = np.nan
        for meter_id, group in meter_groups:
            refill_dates = df[df['meter_id'] == meter_id][df['is_refill']]['date']
            if len(refill_dates) < 2:
                continue

            for i, refill_date in enumerate(refill_dates[:-1]):
                next_refill = refill_dates.iloc[i+1]
                mask = (df['meter_id'] == meter_id) & (df['date'] >= refill_date) & (df['date'] < next_refill)
                df.loc[mask, 'days_to_next_refill'] = [(next_refill - date).days for date in df.loc[mask, 'date']]

        # Calculate relative position in refill cycle (0-100%)
        df['cycle_position'] = np.nan
        for meter_id, group in meter_groups:
            mask = (df['meter_id'] == meter_id) & df['days_since_refill'].notna() & df['days_to_next_refill'].notna()
            if mask.sum() > 0:
                total_days = df.loc[mask, 'days_since_refill'] + df.loc[mask, 'days_to_next_refill']
                if (total_days > 0).all():
                    df.loc[mask, 'cycle_position'] = df.loc[mask, 'days_since_refill'] / total_days * 100

        # Day type features - important for building patterns
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)

        # Holiday features
        df['is_holiday'] = df['date'].dt.date.map(lambda x: x in self.irish_holidays).astype(int)
        df['days_to_holiday'] = df['date'].dt.date.map(self._days_to_next_holiday)
        df['days_from_holiday'] = df['date'].dt.date.map(self._days_from_last_holiday)
        df['near_holiday'] = ((df['days_to_holiday'] <= 1) | (df['days_from_holiday'] <= 1)).astype(int)

        # Detect month start/end
        df['is_month_start'] = (df['day_of_month'] <= 3).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)

        # Calculate z-scores by day of week
        df['dow_zscore'] = 0.0
        for meter_id in df['meter_id'].unique():
            for day in range(7):
                meter_day_data = df[(df['meter_id'] == meter_id) & (df['day_of_week'] == day)]
                if len(meter_day_data) >= 4:  # Need enough data points
                    day_mean = meter_day_data['daily_consumption'].mean()
                    day_std = meter_day_data['daily_consumption'].std()
                    if day_std > 0:
                        day_mask = (df['meter_id'] == meter_id) & (df['day_of_week'] == day)
                        df.loc[day_mask, 'dow_zscore'] = (df.loc[day_mask, 'daily_consumption'] - day_mean) / day_std

        # Account for temperature
        if 'avg_temperature' in df.columns:
            # Temperature change
            df['temp_change'] = df.groupby('meter_id')['avg_temperature'].diff()

            # Calculate temperature sensitivity per building
            for meter_id in df['meter_id'].unique():
                meter_data = df[df['meter_id'] == meter_id]
                if len(meter_data) >= 30 and not meter_data['avg_temperature'].isna().all():
                    try:
                        # Linear regression of consumption vs temperature
                        valid_data = meter_data.dropna(subset=['daily_consumption', 'avg_temperature'])
                        if len(valid_data) >= 10:
                            X = valid_data['avg_temperature'].values.reshape(-1, 1)
                            y = valid_data['daily_consumption'].values

                            from sklearn.linear_model import LinearRegression
                            model = LinearRegression().fit(X, y)

                            # Store temperature coefficient
                            temp_sensitivity = model.coef_[0]

                            # Calculate expected consumption based on temperature
                            df.loc[df['meter_id'] == meter_id, 'temp_expected_consumption'] = (
                                model.intercept_ + model.coef_[0] * df.loc[df['meter_id'] == meter_id, 'avg_temperature']
                            )

                            # Calculate temperature-adjusted anomaly score
                            df.loc[df['meter_id'] == meter_id, 'temp_consumption_deviation'] = (
                                df.loc[df['meter_id'] == meter_id, 'daily_consumption'] -
                                df.loc[df['meter_id'] == meter_id, 'temp_expected_consumption']
                            )
                    except Exception as e:
                        print(f"Error in temperature regression for meter {meter_id}: {str(e)}")

        # Add seasonal decomposition with more flexibility
        for meter_id in df['meter_id'].unique():
            meter_data = df[df['meter_id'] == meter_id].copy().sort_values('date')
            if len(meter_data) >= 30:  # Need enough data for decomposition
                try:
                    # Get daily consumption time series
                    ts = meter_data.set_index('date')['daily_consumption']
                    if len(ts) >= 14 and ts.notna().all():
                        # Try different seasonal periods based on data length
                        if len(ts) >= 365:  # If we have a year of data
                            period = 365  # Annual seasonality
                        elif len(ts) >= 30:
                            period = 7  # Weekly seasonality
                        else:
                            period = None

                        if period:
                            decomposition = seasonal_decompose(ts, model='additive', period=period, extrapolate_trend='freq')
                            df.loc[meter_data.index, 'seasonal_component'] = decomposition.seasonal.values
                            df.loc[meter_data.index, 'trend_component'] = decomposition.trend.values
                            df.loc[meter_data.index, 'residual_component'] = decomposition.resid.values
                except Exception as e:
                    print(f"Seasonal decomposition failed for meter {meter_id}: {str(e)}")

        # Fill NaN values in new columns
        fill_values = {
            'rolling_mean_7d': df['daily_consumption'].mean(),
            'rolling_mean_14d': df['daily_consumption'].mean(),
            'rolling_mean_30d': df['daily_consumption'].mean(),
            'rolling_median_7d': df['daily_consumption'].median(),
            'rolling_std_7d': df['daily_consumption'].std(),
            'rolling_std_14d': df['daily_consumption'].std(),
            'rolling_mad_7d': df['daily_consumption'].std() * 0.5,
            'consumption_diff': 0,
            'consumption_pct_change': 0,
            'days_to_next_refill': df['days_since_refill'].median(),
            'cycle_position': 50,  # Middle of cycle
            'dow_zscore': 0,
            'seasonal_component': 0,
            'trend_component': df['daily_consumption'].mean(),
            'residual_component': 0,
            'temp_expected_consumption': df['daily_consumption'].mean(),
            'temp_consumption_deviation': 0
        }

        # Fill NaNs with sensible defaults
        for col, fill_val in fill_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_val)

        return df.dropna(subset=['daily_consumption'])  # no NaN in key column

    def _days_to_next_holiday(self, date):
        """Calculate days until next Irish holiday"""
        date = pd.Timestamp(date).date()
        next_holiday = min((h for h in self.irish_holidays.keys() if h > date),
                        default=date)
        return (next_holiday - date).days

    def _days_from_last_holiday(self, date):
        """Calculate days since last holiday"""
        date = pd.Timestamp(date).date()
        last_holiday = max((h for h in self.irish_holidays.keys() if h < date),
                        default=date)
        return (date - last_holiday).days

    def calculate_optimal_threshold(self, meter_data, initial_threshold=3.0, min_anomalies_percent=0.5, max_anomalies_percent=5.0):
        """Calculate an optimal z-score threshold for a specific building based on historical data patterns"""
        if len(meter_data) < 30:  # Need enough data for reliable threshold
            return initial_threshold

    
        thresholds = np.arange(2.0, 5.0, 0.2)
        best_threshold = initial_threshold
        best_score = float('inf')  # Lower is better

        # Calculate consumption volatility
        volatility = meter_data['daily_consumption'].std() / meter_data['daily_consumption'].mean()

        for threshold in thresholds:
            # Calculate z-scores for the consumption
            meter_data['test_zscore'] = np.abs((meter_data['daily_consumption'] -
            meter_data['daily_consumption'].mean()) /
            meter_data['daily_consumption'].std())

            # Calculate percent of anomalies with this threshold
            anomaly_percent = (meter_data['test_zscore'] > threshold).mean() * 100

            # stop thresholds that produce too many or too few anomalies
            if anomaly_percent < min_anomalies_percent or anomaly_percent > max_anomalies_percent:
                continue

            # Score is better if anomaly percent is reasonable and threshold accounts for volatility
            score = abs(anomaly_percent - 2.0) + abs(threshold - (3.0 + volatility * 1.5))

            if score < best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold

    def add_domain_knowledge_rules(self, df):
      print("Applying domain knowledge rules with building-specific parameters...")

      # Loop through each building type separately
      for meter_id, building_name in METER_TO_BUILDING.items(): 
          meter_data = df[df['meter_id'] == meter_id]
          if len(meter_data) < 14:  # Skip if not enough data for reliable patterns
              continue

          # Determine building type and apply appropriate rules
          building_type = self._classify_building_type(building_name)

          # Office buildings
          if building_type == 'office':
              # Analyze typical office hours pattern (weekday 9-5)
              weekday_avg = meter_data[meter_data['is_weekend'] == 0]['daily_consumption'].mean()
              weekday_std = meter_data[meter_data['is_weekend'] == 0]['daily_consumption'].std()

              # Weekend detection - more precise by understanding this specific building
              weekend_historic = meter_data[meter_data['is_weekend'] == 1]['daily_consumption']
              if len(weekend_historic) >= 4:
                weekend_avg = weekend_historic.mean()
                weekend_std = weekend_historic.std()
                weekend_threshold = max(
                    weekend_avg + 1.5 * weekend_std,  # Statistical approach
                    weekday_avg * 0.7  # Percentage of weekday usage
                )

                weekend_indices = meter_data[
                      (meter_data['is_weekend'] == 1) &
                      (meter_data['daily_consumption'] > weekend_threshold) &
                      (~meter_data['is_refill'])
                  ].index

                for idx in weekend_indices:
                      df.at[idx, 'is_anomaly'] = True
                      try:
                          self.data_loader.log_anomaly_to_database(df.loc[idx])
                      except Exception as e:
                           print(f"Error logging anomaly to database: {str(e)}")
                      current_types = df.at[idx, 'anomaly_type']
                      new_type = 'Unusual weekend office usage'
                      df.at[idx, 'anomaly_type'] = new_type if not current_types else f"{current_types} & {new_type}"
                      df.at[idx, 'anomaly_score'] = max(df.at[idx, 'anomaly_score'], 0.6)
                      df.at[idx, 'detection_confidence'] = max(df.at[idx, 'detection_confidence'], 0.7)

          # Libraries with specific open/closed patterns
          elif building_type == 'library':
              if len(meter_data) >= 28:  # Need enough data for reliable patterns
                  # Libraries have consistent open hours, check for variations
                  for day in range(7):  # Each day of week
                      day_data = meter_data[meter_data['day_of_week'] == day]
                      if len(day_data) >= 4:
                          day_avg = day_data['daily_consumption'].mean()
                          day_std = day_data['daily_consumption'].std()

                          # Flag extreme variations from typical pattern
                          outlier_indices = meter_data[
                              (meter_data['day_of_week'] == day) &
                              (abs(meter_data['daily_consumption'] - day_avg) > 3 * day_std) &
                              (~meter_data['is_refill'])
                          ].index

                          for idx in outlier_indices:
                              df.at[idx, 'is_anomaly'] = True
                              try:
                                   self.data_loader.log_anomaly_to_database(df.loc[idx])

                              except Exception as e:
                                   print(f"Error logging anomaly to database: {str(e)}")
                              current_types = df.at[idx, 'anomaly_type']
                              day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                              new_type = f'Library pattern violation ({day_names[day]})'
                              df.at[idx, 'anomaly_type'] = new_type if not current_types else f"{current_types} & {new_type}"
                              df.at[idx, 'anomaly_score'] = max(df.at[idx, 'anomaly_score'], 0.65)
                              df.at[idx, 'detection_confidence'] = max(df.at[idx, 'detection_confidence'], 0.7)

          # Specific rule for Skibbereen Library with known schedule
          elif building_type == "skibbereen_library":
              # Define known closed days (Mondays and bank holidays)
              closed_days_mask = (meter_data['day_of_week'] == 0)  # Mondays

              # Calculate a reasonable threshold for "closed" consumption
              typical_open_consumption = meter_data[~closed_days_mask]['daily_consumption'].median()
              closed_threshold = typical_open_consumption * 0.4  # Expect less than 40% of normal

              # Find consumption on days that should be closed
              anomaly_indices = meter_data[
                  closed_days_mask &
                  (meter_data['daily_consumption'] > closed_threshold) &
                  (~meter_data['is_refill'])
              ].index

              # Flag unexpected consumption on closed days
              for idx in anomaly_indices:
                  df.at[idx, 'is_anomaly'] = True
                  try:
                    self.data_loader.log_anomaly_to_database(df.loc[idx])
                  except Exception as e:
                    print(f"Error logging anomaly to database: {str(e)}")
                  current_types = df.at[idx, 'anomaly_type']
                  new_type = 'Unexpected consumption on Monday (building should be closed)'
                  df.at[idx, 'anomaly_type'] = new_type if not current_types else f"{current_types} & {new_type}"
                  df.at[idx, 'anomaly_score'] = max(df.at[idx, 'anomaly_score'], 0.7)
                  df.at[idx, 'detection_confidence'] = max(df.at[idx, 'detection_confidence'], 0.8)

              # Check for unusually low consumption on days that should be open
              open_days_mask = ~closed_days_mask
              if len(meter_data[open_days_mask]) >= 5:
                  open_day_avg = meter_data[open_days_mask]['daily_consumption'].median()
                  open_day_min = open_day_avg * 0.3  # Threshold for suspiciously low

                  low_consumption_indices = meter_data[
                      open_days_mask &
                      (meter_data['daily_consumption'] < open_day_min) &
                      (~meter_data['is_refill'])
                  ].index

                  for idx in low_consumption_indices:
                      df.at[idx, 'is_anomaly'] = True
                      try:
                          self.data_loader.log_anomaly_to_database(df.loc[idx])

                      except Exception as e:
                          print(f"Error logging anomaly to database: {str(e)}")
                      current_types = df.at[idx, 'anomaly_type']
                      day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                      day_name = day_names[meter_data.loc[idx, 'day_of_week']]
                      new_type = f'Unexpectedly low consumption on {day_name} (possible closure)'
                      df.at[idx, 'anomaly_type'] = new_type if not current_types else f"{current_types} & {new_type}"
                      df.at[idx, 'anomaly_score'] = max(df.at[idx, 'anomaly_score'], 0.65)
                      df.at[idx, 'detection_confidence'] = max(df.at[idx, 'detection_confidence'], 0.7)

          # Fire stations should have very consistent baseline with min/max thresholds
          elif building_type == 'fire_station':
              if len(meter_data) >= 14:
                  # Fire stations should have a minimum baseline usage
                  min_usage = meter_data['daily_consumption'].quantile(0.1) * 0.7
                  # And unusual spikes could indicate issues
                  max_usage = meter_data['daily_consumption'].quantile(0.9) * 1.5

                  # Check for violations of either threshold
                  violation_indices = meter_data[
                      ((meter_data['daily_consumption'] < min_usage) |
                      (meter_data['daily_consumption'] > max_usage)) &
                      (~meter_data['is_refill'])
                  ].index

                  for idx in violation_indices:
                        df.at[idx, 'is_anomaly'] = True
                        
                        # Calculate and set expected consumption based on the violation type
                        if df.loc[idx, 'daily_consumption'] < min_usage:
                            new_type = 'Fire station below essential usage'
                            df.at[idx, 'expected_consumption'] = min_usage  # Set expected consumption
                        else:
                            new_type = 'Fire station excessive usage'
                            df.at[idx, 'expected_consumption'] = max_usage  # Set expected consumption
                        
                        current_types = df.at[idx, 'anomaly_type']
                        df.at[idx, 'anomaly_type'] = new_type if not current_types else f"{current_types} & {new_type}"
                        df.at[idx, 'anomaly_score'] = max(df.at[idx, 'anomaly_score'], 0.7)
                        df.at[idx, 'detection_confidence'] = max(df.at[idx, 'detection_confidence'], 0.8)
                        
                        # Log after setting all values
                        try:
                            self.data_loader.log_anomaly_to_database(df.loc[idx])
                        except Exception as e:
                            print(f"Error logging anomaly to database: {str(e)}")

          # Community facilities often have event-based usage
          elif building_type == 'community':
              # These buildings have more variable usage, but check for extreme outliers
              if len(meter_data) >= 21:
                  rolling_median = meter_data['daily_consumption'].rolling(7, center=True, min_periods=3).median()
                  rolling_mad = (meter_data['daily_consumption'] - rolling_median).abs().rolling(7, center=True, min_periods=3).median()

                  for idx, row in meter_data.iterrows():
                      if pd.isna(rolling_median.loc[idx]) or pd.isna(rolling_mad.loc[idx]) or rolling_mad.loc[idx] == 0:
                          continue

                      # Use median absolute deviation
                      mad_score = abs(row['daily_consumption'] - rolling_median.loc[idx]) / max(1.0, rolling_mad.loc[idx])

                      if mad_score > 5.0 and not row['is_refill']:
                          df.at[idx, 'is_anomaly'] = True
                          try:
                            self.data_loader.log_anomaly_to_database(df.loc[idx])

                          except Exception as e:
                            print(f"Error logging anomaly to database: {str(e)}")
                          current_types = df.at[idx, 'anomaly_type']
                          new_type = 'Community center unusual usage'
                          df.at[idx, 'anomaly_type'] = new_type if not current_types else f"{current_types} & {new_type}"
                          df.at[idx, 'anomaly_score'] = max(df.at[idx, 'anomaly_score'], 0.6)
                          df.at[idx, 'detection_confidence'] = max(df.at[idx, 'detection_confidence'], 0.65)

      return df

    def _classify_building_type(self, building_name):
        """Determine building type based on name for specialized rules"""
        building_name = building_name.lower()

        if any(term in building_name for term in ['office', 'engineers', 'md office']):
            return 'office'
        elif 'library' in building_name:
            return 'library'
        elif 'fire' in building_name:
            return 'fire_station'
        elif any(term in building_name for term in ['hall', 'centre', 'center', 'community']):
            return 'community'
        elif 'depot' in building_name or 'environmental' in building_name:
            return 'industrial'
        else:
            return 'general'
        
    def save_isolation_forest(self, meter_id):
        """Save Isolation Forest model to disk"""
        model_path = os.path.join(self.model_dir, f'isolation_forest_{meter_id}.pkl')
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.isolation_forest, f)
            print(f"Successfully saved Isolation Forest for meter {meter_id}")
            return True
        except Exception as e:
            print(f"Error saving Isolation Forest for meter {meter_id}: {str(e)}")
            return False

    def load_isolation_forest(self, meter_id):
        """Load Isolation Forest model from disk if available"""
        model_path = os.path.join(self.model_dir, f'isolation_forest_{meter_id}.pkl')
        if not os.path.exists(model_path):
            print(f"No saved Isolation Forest found for meter {meter_id}")
            return False

        try:
            with open(model_path, 'rb') as f:
                self.isolation_forest = pickle.load(f)
            print(f"Successfully loaded Isolation Forest for meter {meter_id}")
            return True
        except Exception as e:
            print(f"Error loading Isolation Forest for meter {meter_id}: {str(e)}")
            return False


    def detect_anomalies(self, df):
        print("Starting enhanced anomaly detection...")

        try:
            # Extract features including improved tank refill detection
            df = self.extract_features(df)

            # Explicitly create rolling statistics with larger windows for stability
            print("Calculating rolling statistics...")
            df['rolling_mean'] = df.groupby('meter_id')['daily_consumption'].transform(
                lambda x: x.rolling(14, min_periods=3).mean()  # Increased from 7 days
            )

            df['rolling_std'] = df.groupby('meter_id')['daily_consumption'].transform(
                lambda x: x.rolling(14, min_periods=3).std().fillna(x.std())  # Increased from 7 days
            )

            # Initialize anomaly columns
            df['is_anomaly'] = False
            df['is_candidate_anomaly'] = False  # NEW: candidate anomalies before persistence check
            df['anomaly_score'] = 0.0
            df['anomaly_type'] = ''
            df['expected_consumption'] = np.nan
            df['detection_confidence'] = 0.0
            df['detection_methods'] = ''  # NEW: track which methods detected the anomaly

            # NEW: Calculate building-specific thresholds
            print("Calculating building-specific thresholds...")
            building_thresholds = {}
            for meter_id in df['meter_id'].unique():
                meter_data = df[df['meter_id'] == meter_id].copy()
                if len(meter_data) >= 30:  # Only calculate if enough data
                    building_thresholds[meter_id] = self.calculate_optimal_threshold(meter_data)
                else:
                    building_thresholds[meter_id] = 3.0  # Default threshold

            print(f"Building-specific thresholds: {building_thresholds}")

            # Enhanced feature set for ML
            feature_cols = [
                'daily_consumption', 'rolling_mean', 'rolling_std',
                'day_sin', 'day_cos', 'month_sin', 'month_cos',
                'is_holiday', 'is_weekend', 'days_since_refill',
                'zscore', 'consumption_diff_3day', 'consumption_diff_7day'
            ]

            # Add temperature if available
            if 'avg_temperature' in df.columns:
                feature_cols.append('avg_temperature')
                feature_cols.append('consumption_to_temp_ratio')

            # Make sure all feature columns exist (fill with zeros if not)
            for col in feature_cols:
                if col not in df.columns:
                    #print(f"Warning: Column '{col}' not found. Creating with default values.")
                    df[col] = 0.0

            # Process each meter
            meter_groups = df.groupby('meter_id').size().sort_values(ascending=False)
            meters = meter_groups.index
            print(f"Processing {len(meters)} meters...")

            for meter_idx, meter_id in enumerate(meters, 1):
                print(f"\nProcessing meter {meter_id} ({meter_idx}/{len(meters)})...")

                # Use the building-specific threshold
                threshold = building_thresholds.get(meter_id, 3.0)
                print(f"Using threshold {threshold} for meter {meter_id}")

                meter_data = df[df['meter_id'] == meter_id].copy()

                # Skip meters with insufficient data
                if len(meter_data) < 30:
                    print(f"Skipping meter {meter_id}: insufficient data (only {len(meter_data)} points)")
                    continue

                try:
                    # Calculate typical refill cycle for this meter
                    refill_points = meter_data[meter_data['is_refill']].index
                    if len(refill_points) >= 2:
                        cycle_lengths = np.diff(refill_points)
                        typical_cycle = np.median(cycle_lengths)
                        cycle_std = np.std(cycle_lengths)
                    else:
                        typical_cycle = 30  # Default assumption
                        cycle_std = 7

                    # Filter out columns that don't exist or have all NaN values
                    valid_features = [col for col in feature_cols if col in meter_data.columns and not meter_data[col].isna().all()]

                    if len(valid_features) < 5:
                        print(f"Warning: Only {len(valid_features)} valid features for meter {meter_id}")
                        print(f"Valid features: {valid_features}")

                    # Prepare ML features
                    X = meter_data[valid_features].fillna(0).values

                    # Try to load saved models first
                    models_loaded = False
                    if len(X) >= 60:  # Only try loading if we would have trained models
                        lstm_loaded = self.lstm_detector.load_model(meter_id)
                        if_loaded = self.load_isolation_forest(meter_id)
                        models_loaded = lstm_loaded and if_loaded
                        if models_loaded:
                            print(f"Successfully loaded existing models for meter {meter_id}")

                    # Train models if needed (only if enough data and models weren't loaded)
                    if len(X) >= 60 and not models_loaded:
                        print(f"Training new models for meter {meter_id}...")
                        # Configure Isolation Forest with reduced contamination
                        self.isolation_forest = IsolationForest(contamination=0.03, random_state=42)  # Reduced from 0.05
                        self.isolation_forest.fit(X)

                        # Save the Isolation Forest model
                        self.save_isolation_forest(meter_id)

                        try:
                            # Train LSTM with more epochs for better convergence
                            self.lstm_detector.fit(X, epochs=10)
                            # Save the LSTM model
                            self.lstm_detector.save_model(meter_id)
                        except Exception as e:
                            print(f"ML model training failed for meter {meter_id}: {str(e)}")
                            # Create dummy results as fallback
                            if_predictions = np.zeros(len(X))
                            lstm_anomalies = np.zeros(len(X), dtype=bool)
                            lstm_scores = np.zeros(len(X))
                    elif len(X) < 60:
                        # Not enough data, use simple statistical approaches instead
                        print(f"Not enough data for ML models for meter {meter_id}, using statistical approaches")
                        if_predictions = np.zeros(len(X))
                        lstm_anomalies = np.zeros(len(X), dtype=bool)
                        lstm_scores = np.zeros(len(X))

                    # Get predictions from loaded or newly trained models
                    try:
                        if len(X) >= 60:  # Only try prediction if we have enough data
                            if_predictions = self.isolation_forest.predict(X)
                            lstm_anomalies, lstm_scores = self.lstm_detector.predict(X)
                        else:
                            # Not enough data for ML predictions
                            if_predictions = np.zeros(len(X))
                            lstm_anomalies = np.zeros(len(X), dtype=bool)
                            lstm_scores = np.zeros(len(X))
                    except Exception as e:
                        print(f"Prediction failed for meter {meter_id}: {str(e)}")
                        # Create dummy results as fallback
                        if_predictions = np.zeros(len(X))
                        lstm_anomalies = np.zeros(len(X), dtype=bool)
                        lstm_scores = np.zeros(len(X))

                    # Process each day with improved thresholds
                    for idx, row in meter_data.iterrows():
                        anomaly_score = 0.0
                        anomaly_types = []
                        confidence_factors = []
                        detection_methods = []

                        # Skip anomaly detection if this is a refill day
                        if not row['is_refill']:
                            # 1. Z-score based detection with building-specific threshold
                            if 'zscore' in row and pd.notna(row['zscore']) and abs(row['zscore']) > threshold:  # Use building-specific threshold
                                anomaly_score += 0.4  # Reduced contribution from 0.5
                                anomaly_types.append(f'Statistical outlier (z-score: {row["zscore"]:.1f})')
                                confidence_factors.append(min(1.0, abs(row['zscore']) / 5))
                                detection_methods.append('zscore')

                            # 2. Day-of-week pattern with building-specific threshold
                            if 'day_of_week' in row and pd.notna(row['day_of_week']):
                                dow_data = meter_data[meter_data['day_of_week'] == row['day_of_week']]
                                if len(dow_data) >= 4:  # Need more examples for stability
                                    dow_mean = dow_data['daily_consumption'].mean()
                                    dow_std = dow_data['daily_consumption'].std()
                                    if dow_std > 0:
                                        dow_zscore = abs(row['daily_consumption'] - dow_mean) / dow_std
                                        if dow_zscore > threshold:  # Use building-specific threshold
                                            anomaly_score += 0.25  # Reduced from 0.3
                                            anomaly_types.append(f'Unusual day-of-week pattern (z: {dow_zscore:.1f})')
                                            confidence_factors.append(min(1.0, dow_zscore / 4.0))
                                            detection_methods.append('day_pattern')

                            # 3. Check consumption relative to cycle position with building-specific threshold
                            if 'days_since_refill' in row and 'rolling_mean' in row and 'rolling_std' in row and row['rolling_std'] > 0:
                                days_in_cycle = row['days_since_refill']
                                if days_in_cycle < typical_cycle * 1.5:
                                    expected_consumption = row['rolling_mean']
                                    consumption_deviation = abs(row['daily_consumption'] - expected_consumption)
                                    if consumption_deviation > threshold * 0.7 * row['rolling_std']:  # Adjusted threshold
                                        anomaly_score += 0.25  # Reduced from 0.3
                                        anomaly_types.append('Unusual cycle consumption')
                                        confidence_factors.append(min(1.0, consumption_deviation / (3.5 * row['rolling_std'])))
                                        detection_methods.append('cycle')

                            # 4. Holiday checks with building-specific threshold
                            if 'is_holiday' in row and row['is_holiday']:
                                holiday_data = meter_data[meter_data['is_holiday'] == 1]
                                if len(holiday_data) >= 3:  # Need enough holiday examples
                                    holiday_mean = holiday_data['daily_consumption'].mean()
                                    holiday_std = holiday_data['daily_consumption'].std()
                                    if holiday_std > 0:
                                        holiday_zscore = (row['daily_consumption'] - holiday_mean) / holiday_std
                                        if abs(holiday_zscore) > threshold:  # Use building-specific threshold
                                            anomaly_score += 0.25  # Reduced from 0.3
                                            anomaly_types.append('Unusual holiday consumption')
                                            confidence_factors.append(min(1.0, abs(holiday_zscore) / 3.5))
                                            detection_methods.append('holiday')

                            # 5. Weekend checks with building-specific threshold
                            if 'is_weekend' in row and row['is_weekend']:
                                weekend_data = meter_data[meter_data['is_weekend'] == 1]
                                if len(weekend_data) >= 4:  # Need enough weekend examples
                                    weekend_mean = weekend_data['daily_consumption'].mean()
                                    weekend_std = weekend_data['daily_consumption'].std()
                                    if weekend_std > 0:
                                        weekend_zscore = (row['daily_consumption'] - weekend_mean) / weekend_std
                                        if abs(weekend_zscore) > threshold:  # Use building-specific threshold
                                            anomaly_score += 0.25  # Reduced from 0.3
                                            anomaly_types.append('Unusual weekend consumption')
                                            confidence_factors.append(min(1.0, abs(weekend_zscore) / 3.5))
                                            detection_methods.append('weekend')

                            # 6. Rapid change detection with building-specific threshold
                            if 'consumption_diff' in row and pd.notna(row['consumption_diff']):
                                consumption_diff_std = meter_data['consumption_diff'].std()
                                if consumption_diff_std > 0 and abs(row['consumption_diff']) > threshold * 1.1 * consumption_diff_std:  # Adjusted threshold
                                    anomaly_score += 0.3  # Reduced from 0.4
                                    anomaly_types.append('Sudden consumption change')
                                    confidence_factors.append(0.7)
                                    detection_methods.append('rapid_change')

                            # 7. ML model results with consensus requirement
                            try:
                                local_idx = meter_data.index.get_loc(idx)
                                # NEW: Require consensus from multiple methods
                                ml_methods_agree = False

                                # Only count ML methods if both models agree
                                if (local_idx < len(if_predictions) and if_predictions[local_idx] == -1 and
                                    local_idx < len(lstm_anomalies) and lstm_anomalies[local_idx]):
                                    ml_methods_agree = True
                                    anomaly_score += 0.4  # Increased from separate 0.25+0.25
                                    anomaly_types.append('ML model detection (multiple methods agree)')
                                    confidence_factors.append(0.9)
                                    detection_methods.append('ml_consensus')
                                # Or if LSTM is very confident
                                elif local_idx < len(lstm_anomalies) and lstm_anomalies[local_idx] and lstm_scores[local_idx] > 1.5:
                                    anomaly_score += 0.3
                                    anomaly_types.append('LSTM detection (high confidence)')
                                    confidence_factors.append(min(1.0, lstm_scores[local_idx] / 1.5))
                                    detection_methods.append('lstm')
                            except Exception as e:
                                print(f"Error in ML result processing for index {idx}: {str(e)}")

                            # 8. Temperature impact check only if strong correlation exists
                            if ('avg_temperature' in row and pd.notna(row['avg_temperature']) and
                                'avg_temperature' in meter_data.columns and not meter_data['avg_temperature'].isna().all()):

                                # Calculate correlation between temp and consumption
                                temp_corr = meter_data['daily_consumption'].corr(meter_data['avg_temperature'])
                                # Only if there's strong correlation
                                if abs(temp_corr) > 0.5:  # Increased from 0.3
                                    # Check if temperature change matches expected consumption change
                                    if 'temp_change' in row and 'consumption_diff' in row:
                                        temp_impact = abs(row['temp_change']) > 1.5 * meter_data['temp_change'].std()
                                        consumption_impact = abs(row['consumption_diff']) > meter_data['consumption_diff'].std()

                                        # Temp went up but consumption didn't change as expected
                                        expected_relationship = (temp_corr > 0 and
                                                            ((row['temp_change'] > 0 and row['consumption_diff'] < 0) or
                                                            (row['temp_change'] < 0 and row['consumption_diff'] > 0)))

                                        if temp_impact and not consumption_impact and expected_relationship:
                                            anomaly_score += 0.2
                                            anomaly_types.append('Temperature-consumption mismatch')
                                            confidence_factors.append(0.6)
                                            detection_methods.append('temperature')

                        # Calculate confidence and determine if it's an anomaly candidate
                        confidence = np.mean(confidence_factors) if confidence_factors else 0.0
                        df.at[idx, 'anomaly_score'] = min(1.0, anomaly_score)
                        df.at[idx, 'detection_confidence'] = confidence
                        df.at[idx, 'detection_methods'] = ', '.join(detection_methods)

                        # Mark as a candidate anomaly if score is high enough or multiple methods agree
                        # This is before we check persistence
                        if anomaly_score > 0.55 or len(detection_methods) >= 2:
                            df.at[idx, 'is_candidate_anomaly'] = True
                            df.at[idx, 'anomaly_type'] = ' & '.join(anomaly_types)
                            if 'rolling_mean' in row and pd.notna(row['rolling_mean']):
                                df.at[idx, 'expected_consumption'] = row['rolling_mean']
                            else:
                                # If rolling_mean is not available, use overall meter average
                                df.at[idx, 'expected_consumption'] = meter_data['daily_consumption'].mean()

                except Exception as e:
                    print(f"Error processing meter {meter_id}: {str(e)}")
                    continue

            # NEW: Apply persistence filter - only mark as true anomaly if consistent across multiple days
            # or if extremely high score (>0.8) for a single day
            for meter_id in df['meter_id'].unique():
                meter_indices = df[df['meter_id'] == meter_id].index
                sorted_indices = sorted(meter_indices)  # Ensure chronological order

                # Process each day except first and last (need before/after context)
                for i in range(1, len(sorted_indices)-1):
                    idx = sorted_indices[i]
                    prev_idx = sorted_indices[i-1]
                    next_idx = sorted_indices[i+1]

                    # Flag as true anomaly if:
                    # 1. It's an extreme anomaly (score > 0.8)
                    if df.at[idx, 'anomaly_score'] > 0.8:
                        df.at[idx, 'is_anomaly'] = True
                        try:
                            self.data_loader.log_anomaly_to_database(df.loc[idx])

                        except Exception as e:
                            print(f"Error logging anomaly to database: {str(e)}")
                        continue

                    # 2. Or it's persistent (candidate anomaly with candidates before/after)
                    if (df.at[idx, 'is_candidate_anomaly'] and
                       (df.at[prev_idx, 'is_candidate_anomaly'] or df.at[next_idx, 'is_candidate_anomaly'])):
                        df.at[idx, 'is_anomaly'] = True
                        # Boost confidence if persistent
                        df.at[idx, 'detection_confidence'] = min(1.0, df.at[idx, 'detection_confidence'] + 0.1)
                        # Log to database
                        try:
                            self.data_loader.log_anomaly_to_database(df.loc[idx])

                        except Exception as e:
                            print(f"Error logging anomaly to database: {str(e)}")
                    # 3. Or multiple detection methods agree (3 or more)
                    elif df.at[idx, 'is_candidate_anomaly'] and len(df.at[idx, 'detection_methods'].split(',')) >= 3:
                        df.at[idx, 'is_anomaly'] = True
                        # Boost confidence for multiple methods
                        df.at[idx, 'detection_confidence'] = min(1.0, df.at[idx, 'detection_confidence'] + 0.15)
                        # Log to database
                        try:
                            self.data_loader.log_anomaly_to_database(df.loc[idx])

                        except Exception as e:
                            print(f"Error logging anomaly to database: {str(e)}")
                    # Otherwise, it's not a confirmed anomaly
                    else:
                        df.at[idx, 'is_anomaly'] = False

            # Apply domain knowledge rules
            try:
                df = self.add_domain_knowledge_rules(df)
            except Exception as e:
                print(f"Error applying domain knowledge rules: {str(e)}")

            print("Daily anomaly detection completed")
            return df

        except Exception as e:
            print(f"Critical error in anomaly detection: {type(e).__name__}: {str(e)}")
            print(traceback.format_exc())
    
    # Return the original dataframe without modifications
        return df
        
    def generate_anomaly_explanation(self, anomaly_row):
        explanation = []

        # Get building name if available
        building_name = METER_TO_BUILDING.get(str(anomaly_row['meter_id']), f"Building {anomaly_row['meter_id']}")

        # Get day name
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_name = day_names[anomaly_row['day_of_week']]

        # Basic information with more direct formatting
        explanation.append(f"Building: {building_name}")
        explanation.append(f"Date: {anomaly_row['date'].strftime('%A, %B %d, %Y')}")
        explanation.append(f"Consumption: {anomaly_row['daily_consumption']:.2f} kWh")

        # Expected consumption with clearer difference metrics
        if not pd.isna(anomaly_row['expected_consumption']):
            # Calculate percent and absolute difference
            pct_diff = ((anomaly_row['daily_consumption'] - anomaly_row['expected_consumption']) /
                        max(0.01, anomaly_row['expected_consumption']) * 100)
            abs_diff = anomaly_row['daily_consumption'] - anomaly_row['expected_consumption']

            explanation.append(f"Expected: {anomaly_row['expected_consumption']:.2f} kWh")
            explanation.append(f"Difference: {abs_diff:.2f} kWh ({pct_diff:.1f}% {'higher' if pct_diff > 0 else 'lower'})")

            # Categorize the severity based on percentage difference
            if abs(pct_diff) > 500:
                severity = "CRITICAL"
            elif abs(pct_diff) > 200:
                severity = "MAJOR"
            elif abs(pct_diff) > 100:
                severity = "SIGNIFICANT"
            elif abs(pct_diff) > 50:
                severity = "MODERATE"
            else:
                severity = "MINOR"

            explanation.append(f"Severity: {severity}")

        # Context information in a more structured way
        context_parts = []
        if 'avg_temperature' in anomaly_row and not pd.isna(anomaly_row['avg_temperature']):
            temp = anomaly_row['avg_temperature']
            if 'temp_change' in anomaly_row and not pd.isna(anomaly_row['temp_change']):
                temp_change = anomaly_row['temp_change']
                context_parts.append(f"Temperature: {temp:.1f}°C ({'+' if temp_change > 0 else ''}{temp_change:.1f}°C change)")
            else:
                context_parts.append(f"Temperature: {temp:.1f}°C")

        # Specific day context
        if anomaly_row['is_holiday']:
            context_parts.append(f"{day_name} (Holiday)")
        elif anomaly_row['is_weekend']:
            context_parts.append(f"{day_name} (Weekend)")
        else:
            context_parts.append(f"{day_name} (Weekday)")

        if context_parts:
            explanation.append("Context: " + " | ".join(context_parts))

        # Tank refill information
        if 'days_since_refill' in anomaly_row:
            days = anomaly_row['days_since_refill']
            if 'cycle_position' in anomaly_row and not pd.isna(anomaly_row['cycle_position']):
                pos = anomaly_row['cycle_position']
                explanation.append(f"Tank status: {days} days since last refill (approximately {pos:.0f}% through typical cycle)")
            else:
                explanation.append(f"Tank status: {days} days since last refill")

        # Generate precise anomaly descriptions
        precise_anomalies = []

        # Handle day-of-week anomalies
        if 'dow_zscore' in anomaly_row and not pd.isna(anomaly_row['dow_zscore']) and abs(anomaly_row['dow_zscore']) > 2:
            z = anomaly_row['dow_zscore']
            direction = "higher" if z > 0 else "lower"
            precise_anomalies.append(f"Consumption is {abs(z):.1f} standard deviations {direction} than typical {day_name}s")

        # Handle weekend/weekday specific patterns
        if 'is_weekend' in anomaly_row and anomaly_row['is_weekend']:
            if 'Unusual weekend' in str(anomaly_row['anomaly_type']):
                precise_anomalies.append(f"Abnormal usage pattern for a weekend day (typical weekend consumption is much lower)")
        elif 'Unusual day-of-week' in str(anomaly_row['anomaly_type']):
            precise_anomalies.append(f"Abnormal usage pattern for a {day_name}")

        # Handle sudden changes
        if 'Sudden consumption change' in str(anomaly_row['anomaly_type']):
            if 'consumption_diff' in anomaly_row and not pd.isna(anomaly_row['consumption_diff']):
                diff = anomaly_row['consumption_diff']
                change_direction = 'increase' if diff > 0 else 'decrease'
                
                # Add context about why this change is flagged as an anomaly
                if (change_direction == 'decrease' and anomaly_row['daily_consumption'] > anomaly_row.get('expected_consumption', 0)):
                    precise_anomalies.append(f"Abrupt {change_direction} of {abs(diff):.2f} kWh from previous day (still higher than expected)")
                elif (change_direction == 'increase' and anomaly_row['daily_consumption'] < anomaly_row.get('expected_consumption', 0)):
                    precise_anomalies.append(f"Abrupt {change_direction} of {abs(diff):.2f} kWh from previous day (but still below expected)")
                else:
                    precise_anomalies.append(f"Abrupt {change_direction} of {abs(diff):.2f} kWh from previous day")

        # Handle holiday anomalies
        if 'holiday' in str(anomaly_row['anomaly_type']).lower() and anomaly_row['is_holiday']:
            precise_anomalies.append(f"Unusual consumption for a holiday (building should likely have reduced usage)")

        # Add specific detected anomaly explanations
        if precise_anomalies:
            explanation.append("SPECIFIC ANOMALIES DETECTED:")
            for anomaly in precise_anomalies:
                explanation.append(f"• {anomaly}")

        # Generate clear, specific causes based on patterns and magnitude
        # Calculate pct_diff properly
        pct_diff = 0
        if not pd.isna(anomaly_row.get('expected_consumption')) and anomaly_row.get('expected_consumption', 0) > 0:
            pct_diff = ((anomaly_row['daily_consumption'] - anomaly_row['expected_consumption']) / 
                        anomaly_row['expected_consumption'] * 100)
        pct_diff_abs = abs(pct_diff)

        # Determine potential causes based on anomaly magnitude and context
        causes = []  # Ensure this is initialized

        if pct_diff_abs > 500:
            causes.append("CRITICAL: Potential major gas leak or system failure")
            causes.append("Meter reading error or data corruption")
            causes.append("Critical equipment malfunction")
        elif pct_diff_abs > 200:
            causes.append("ALERT: Significant system inefficiency or partial leak")
            causes.append("Unauthorized usage or equipment malfunction")
            causes.append("Multiple systems operating outside normal parameters")
        elif pct_diff_abs > 100:
            if anomaly_row.get('is_weekend', False) and not anomaly_row.get('is_holiday', False):
                causes.append("Unexpected weekend operation (building normally closed)")
            if anomaly_row.get('is_holiday', False):
                causes.append("Unexpected holiday operation (building normally closed)")
            causes.append("Heating/cooling system malfunction")
            causes.append("Major deviation from scheduled operation")
        elif pct_diff_abs > 50:
            causes.append("Equipment left running outside normal hours")
            causes.append("Inefficient operation of heating/cooling systems")
            causes.append("Change in building usage not reflected in baseline")
        else:
            causes.append("Minor variation from expected pattern")
            causes.append("Potential early indicator of equipment degradation")

        # Add default cause if none were added but it's an anomaly
        if not causes and anomaly_row.get('is_anomaly', False):
            causes.append("Unusual consumption pattern detected")
            causes.append("Potential operation outside of expected schedule")

        # Add the causes to the explanation
        if causes:
            explanation.append("POTENTIAL CAUSES:")
            for cause in causes:
                explanation.append(f"• {cause}")

        # Generate actionable recommendations
        actions = []

        # Set recommendations based on severity
        if pct_diff_abs > 500:
            actions.append("URGENT: Dispatch maintenance team immediately")
            actions.append("Check for physical signs of gas leakage or major equipment failure")
            actions.append("Verify meter reading accuracy and data transmission")
            actions.append("Consider emergency shutdown if safety concerns exist")
        elif pct_diff_abs > 200:
            actions.append("Schedule urgent inspection within 24 hours")
            actions.append("Review all major equipment operation status")
            actions.append("Check building scheduling and occupancy systems")
            actions.append("Verify no unauthorized usage is occurring")
        elif pct_diff_abs > 100:
            actions.append("Schedule inspection within 48 hours")
            actions.append("Review building operation schedule and recent changes")
            actions.append("Check HVAC control settings and operation")
        elif pct_diff_abs > 50:
            actions.append("Review building schedule and occupancy patterns")
            actions.append("Check timer settings on major equipment")
            actions.append("Inspect for equipment running outside needed hours")
        else:
            actions.append("Monitor in coming days for pattern development")
            actions.append("No immediate action required unless pattern continues")

        # Add default actions if none were added but it's an anomaly
        if not actions and anomaly_row.get('is_anomaly', False):
            actions.append("Review building operation schedule")
            actions.append("Check for unauthorized usage or system malfunctions")

        # Add the actions to explanation
        if actions:
            explanation.append("RECOMMENDED ACTIONS:")
            for action in actions:
                explanation.append(f"• {action}")

        # Add confidence score with interpretation
        conf = anomaly_row['detection_confidence']
        conf_text = "Low" if conf < 0.4 else "Medium" if conf < 0.7 else "High"
        explanation.append(f"Detection confidence: {conf:.2f} ({conf_text})")

        return "\n".join(explanation)

