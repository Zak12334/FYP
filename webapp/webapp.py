from flask import Flask, jsonify, request
from flask_cors import CORS
from dash import Dash, html, dcc
from dash.dependencies import Input as DashInput, Output as DashOutput
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
import traceback
import numpy as np

# Import your project modules using correct paths
from analytics.analytics import EnhancedSmartMeterAnalytics
from data.data import SmartMeterDataLoader

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

class SmartMeterDashboard:
    def __init__(self):
        # Initialize Flask server and enable CORS
        self.server = Flask(__name__)
        CORS(self.server,
             resources={r"/*": {"origins": "*"}},
             supports_credentials=True,
             methods=['GET', 'POST', 'OPTIONS'])
        # Initialize Dash using the Flask server
        self.app = Dash(__name__, server=self.server,
                        external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
        
        self.data_loader = SmartMeterDataLoader(
            host='162.19.76.199',
            user='root',
            password='YNMIc3sF6b!Jsg8OKdUf',
            db='elighthouse_building',
            port=3307
        )
        self.analytics = EnhancedSmartMeterAnalytics(data_loader=self.data_loader)
        # Load data and run anomaly detection
        self.df = self.data_loader.get_data()
        self.df = self.analytics.detect_anomalies(self.df)
        
        # Setup API routes, Dash layout, and callbacks
        self.setup_routes()
        self.setup_layout()
        self.setup_callbacks()

    def fetch_temperature_data(self, start_date=None, end_date=None):
        """Create simple temperature data"""
        # Get the date range from your consumption data
        all_dates = pd.date_range(start=self.df['date'].min(), end=self.df['date'].max(), freq='D')
        
        # Create synthetic temperature data (constant with small variations)
        temp_df = pd.DataFrame({
            'date': all_dates,
            'temperature': [10.0 + np.random.normal(0, 0.2) for _ in range(len(all_dates))]
        })
        
        # Filter if needed
        if start_date and end_date:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            temp_df = temp_df[(temp_df['date'] >= start_date) & (temp_df['date'] <= end_date)]
        
        return temp_df

    def setup_routes(self):
        @self.server.route('/meters', methods=['GET', 'OPTIONS'])
        def get_meters():
            try:
                result = {
                    "meters": list(self.df['meter_id'].unique()),
                    "meter_map": {m: METER_TO_BUILDING.get(str(m), f'Building {m}')
                                  for m in self.df['meter_id'].unique()}
                }
                return jsonify(result)
            except Exception as e:
                print(f"Error in get_meters: {e}")
                return jsonify({"error": str(e)}), 500

        @self.server.route('/data', methods=['GET', 'OPTIONS'])
        def get_data():
            try:
                meter_id = request.args.get('meter_id', type=int)
                start_date = request.args.get('start_date')
                end_date = request.args.get('end_date')

                filtered_data = self.df.copy()
                if meter_id:
                    filtered_data = filtered_data[filtered_data['meter_id'] == str(meter_id)]
                if start_date:
                    filtered_data = filtered_data[filtered_data['date'] >= start_date]
                if end_date:
                    filtered_data = filtered_data[filtered_data['date'] <= end_date]

                return jsonify(filtered_data.to_dict(orient='records'))
            except Exception as e:
                print(f"Error in get_data: {e}")
                return jsonify({"error": str(e)}), 500

        @self.server.route('/anomalies', methods=['GET', 'OPTIONS'])
        def get_anomalies():
            try:
                meter_id = request.args.get('meter_id', type=int)
                threshold = request.args.get('threshold', default=0.5, type=float)
                start_date = request.args.get('start_date')
                end_date = request.args.get('end_date')

                anomalies = self.df[self.df['anomaly_score'] > threshold]
                if meter_id:
                    anomalies = anomalies[anomalies['meter_id'] == str(meter_id)]
                if start_date:
                    anomalies = anomalies[anomalies['date'] >= start_date]
                if end_date:
                    anomalies = anomalies[anomalies['date'] <= end_date]

                return jsonify(anomalies.to_dict(orient='records'))
            except Exception as e:
                print(f"Error in get_anomalies: {e}")
                return jsonify({"error": str(e)}), 500

    def setup_layout(self):
        # Define the Dash layout.
        # (You can embed your complete layout code here as in your original non-OOP version.)
        self.app.layout = html.Div([
            html.Div([
                html.H1("Cork County Council Smart Meter Analytics Dashboard",
                        style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px',
                               'marginTop': '20px', 'fontFamily': 'Arial, sans-serif'}),
            ]),
            html.Div([
                html.Div([
                    html.Div([
                        html.H3("Controls", style={'color': '#2c3e50', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
                        html.Label("Select Date Range", style={'fontWeight': 'bold', 'marginTop': '20px'}),
                        dcc.DatePickerRange(
                            id='date-range',
                            start_date=self.df['date'].min(),
                            end_date=self.df['date'].max(),
                            calendar_orientation='horizontal',
                            style={'marginBottom': '20px'}
                        ),
                        html.Label("Select Building", style={'fontWeight': 'bold', 'marginTop': '20px'}),
                        dcc.Dropdown(
                            id='meter-selector',
                            options=[{'label': METER_TO_BUILDING.get(str(m), f'Building {m}'), 'value': str(m)}
                                     for m in sorted(self.df['meter_id'].unique())],
                            value=str(self.df['meter_id'].iloc[0]) if not self.df.empty else None,
                            style={'marginBottom': '20px'}
                        ),
                        html.Label("Display Options", style={'fontWeight': 'bold', 'marginTop': '20px'}),
                        dcc.Checklist(
                            id='view-options',
                            options=[
                                {'label': ' Show Temperature', 'value': 'temp'},
                                {'label': ' Show Moving Average', 'value': 'ma'},
                                {'label': ' Highlight Weekends', 'value': 'weekend'},
                                {'label': ' Highlight Holidays', 'value': 'holiday'},
                                {'label': ' Show Seasonal Component', 'value': 'seasonal'}
                            ],
                            value=['temp', 'ma'],
                            style={'marginTop': '10px'}
                        )
                    ])
                ], className='row'),
                html.Div([
                    html.Div([
                        dcc.Graph(id='main-graph', style={'height': '500px'})
                    ], style={'marginBottom': '20px', 'padding': '20px',
                              'backgroundColor': 'white', 'borderRadius': '5px',
                              'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                    html.Div([
                        dcc.Graph(id='pattern-graph', style={'height': '400px'})
                    ], style={'marginBottom': '20px', 'padding': '20px',
                              'backgroundColor': 'white', 'borderRadius': '5px',
                              'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
                ], className='nine columns'),
            ], className='row'),
            html.Div([
                html.Div([
                    html.Div(id='anomaly-stats', style={'padding': '20px',
                                                         'backgroundColor': 'white',
                                                         'borderRadius': '5px',
                                                         'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                                                         'marginBottom': '20px'})
                ], className='six columns'),
                html.Div([
                    html.Div(id='consumption-summary', style={'padding': '20px',
                                                               'backgroundColor': 'white',
                                                               'borderRadius': '5px',
                                                               'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
                ], className='six columns'),
            ], className='row'),
            html.Div([
                html.Div(id='anomaly-details', style={'padding': '20px',
                                                       'backgroundColor': 'white',
                                                       'borderRadius': '5px',
                                                       'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                                                       'marginTop': '20px'})
            ], className='row')
        ], style={'padding': '20px', 'backgroundColor': '#f0f2f5', 'minHeight': '100vh'})

    def setup_callbacks(self):
        @self.app.callback(
            [DashOutput('main-graph', 'figure'),
             DashOutput('pattern-graph', 'figure'),
             DashOutput('anomaly-stats', 'children'),
             DashOutput('consumption-summary', 'children')],
            [DashInput('date-range', 'start_date'),
             DashInput('date-range', 'end_date'),
             DashInput('meter-selector', 'value'),
             DashInput('view-options', 'value')]
        )
        def update_main_graph(start_date, end_date, meter_id, view_options):
            print(f"Running update_main_graph with meter_id: {meter_id}, start_date: {start_date}, end_date: {end_date}")
            anomaly_threshold = 0.5
            meter_id = str(meter_id) if meter_id is not None else None

            print(f"All available meters: {self.df['meter_id'].unique()}")
            print(f"Date range in data: {self.df['date'].min()} to {self.df['date'].max()}")

            if start_date:
                start_date = pd.to_datetime(start_date)
            if end_date:
                end_date = pd.to_datetime(end_date)
            
            filtered_df = self.df[self.df['meter_id'].astype(str) == meter_id]
            print(f"After filtering by meter_id {meter_id}: {len(filtered_df)} rows")

            if len(filtered_df) == 0:
                print(f"WARNING: No data found for meter_id {meter_id}")
                empty_fig = go.Figure().update_layout(
                    title="No data available for the selected meter",
                    annotations=[{"text": "No data found", "showarrow": False, "font": {"size": 28}}]
                )
                empty_stats = html.Div([html.H4("No data available")])
                return empty_fig, empty_fig, empty_stats, empty_stats

            if start_date and end_date:
                if not pd.api.types.is_datetime64_any_dtype(filtered_df['date']):
                    filtered_df['date'] = pd.to_datetime(filtered_df['date'])
                filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]
                print(f"After date filtering: {len(filtered_df)} rows")

            main_fig = make_subplots(specs=[[{"secondary_y": True}]])
            main_fig.add_trace(
                go.Scatter(
                    x=filtered_df['date'],
                    y=filtered_df['daily_consumption'],
                    mode='lines+markers',
                    name='Daily Consumption',
                    line=dict(color='#3498db', width=2),
                    marker=dict(size=6)
                ),
                secondary_y=False
            )

            if 'ma' in view_options and len(filtered_df) > 7:
                filtered_df['rolling_mean'] = filtered_df['daily_consumption'].rolling(window=7, min_periods=1).mean()
                main_fig.add_trace(
                    go.Scatter(
                        x=filtered_df['date'],
                        y=filtered_df['rolling_mean'],
                        mode='lines',
                        name='7-Day Moving Avg',
                        line=dict(color='#2ecc71', width=2, dash='dash')
                    ),
                    secondary_y=False
                )

            # Temperature section
            if 'temp' in view_options:
                temp_df = self.fetch_temperature_data(start_date, end_date)
                
                if not temp_df.empty:
                    main_fig.add_trace(
                        go.Scatter(
                            x=temp_df['date'],
                            y=temp_df['temperature'],
                            mode='lines',
                            name='Temperature',
                            line=dict(color='#e74c3c', width=2)
                        ),
                        secondary_y=True
                    )
                    
                    # Set explicit range for temperature axis
                    main_fig.update_yaxes(
                        title_text="Temperature (°C)", 
                        range=[min(temp_df['temperature'])-1, max(temp_df['temperature'])+1],
                        secondary_y=True
                    )

            if 'seasonal' in view_options and 'seasonal_component' in filtered_df.columns:
                main_fig.add_trace(
                    go.Scatter(
                        x=filtered_df['date'],
                        y=filtered_df['seasonal_component'],
                        mode='lines',
                        name='Seasonal Pattern',
                        line=dict(color='#9b59b6', width=2)
                    ),
                    secondary_y=False
                )

            if 'weekend' in view_options and 'is_weekend' in filtered_df.columns:
                weekend_days = filtered_df[filtered_df['is_weekend'] == 1]
                main_fig.add_trace(
                    go.Scatter(
                        x=weekend_days['date'],
                        y=weekend_days['daily_consumption'],
                        mode='markers',
                        name='Weekends',
                        marker=dict(color='rgba(255, 193, 7, 0.6)', size=10, symbol='square')
                    ),
                    secondary_y=False
                )

            anomalies = filtered_df[filtered_df['is_anomaly'] == True]
            print(f"Anomalies found: {len(anomalies)} with threshold {anomaly_threshold}")

            if anomalies.empty:
                critical = pd.DataFrame()
                significant = pd.DataFrame()
                minor = pd.DataFrame()
                slight = pd.DataFrame()
            else:
                # Create severity_pct column safely
                anomalies = anomalies.copy()
                mask = (~anomalies['expected_consumption'].isna()) & (anomalies['expected_consumption'] > 0)
                anomalies.loc[mask, 'severity_pct'] = abs(
                    (anomalies.loc[mask, 'daily_consumption'] -
                    anomalies.loc[mask, 'expected_consumption']) /
                    anomalies.loc[mask, 'expected_consumption'] * 100
                )
                anomalies['severity_pct'] = anomalies['severity_pct'].fillna(0)

                critical = anomalies[anomalies['severity_pct'] > 100]
                significant = anomalies[(anomalies['severity_pct'] <= 100) & (anomalies['severity_pct'] > 50)]
                minor = anomalies[(anomalies['severity_pct'] <= 50) & (anomalies['severity_pct'] > 20)]
                slight = anomalies[anomalies['severity_pct'] <= 20]

                # Add markers for different anomaly types
                if not critical.empty:
                    main_fig.add_trace(
                        go.Scatter(
                            x=critical['date'],
                            y=critical['daily_consumption'],
                            mode='markers',
                            name='Critical Anomalies',
                            marker=dict(color='rgba(255, 0, 0, 0.9)', size=14, symbol='x', line=dict(width=2, color='red'))
                        ),
                        secondary_y=False
                    )
                if not significant.empty:
                    main_fig.add_trace(
                        go.Scatter(
                            x=significant['date'],
                            y=significant['daily_consumption'],
                            mode='markers',
                            name='Significant Anomalies',
                            marker=dict(color='rgba(255, 140, 0, 0.9)', size=12, symbol='triangle-up', line=dict(width=2, color='darkorange'))
                        ),
                        secondary_y=False
                    )
                if not minor.empty:
                    main_fig.add_trace(
                        go.Scatter(
                            x=minor['date'],
                            y=minor['daily_consumption'],
                            mode='markers',
                            name='Minor Anomalies',
                            marker=dict(color='rgba(30, 144, 255, 0.9)', size=10, symbol='circle', line=dict(width=2, color='dodgerblue'))
                        ),
                        secondary_y=False
                    )
                if not slight.empty:
                    main_fig.add_trace(
                        go.Scatter(
                            x=slight['date'],
                            y=slight['daily_consumption'],
                            mode='markers',
                            name='Slight Anomalies',
                            marker=dict(color='rgba(60, 179, 113, 0.9)', size=8, symbol='circle', line=dict(width=2, color='mediumseagreen'))
                        ),
                        secondary_y=False
                    )

            building_name = METER_TO_BUILDING.get(meter_id, f"Building {meter_id}")
            main_fig.update_layout(
                title=f"{building_name} - Consumption Analysis",
                xaxis_title="Date",
                xaxis=dict(
                    range=[start_date, end_date] if start_date and end_date else None
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=60, r=60, t=60, b=60),
                height=500,
                hovermode="x unified"
            )
            refill_days = filtered_df[filtered_df['is_refill']]
            if not refill_days.empty:
                main_fig.add_trace(
                    go.Scatter(
                        x=refill_days['date'],
                        y=refill_days['daily_consumption'],
                        mode='markers',
                        name='Refill Days',
                        marker=dict(
                            color='blue',  
                            size=10,  
                            symbol='circle',  
                            line=dict(
                                color='darkgreen',  
                                width=2
                            )
                        ),
                        hovertemplate=(
                            "<b>Refill Day</b><br>"
                            "Date: %{x}<br>"
                            "Consumption: %{y:.2f} kWh<extra></extra>"
                        )
                    ),
                    secondary_y=False
                )

            if 'holiday' in view_options and 'is_holiday' in filtered_df.columns:
                holiday_days = filtered_df[filtered_df['is_holiday'] == 1]
                if not holiday_days.empty:
                    main_fig.add_trace(
                        go.Scatter(
                            x=holiday_days['date'],
                            y=holiday_days['daily_consumption'],
                            mode='markers',
                            name='Holidays',
                            marker=dict(
                                color='orange',  # Distinctive color
                                size=10,  # Consistent with other markers
                                symbol='star',   # Distinctive shape
                                line=dict(
                                    color='red',  # Outline color
                                    width=2
                                )
                            )
                        ),
                        secondary_y=False
                    )
            main_fig.update_yaxes(title_text="Daily Consumption (kWh)", secondary_y=False)
            main_fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True)

            pattern_fig = make_subplots(rows=1, cols=2,
                                        subplot_titles=('Consumption by Day of Week', 'Monthly Consumption Pattern'),
                                        specs=[[{"type": "bar"}, {"type": "bar"}]])
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            if 'day_of_week' in filtered_df.columns:
                daily_avg = filtered_df.groupby('day_of_week')['daily_consumption'].mean()
                daily_avg = daily_avg.reindex(range(7), fill_value=0)
                pattern_fig.add_trace(
                    go.Bar(
                        x=day_names,
                        y=daily_avg.values,
                        marker_color='#3498db',
                        name='Avg by Day'
                    ),
                    row=1, col=1
                )
                if len(filtered_df) > 14:
                    daily_std = filtered_df.groupby('day_of_week')['daily_consumption'].std()
                    daily_std = daily_std.reindex(range(7), fill_value=0)
                    pattern_fig.add_trace(
                        go.Bar(
                            x=day_names,
                            y=daily_std.values,
                            marker_color='#e74c3c',
                            name='Variability'
                        ),
                        row=1, col=1
                    )
            if 'month' in filtered_df.columns:
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                monthly_avg = filtered_df.groupby('month')['daily_consumption'].mean()
                monthly_avg = monthly_avg.reindex(range(1, 13), fill_value=0)
                pattern_fig.add_trace(
                    go.Bar(
                        x=month_names,
                        y=monthly_avg.values,
                        marker_color='#2ecc71',
                        name='Avg by Month'
                    ),
                    row=1, col=2
                )
            pattern_fig.update_layout(
                height=400,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            pattern_fig.update_xaxes(title_text="Day of Week", row=1, col=1)
            pattern_fig.update_xaxes(title_text="Month", row=1, col=2)
            pattern_fig.update_yaxes(title_text="Average Consumption (kWh)", row=1, col=1)
            pattern_fig.update_yaxes(title_text="Average Consumption (kWh)", row=1, col=2)

            try:
                date_filtered_anomalies = anomalies
                if start_date and end_date:
                    date_filtered_anomalies = anomalies[(anomalies['date'] >= start_date) & (anomalies['date'] <= end_date)]
                total_anomalies = len(date_filtered_anomalies)
                anomaly_rate = (total_anomalies / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
                avg_confidence = date_filtered_anomalies['detection_confidence'].mean() if total_anomalies > 0 else 0

                anomaly_types = []
                if total_anomalies > 0 and 'anomaly_type' in anomalies.columns:
                    for anomaly_type in anomalies['anomaly_type']:
                        if anomaly_type and pd.notna(anomaly_type):
                            types = [t.strip() for t in str(anomaly_type).split('&')]
                            anomaly_types.extend(types)
                    if anomaly_types:
                        type_counts = pd.Series(anomaly_types).value_counts().head(3)
                        top_types = [html.Li(f"{type_name} ({count})") for type_name, count in type_counts.items()]
                    else:
                        top_types = [html.Li("No specific anomaly types identified")]
                else:
                    top_types = [html.Li("No anomalies detected")]

                stats_div = html.Div([
                    html.H4('Anomaly Statistics', style={'color': '#2c3e50', 'borderBottom': '2px solid #e74c3c', 'paddingBottom': '10px'}),
                    html.Div([
                        html.Div([
                            html.P("Total Anomalies", style={'fontWeight': 'bold', 'margin': '0'}),
                            html.H3(f"{total_anomalies}", style={'color': '#e74c3c', 'margin': '5px 0 15px 0'})
                        ], className="six columns"),
                        html.Div([
                            html.P("Anomaly Percentage", style={'fontWeight': 'bold', 'margin': '0'}),
                            html.H3(f"{anomaly_rate:.1f}%", style={'color': '#e74c3c', 'margin': '5px 0 15px 0'})
                        ], className="six columns"),
                    ], className="row"),
                    html.H5("Top Anomaly Types:", style={'marginTop': '15px', 'marginBottom': '5px'}),
                    html.Ul(top_types, style={'marginTop': '0', 'paddingLeft': '20px'})
                ])
            except Exception as e:
                print(f"Error creating stats_div: {str(e)}")
                stats_div = html.Div([html.H4("Error in Anomaly Statistics")])

            try:
                total_consumption = filtered_df['daily_consumption'].sum()
                avg_consumption = filtered_df['daily_consumption'].mean()
                max_consumption = filtered_df['daily_consumption'].max() if not filtered_df.empty else 0

                weekday_data = filtered_df[filtered_df['is_weekend'] == 0] if 'is_weekend' in filtered_df.columns else pd.DataFrame()
                weekend_data = filtered_df[filtered_df['is_weekend'] == 1] if 'is_weekend' in filtered_df.columns else pd.DataFrame()

                weekday_avg = weekday_data['daily_consumption'].mean() if not weekday_data.empty else 0
                weekend_avg = weekend_data['daily_consumption'].mean() if not weekend_data.empty else 0

                summary_div = html.Div([
                    html.H4('Consumption Summary', style={'color': '#2c3e50', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
                    html.Div([
                        html.Div([
                            html.P("Building", style={'fontWeight': 'bold', 'margin': '0'}),
                            html.H5(f"{building_name}", style={'margin': '5px 0 15px 0'})
                        ], className="twelve columns"),
                    ], className="row"),
                    html.Div([
                        html.Div([
                            html.P("Total Consumption", style={'fontWeight': 'bold', 'margin': '0'}),
                            html.H3(f"{total_consumption:.1f} kWh", style={'color': '#3498db', 'margin': '5px 0 15px 0'})
                        ], className="four columns"),
                        html.Div([
                            html.P("Average Daily", style={'fontWeight': 'bold', 'margin': '0'}),
                            html.H3(f"{avg_consumption:.1f} kWh", style={'color': '#3498db', 'margin': '5px 0 15px 0'})
                        ], className="four columns"),
                        html.Div([
                            html.P("Peak Consumption", style={'fontWeight': 'bold', 'margin': '0'}),
                            html.H3(f"{max_consumption:.1f} kWh", style={'color': '#3498db', 'margin': '5px 0 15px 0'})
                        ], className="four columns"),
                    ], className="row"),
                    html.Div([
                        html.Div([
                            html.P("Weekday Average", style={'fontWeight': 'bold', 'margin': '0'}),
                            html.H5(f"{weekday_avg:.1f} kWh", style={'margin': '5px 0'})
                        ], className="six columns"),
                        html.Div([
                            html.P("Weekend Average", style={'fontWeight': 'bold', 'margin': '0'}),
                            html.H5(f"{weekend_avg:.1f} kWh", style={'margin': '5px 0'})
                        ], className="six columns"),
                    ], className="row")
                ])
            except Exception as e:
                print(f"Error creating summary_div: {str(e)}")
                summary_div = html.Div([html.H4("Error in Consumption Summary")])

            print(f"Returning graph with {len(filtered_df)} data points, {len(anomalies)} anomalies")
            print(f"Stats_div: {stats_div is not None}, Summary_div: {summary_div is not None}")

            return main_fig, pattern_fig, stats_div, summary_div

        @self.app.callback(
            DashOutput('anomaly-details', 'children'),
            [DashInput('main-graph', 'clickData'),
             DashInput('meter-selector', 'value')]
        )
        def display_anomaly_details(click_data, meter_id):
            print(f"Click data received: {click_data}")
            print(f"Meter ID for click: {meter_id}, type: {type(meter_id)}")
            if not click_data:
                return html.Div([
                    html.H4('Anomaly Details', style={'color': '#2c3e50',
                                                        'borderBottom': '2px solid #9b59b6',
                                                        'paddingBottom': '10px'}),
                    html.P('Click on a point in the graph to view detailed anomaly information.')
                ])
            try:
                date_str = click_data['points'][0]['x']
                date = pd.to_datetime(date_str)
                meter_id = str(meter_id)
                matching_rows = self.df[(self.df['meter_id'].astype(str) == meter_id) & (self.df['date'] == date)]
                print(f"Found {len(matching_rows)} matching rows for date {date}")
                if matching_rows.empty:
                    return html.Div([
                        html.H4('Data Details', style={'color': '#2c3e50',
                                                        'borderBottom': '2px solid #9b59b6',
                                                        'paddingBottom': '10px'}),
                        html.P(f'No data found for building {meter_id} on {date_str}')
                    ])
                row = matching_rows.iloc[0]
                is_anomaly = row.get('is_anomaly', False) if 'is_anomaly' in row else False
                anomaly_score = row.get('anomaly_score', 0) if 'anomaly_score' in row else 0
                anomaly_type = row.get('anomaly_type', "N/A") if 'anomaly_type' in row and pd.notna(row['anomaly_type']) else "N/A"
                building_name = METER_TO_BUILDING.get(str(meter_id), f"Building {meter_id}")
                consumption = row['daily_consumption'] if 'daily_consumption' in row else 0
                expected = row['expected_consumption'] if 'expected_consumption' in row and pd.notna(row['expected_consumption']) else None
                formatted_date = date.strftime('%A, %B %d, %Y')
                if is_anomaly:
                    pct_diff = ((consumption - expected) / expected * 100) if expected is not None and expected != 0 else 0
                    diff_direction = 'higher' if pct_diff > 0 else 'lower'
                    analyzer = EnhancedSmartMeterAnalytics()
                    explanation_text = analyzer.generate_anomaly_explanation(row)
                    sections = {}
                    current_section = "Details"
                    sections[current_section] = []
                    for line in explanation_text.split('\n'):
                        if ':' in line and line.split(':')[0].isupper() and line.split(':')[0].rstrip(':').strip():
                            current_section = line.split(':')[0].strip()
                            sections[current_section] = []
                        else:
                            sections[current_section].append(line)
                    explanation_divs = []
                    for section, lines in sections.items():
                        if section in ["POTENTIAL CAUSES", "RECOMMENDED ACTIONS", "SPECIFIC ANOMALIES DETECTED"]:
                            explanation_divs.append(
                                html.Div([
                                    html.H5(f"{section}:", style={'marginTop': '15px', 'fontWeight': 'bold'}),
                                    html.Ul([html.Li(line.strip('• ')) for line in lines if line.strip()], style={'paddingLeft': '20px'})
                                ], className="twelve columns")
                            )
                    content = html.Div([
                        html.H4('Anomaly Details', style={'color': '#2c3e50', 'borderBottom': '2px solid #e74c3c', 'paddingBottom': '10px'}),
                        html.Div([
                            html.Div([
                                html.H5(f"{building_name}", style={'marginBottom': '5px', 'color': '#c0392b'}),
                                html.P(f"{formatted_date}", style={'fontStyle': 'italic', 'marginBottom': '15px'})
                            ], className="twelve columns")
                        ], className="row"),
                        html.Div([
                            html.Div([
                                html.P("Actual Consumption", style={'fontWeight': 'bold', 'margin': '0'}),
                                html.H3(f"{consumption:.1f} kWh", style={'color': '#e74c3c'})
                            ], className="four columns"),
                            html.Div([
                                html.P("Expected Consumption", style={'fontWeight': 'bold', 'margin': '0'}),
                                html.H3(f"{expected:.1f} kWh", style={'color': '#27ae60'}) if expected is not None else html.H3("Unknown", style={'color': '#7f8c8d'})
                            ], className="four columns"),
                            html.Div([
                                html.P("Difference", style={'fontWeight': 'bold', 'margin': '0'}),
                                html.H3(f"{abs(pct_diff):.1f}% {diff_direction}", style={'color': '#e74c3c' if pct_diff > 0 else '#2980b9'})
                            ], className="four columns") if expected is not None else None,
                        ], className="row"),
                        html.Div([
                            html.Div([
                                html.P("Anomaly Score", style={'fontWeight': 'bold', 'margin': '0'}),
                                html.Div([
                                    html.Div(style={'height': '10px', 'width': f'{anomaly_score*100}%', 'backgroundColor': '#e74c3c', 'borderRadius': '5px'}),
                                    html.Span(f"{anomaly_score:.2f}", style={'margin'
                                    'eft': '10px'})
                                ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '5px'})
                            ], className="six columns"),
                            html.Div([
                                html.P("Anomaly Type", style={'fontWeight': 'bold', 'margin': '0'}),
                                html.H5(f"{anomaly_type}", style={'color': '#e67e22', 'marginTop': '5px'})
                            ], className="six columns"),
                        ], className="row"),
                        html.Div(explanation_divs, className="row")
                    ])
                else:
                    content = html.Div([
                        html.H4('Consumption Details', style={'color': '#2c3e50', 'borderBottom': '2px solid #2980b9', 'paddingBottom': '10px'}),
                        html.Div([
                            html.Div([
                                html.H5(f"{building_name}", style={'marginBottom': '5px'}),
                                html.P(f"{formatted_date}", style={'fontStyle': 'italic', 'marginBottom': '15px'})
                            ], className="twelve columns")
                        ], className="row"),
                        html.Div([
                            html.Div([
                                html.P("Daily Consumption", style={'fontWeight': 'bold', 'margin': '0'}),
                                html.H3(f"{consumption:.1f} kWh", style={'color': '#2980b9'})
                            ], className="six columns"),
                            html.Div([
                                html.P("Status", style={'fontWeight': 'bold', 'margin': '0'}),
                                html.H3("Normal", style={'color': '#27ae60'})
                            ], className="six columns"),
                        ], className="row"),
                        html.P("This consumption reading falls within normal parameters.", style={'marginTop': '15px'})
                    ])
                return content
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                return html.Div([
                    html.H4('Error in Anomaly Details', style={'color': '#c0392b', 'borderBottom': '2px solid #9b59b6', 'paddingBottom': '10px'}),
                    html.P(f'Error type: {type(e).__name__}'),
                    html.P(f'Error message: {str(e)}'),
                    html.Details([html.Summary('Show technical details'),
                                  html.Pre(error_trace, style={'backgroundColor': '#f8f9fa', 'padding': '10px', 'borderRadius': '5px', 'whiteSpace': 'pre-wrap'})])
                ])

    def run(self, port=5000):
        self.app.run(port=port, host='127.0.0.1')

if __name__ == '__main__':
    dashboard = SmartMeterDashboard()
    dashboard.run()
