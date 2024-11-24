import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output 
import plotly.express as px
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    data['made_at'] = pd.to_datetime(data['made_at'], dayfirst=True)
    data = data.sort_values('made_at')
    data = data.drop_duplicates().fillna(0)
    
    # Calculate baseline features
    data['hour'] = data['made_at'].dt.hour
    data['day_of_week'] = data['made_at'].dt.dayofweek
    
    # Calculate consumption changes
    data['consumption_change'] = data['consumption'].diff()
    data['consecutive_zeros'] = (data['consumption'] == 0).astype(int).groupby(
        (data['consumption'] != 0).cumsum()).cumsum()
    
    return data

def detect_anomalies(data, window_size=72):
    # Focus on non-zero consumption periods
    avg_consumption = data[data['consumption'] > 0]['consumption'].mean()
    std_consumption = data[data['consumption'] > 0]['consumption'].std()
    
    # Calculate rolling statistics for visualization
    data['rolling_mean'] = data['consumption'].rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    data['rolling_std'] = data['consumption'].rolling(window=window_size, center=True).std().fillna(method='bfill').fillna(method='ffill')
    
    # Mark anomalies based on extreme deviations and pattern
    data['final_anomaly'] = (
        (data['consumption'] > avg_consumption + 3 * std_consumption) & 
        (data['consecutive_zeros'] < 3)  # Ignore initial spikes after zero periods
    ).astype(int)
    
    return data

def create_dashboard(data):
    app = Dash(__name__)
    
    app.layout = html.Div([
        html.H1("Gas Consumption Monitoring System"),
        html.Div([
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date=data['made_at'].min().date(),
                end_date=data['made_at'].max().date(),
                display_format='YYYY-MM-DD'
            ),
            dcc.Checklist(
                id='display-options',
                options=[
                    {'label': 'Show Normal Usage Pattern', 'value': 'pattern'},
                    {'label': 'Show Thresholds', 'value': 'thresholds'}
                ],
                value=['thresholds']
            ),
        ]),
        dcc.Graph(id='consumption-graph')
    ])

    @app.callback(
        Output('consumption-graph', 'figure'),
        [Input('date-picker-range', 'start_date'),
         Input('date-picker-range', 'end_date'),
         Input('display-options', 'value')]
    )
    def update_graph(start_date, end_date, display_options):
        filtered_data = data[(data['made_at'].dt.date >= pd.to_datetime(start_date).date()) &
                           (data['made_at'].dt.date <= pd.to_datetime(end_date).date())]
        
        fig = px.line(filtered_data, x='made_at', y='consumption', 
                     title='Gas Consumption Over Time')
        
        # Show normal usage patterns if selected
        if 'pattern' in display_options:
            normal_usage = filtered_data[
                (filtered_data['consumption'] > 0) & 
                (filtered_data['consumption'] <= filtered_data['rolling_mean'] + 2 * filtered_data['rolling_std'])
            ]
            fig.add_scatter(x=normal_usage['made_at'], y=normal_usage['consumption'],
                          mode='markers', name='Normal Usage', 
                          marker=dict(color='green', size=8))
        
        # Show anomalies
        anomalies = filtered_data[filtered_data['final_anomaly'] == 1]
        fig.add_scatter(x=anomalies['made_at'], y=anomalies['consumption'],
                       mode='markers', name='Anomalies', 
                       marker=dict(color='red', size=10))
        
        if 'thresholds' in display_options:
            fig.add_scatter(x=filtered_data['made_at'], 
                          y=filtered_data['rolling_mean'] + 3 * filtered_data['rolling_std'],
                          mode='lines', name='Upper Threshold', 
                          line=dict(dash='dash', color='rgba(255,0,0,0.3)'))
        
        fig.update_layout(
            height=700,
            showlegend=True,
            hovermode='closest',
            yaxis_title='Consumption',
            xaxis_title='Time'
        )
        return fig

    return app

if __name__ == '__main__':
    # Load data
    data = pd.read_csv('smart_eter_raw_data.csv')
    
    # Process and analyze data
    processed_data = preprocess_data(data)
    analyzed_data = detect_anomalies(processed_data)
    
    # Create and run dashboard
    app = create_dashboard(analyzed_data)
    app.run_server(debug=False, port=8050)