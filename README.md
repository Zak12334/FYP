Smart Meter Anomaly Detection
A machine learning system that identifies unusual energy consumption patterns across Cork County Council buildings with 93% accuracy.
Overview
This project develops an anomaly detection system for Cork County Council's smart meters to identify unusual consumption patterns. Using LSTM autoencoders and isolation forests, the system detects anomalies across different building types to optimize energy usage, reduce costs, and support sustainability initiatives.
Features

Multi-Model Anomaly Detection: Combines LSTM Autoencoder, Isolation Forest, and statistical Z-score analysis for high-accuracy detection with minimal false positives
Building-Specific Thresholds: Customized detection parameters for different building types (libraries, offices, fire stations)
Interactive Dashboard: Real-time visualization of consumption patterns and anomaly details
Anomaly Classification: Categorizes issues as Low, Medium, High, or Critical for efficient prioritization
Explainable Results: Provides detailed anomaly explanations with potential causes and recommended actions
Automated Monitoring: Continuous analysis with results logged to database

Technology Stack

Backend: Python, TensorFlow, scikit-learn
Data Storage: MySQL
Data Processing: pandas, NumPy
Frontend/Visualization: Dash, Plotly
Deployment: Flask, Python scheduler
