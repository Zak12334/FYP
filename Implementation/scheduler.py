import schedule
import time
from webapp.webapp import SmartMeterDashboard

def run_analytics():
    dashboard = SmartMeterDashboard()
    dashboard.data_loader.get_data()
    dashboard.analytics.detect_anomalies(dashboard.df)
    print(f"Daily analytics completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Schedule to run at 7:00 AM
schedule.every().day.at("07:00").do(run_analytics)

if __name__ == "__main__":
    print(f"Scheduler started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    # Run once immediately at startup
    run_analytics()
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute