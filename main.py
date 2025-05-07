import warnings
warnings.filterwarnings("ignore")
from webapp.webapp import SmartMeterDashboard
def main():
    
    # Start dashboard (main thread)
    dashboard = SmartMeterDashboard()
    dashboard.run(port=5000)

if __name__ == '__main__':
    main()