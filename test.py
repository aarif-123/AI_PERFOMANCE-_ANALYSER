import pandas as pd
import psutil
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.ensemble import IsolationForest
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Configuration
class Config:
    DATA_FILE = "system_monitor.csv"
    LOG_FILE = "system_monitor.log"
    UPDATE_INTERVAL = 5000  # milliseconds
    PREDICTION_WINDOW = 10  # seconds
    FORECAST_POINTS = 3
    ANOMALY_CONTAMINATION = 0.05
    WARNING_THRESHOLDS = {
        'cpu_usage': 80,
        'memory_usage': 85
    }
    SUSPICIOUS_PROCESS_THRESHOLDS = {
        'cpu_percent': 50.0,
        'memory_percent': 20.0
    }

# Setup logging
logging.basicConfig(
    filename=Config.LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SystemMonitor:
    def __init__(self):
        self.df = self._initialize_dataframe()
        self.anomaly_model = IsolationForest(
            contamination=Config.ANOMALY_CONTAMINATION,
            random_state=42
        )
        self.forecast_model = make_pipeline(
            PolynomialFeatures(degree=2),
            LinearRegression()
        )
        # Track alert states for debouncing
        self.alert_states = {
            'cpu_usage': False,
            'memory_usage': False,
            'predicted_cpu_usage': False,
            'anomaly': False  # Added for anomaly debouncing
        }
        
    def _initialize_dataframe(self) -> pd.DataFrame:
        columns = ['timestamp', 'cpu_usage', 'memory_usage', 'predicted_cpu_usage']
        if os.path.exists(Config.DATA_FILE):
            try:
                df = pd.read_csv(Config.DATA_FILE)
                for col in columns:
                    if col not in df.columns:
                        df[col] = 0.0 if col != 'timestamp' else pd.NaT
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df[columns]
            except Exception as e:
                logging.error(f"Error loading data: {e}")
                return pd.DataFrame(columns=columns)
        return pd.DataFrame(columns=columns)
    
    def collect_system_data(self) -> dict:
        return {
            "timestamp": datetime.now(),
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "predicted_cpu_usage": 0.0
        }
    
    def check_suspicious_processes(self) -> list:
        suspicious_processes = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    proc_info = proc.as_dict(attrs=['pid', 'name', 'cpu_percent', 'memory_percent'])
                    # Skip System Idle Process (PID 0)
                    if proc_info['pid'] == 0 or proc_info['name'].lower() == 'system idle process':
                        continue
                    cpu_percent = proc_info['cpu_percent']
                    memory_percent = proc_info['memory_percent']

                    if (cpu_percent > Config.SUSPICIOUS_PROCESS_THRESHOLDS['cpu_percent'] or
                        memory_percent > Config.SUSPICIOUS_PROCESS_THRESHOLDS['memory_percent']):
                        suspicious_processes.append({
                            'pid': proc_info['pid'],
                            'name': proc_info['name'],
                            'cpu_percent': round(cpu_percent, 2),  # Round for readability
                            'memory_percent': round(memory_percent, 2)  # Round for readability
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logging.error(f"Error checking processes: {e}")
        return suspicious_processes
    
    def update_data(self, new_data: dict) -> None:
        try:
            # Check for threshold violations and log alerts with debouncing
            if new_data['cpu_usage'] > Config.WARNING_THRESHOLDS['cpu_usage']:
                if not self.alert_states['cpu_usage']:
                    logging.warning(f"High CPU usage alert: {new_data['cpu_usage']:.2f}% exceeds threshold of {Config.WARNING_THRESHOLDS['cpu_usage']}%")
                    self.alert_states['cpu_usage'] = True
            else:
                if self.alert_states['cpu_usage']:
                    logging.info(f"CPU usage returned to normal: {new_data['cpu_usage']:.2f}%")
                    self.alert_states['cpu_usage'] = False

            if new_data['memory_usage'] > Config.WARNING_THRESHOLDS['memory_usage']:
                if not self.alert_states['memory_usage']:
                    logging.warning(f"High memory usage alert: {new_data['memory_usage']:.2f}% exceeds threshold of {Config.WARNING_THRESHOLDS['memory_usage']}%")
                    self.alert_states['memory_usage'] = True
            else:
                if self.alert_states['memory_usage']:
                    logging.info(f"Memory usage returned to normal: {new_data['memory_usage']:.2f}%")
                    self.alert_states['memory_usage'] = False

            future_times, future_vals = self.predict_cpu_usage()
            if future_vals is not None:
                new_data['predicted_cpu_usage'] = future_vals[-1]
                if new_data['predicted_cpu_usage'] > Config.WARNING_THRESHOLDS['cpu_usage']:
                    if not self.alert_states['predicted_cpu_usage']:
                        logging.warning(f"Predicted CPU usage alert: {new_data['predicted_cpu_usage']:.2f}% (in {Config.PREDICTION_WINDOW}s) exceeds threshold of {Config.WARNING_THRESHOLDS['cpu_usage']}%")
                        self.alert_states['predicted_cpu_usage'] = True
                else:
                    if self.alert_states['predicted_cpu_usage']:
                        logging.info(f"Predicted CPU usage returned to normal: {new_data['predicted_cpu_usage']:.2f}%")
                        self.alert_states['predicted_cpu_usage'] = False
            else:
                new_data['predicted_cpu_usage'] = 0.0

            new_df = pd.DataFrame([new_data])
            required_columns = ['timestamp', 'cpu_usage', 'memory_usage', 'predicted_cpu_usage']
            new_df = new_df.reindex(columns=required_columns)
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            self.df.to_csv(
                Config.DATA_FILE,
                index=False,
                float_format='%.2f',
                mode='w',
                encoding='utf-8'
            )
            if len(self.df) > 1000:
                self.df = self.df.tail(1000)

            # Check for anomalies and suspicious processes with debouncing
            anomalies = self.detect_anomalies()
            if len(anomalies) > 0 and -1 in anomalies:
                if not self.alert_states['anomaly']:
                    # Log system metrics for context
                    logging.warning(
                        f"Anomaly detected in system metrics - "
                        f"CPU Usage: {new_data['cpu_usage']:.2f}%, "
                        f"Memory Usage: {new_data['memory_usage']:.2f}%"
                    )
                    suspicious_procs = self.check_suspicious_processes()
                    if suspicious_procs:
                        for proc in suspicious_procs:
                            logging.warning(
                                f"Suspicious process detected: "
                                f"PID={proc['pid']:<6} | "
                                f"Name={proc['name']:<20} | "
                                f"CPU={proc['cpu_percent']:<6.2f}% | "
                                f"Memory={proc['memory_percent']:<6.2f}%"
                            )
                    else:
                        logging.warning("No suspicious processes identified.")
                    self.alert_states['anomaly'] = True
            else:
                if self.alert_states['anomaly']:
                    logging.info("System metrics returned to normal - No anomalies detected.")
                    self.alert_states['anomaly'] = False

        except Exception as e:
            logging.error(f"Error updating data: {e}")

    def predict_cpu_usage(self) -> tuple:
        MIN_POINTS = 5
        WINDOW_SIZE = 10

        if len(self.df) < MIN_POINTS:
            return None, None

        cpu_data = self.df['cpu_usage'].tail(WINDOW_SIZE).values
        
        if len(cpu_data) >= WINDOW_SIZE:
            smoothed_data = pd.Series(cpu_data).rolling(window=WINDOW_SIZE, min_periods=1).mean().values
        else:
            smoothed_data = cpu_data

        X = np.arange(len(smoothed_data)).reshape(-1, 1)
        y = smoothed_data

        try:
            linear_model = LinearRegression()
            linear_model.fit(X, y)
        except Exception as e:
            logging.error(f"Prediction model training failed: {e}")
            return None, None

        last_time = self.df['timestamp'].iloc[-1]
        future_times = [
            last_time + timedelta(seconds=(i+1)*Config.PREDICTION_WINDOW/Config.FORECAST_POINTS)
            for i in range(Config.FORECAST_POINTS)
        ]

        future_X = np.arange(len(smoothed_data), len(smoothed_data) + Config.FORECAST_POINTS).reshape(-1, 1)
        future_y = linear_model.predict(future_X)

        future_y = np.clip(future_y, 0, 100)

        return future_times, future_y
    
    def detect_anomalies(self) -> list:
        if len(self.df) < 10:
            return []
        features = self.df[['cpu_usage', 'memory_usage']]
        return self.anomaly_model.fit_predict(features)

class MonitoringVisualization:
    def __init__(self, monitor: SystemMonitor):
        self.monitor = monitor
        self.fig, self.axes = plt.subplots(2, 1, figsize=(12, 10))
        plt.subplots_adjust(hspace=0.4)
        
    def update_plot(self, frame):
        try:
            new_data = self.monitor.collect_system_data()
            self.monitor.update_data(new_data)
            df = self.monitor.df.tail(20)
            for ax in self.axes:
                ax.clear()
            self._plot_cpu_with_prediction(ax=self.axes[0], df=df)
            self._plot_system_metrics(ax=self.axes[1], df=df)
        except Exception as e:
            logging.error(f"Error in plot update: {e}")

    def _plot_cpu_with_prediction(self, ax, df):
        ax.plot(df['timestamp'], df['cpu_usage'], 
                label='Actual CPU Usage', color='blue', marker='o')
        future_times, future_vals = self.monitor.predict_cpu_usage()
        if future_times is not None and future_vals is not None:
            ax.plot(future_times, future_vals, 
                    label='Predicted CPU Usage', color='red', linestyle='--', marker='x')
            ax.scatter(future_times[-1], future_vals[-1], 
                      color='red', s=100, zorder=5,
                      label=f'Prediction in {Config.PREDICTION_WINDOW}s: {future_vals[-1]:.1f}%')
        ax.set_title('CPU Usage with Prediction')
        ax.set_ylabel('Usage (%)')
        ax.set_ylim(0, 100)
        ax.axhline(y=Config.WARNING_THRESHOLDS['cpu_usage'], 
                  color='orange', linestyle='--', 
                  label='Warning Threshold')
        ax.legend()
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_system_metrics(self, ax, df):
        metrics = {
            'cpu_usage': 'blue',
            'memory_usage': 'green'
        }
        for metric, color in metrics.items():
            valid_data = df[pd.to_numeric(df[metric], errors='coerce').between(0, 100, inclusive='both')]
            if not valid_data.empty:
                ax.plot(valid_data['timestamp'], valid_data[metric], 
                        label=metric, color=color)
            else:
                logging.warning(f"No valid data to plot for {metric}")
        
        anomalies = self.monitor.detect_anomalies()
        if len(anomalies) == len(df):
            anomaly_indices = np.where(anomalies == -1)[0]
            for i in anomaly_indices:
                if i < len(df):
                    ax.scatter(df['timestamp'].iloc[i], df['cpu_usage'].iloc[i],
                              color='red', marker='x', s=100, 
                              label='Anomaly' if i == anomaly_indices[0] else "")
        ax.set_title('System Metrics Overview')
        ax.set_ylabel('Usage (%)')
        ax.set_ylim(0, 100)
        for metric in ['cpu_usage', 'memory_usage']:
            ax.axhline(y=Config.WARNING_THRESHOLDS[metric], color='orange', linestyle='--',
                      label=f'{metric} threshold' if metric == 'cpu_usage' else "")
        ax.legend()
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

def main():
    try:
        monitor = SystemMonitor()
        visualization = MonitoringVisualization(monitor)
        ani = FuncAnimation(
            visualization.fig,
            visualization.update_plot,
            interval=Config.UPDATE_INTERVAL,
            cache_frame_data=False
        )
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Error in main: {e}")
        print(f"An error occurred. Check {Config.LOG_FILE} for details.")

if __name__ == "__main__":
    main()