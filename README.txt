# 🚨 GTFS Real-Time Transit Disruption Detection Dashboard

A Streamlit-based dashboard for detecting and classifying traffic disruptions in real-time using GTFS-RT feeds and machine learning.

![Dashboard Preview](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![ML Models](https://img.shields.io/badge/ML-XGBoost%7CRandomForest%7CNeuralNet-green.svg)

---

## 🌟 Features

- **Real-Time Data**: Fetches live data from GTFS-RT feeds (OVAPI - Netherlands)
- **ML Predictions**: Uses XGBoost, Random Forest, Neural Networks for disruption detection
- **Interactive Maps**: Folium-based geographic visualization
- **Dynamic Charts**: Plotly charts for metrics analysis
- **Alert Feed**: Real-time alerts for severe disruptions
- **Filtering**: Filter by severity, route, and confidence level
- **Auto-Refresh**: Configurable automatic data refresh

---

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMLIT DASHBOARD                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Sidebar   │  │  Route Table │  │    Charts    │          │
│  │  Controls   │  │   & Filters  │  │ (Plotly)     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ML PREDICTION LAYER                           │
│  ┌──────────────────┐    ┌──────────────────────────────────┐   │
│  │  FastAPI Backend │    │  Simulated Predictions          │   │
│  │  (Optional)      │    │  (Fallback)                     │   │
│  └──────────────────┘    └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION LAYER                         │
│  ┌────────────────────┐  ┌────────────────────┐                │
│  │  gtfs_loader.py   │  │  gtfs_collector.py│                │
│  │  - Vehicle Pos    │  │  - Feature Engine │                │
│  │  - Trip Updates   │  │  - Data Merge    │                │
│  │  - Alerts         │  │                   │                │
│  └────────────────────┘  └────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL DATA SOURCES                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  OVAPI (Netherlands GTFS-RT)                          │   │
│  │  • vehiclePositions.pb                                │   │
│  │  • tripUpdates.pb                                    │   │
│  │  • alerts.pb                                         │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/gtfs-dashboard.git
cd gtfs-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Locally

```bash
# Start the dashboard
streamlit run realtime_early_warning_dashboard.py
```

### 3. Deploy to Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy!

---

## 📁 Project Structure

```
├── realtime_early_warning_dashboard.py   # Main dashboard app
├── fast_api.py                           # FastAPI backend (optional)
├── utils/
│   ├── gtfs_loader.py                    # GTFS-RT data loader
│   ├── gtfs_collector.py                 # Data collection & features
│   └── data_fetcher.py                   # Data fetching utilities
├── my_api/
│   └── models/                           # ML model files
│       ├── BEST_binary_model_*.pkl
│       ├── XGBoost_multi_model_*.pkl
│       └── scaler_*.pkl
├── .streamlit/
│   └── config.toml                       # Streamlit configuration
├── docs/
│   ├── ARCHITECTURE.md                   # System architecture
│   ├── DEPLOYMENT.md                    # Deployment guide
│   └── TRANSIT_AGENCIES.md              # GTFS feed references
└── requirements.txt                      # Python dependencies
```

---

## 🔧 Configuration

### GTFS Feed URLs

Configure in `.streamlit/secrets.toml`:

```toml
[api_keys]
GTFS_VEHICLE_URL = "http://gtfs.ovapi.nl/nl/vehiclePositions.pb"
GTFS_TRIP_UPDATE_URL = "http://gtfs.ovapi.nl/nl/tripUpdates.pb"
GTFS_ALERTS_URL = "http://gtfs.ovapi.nl/nl/alerts.pb"
```

### Model Settings

```python
MODEL_DIR = "my_api/models"
SCALER_PATH = "my_api/models/scaler_20260305_061409.pkl"
```

---

## 🤖 ML Models

The dashboard supports multiple ML models:

| Model | Type | Use Case |
|-------|------|----------|
| XGBoost | Gradient Boosting | Best overall performance |
| Random Forest | Ensemble | Robust baseline |
| Neural Network | Deep Learning | Complex patterns |
| BEST | Auto-selected | Best model for task |

---

## 📈 Features

### Prediction Features (12 features)

1. `speed_mean` - Average vehicle speed
2. `speed_std` - Speed variation
3. `delay_mean_5m` - Delay in last 5 minutes
4. `delay_mean_15m` - Delay in last 15 minutes
5. `delay_mean_30m` - Delay in last 30 minutes
6. `bunching_index` - Bus bunching indicator
7. `on_time_pct` - On-time performance percentage
8. `headway_variance` - Headway variation
9. `alert_nlp_score` - NLP-based alert score
10. `alert_count` - Number of active alerts
11. `fleet_utilization` - Fleet utilization rate
12. `speed_drop_ratio` - Speed drop from normal

### Severity Classes

- **NORMAL** (Class 0) - No disruption
- **MINOR** (Class 1) - Minor delays
- **MODERATE** (Class 2) - Moderate disruption
- **SEVERE** (Class 3) - Severe disruption

---

## 🔨 Development

### Adding New Features

1. Add feature engineering in `utils/gtfs_collector.py`
2. Update `DEFAULT_FEATURE_NAMES` in dashboard
3. Retrain models with new features

### Adding New Models

1. Train model and save to `my_api/models/`
2. Update `fast_api.py` model loading
3. Add model option in dashboard sidebar

---

## 📝 License

MIT License

---

## 🙏 Acknowledgments

- [OVAPI](https://gtfs.ovapi.nl/) - Netherlands GTFS data
- [Streamlit](https://streamlit.io/) - Dashboard framework
- [Plotly](https://plotly.com/) - Charts
- [Folium](https://python-visualization.github.io/folium/) - Maps

---

## 📧 Contact

For questions or issues, please open an issue on GitHub.
