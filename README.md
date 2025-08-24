# Flight Path Optimization with Wind Data

A sophisticated flight route optimization system that leverages meteorological wind data to calculate optimal flight paths using both heuristic (A*) and gradient-based optimization algorithms. The system processes real flight data, applies meteorological filtering, and provides interactive visualizations comparing actual versus optimized routes.

**Developed as part of ESGI 2025 in collaboration with NATS (National Air Traffic Services)**

## 🚀 Features

- **Complete Data Processing Pipeline**
  - Automatic flight route extraction and filtering
  - Meteorological boundary validation
  - Pre-computed wind grid generation
- **Dual Optimization Algorithms**
  - A* Heuristic Search for fast pathfinding
  - Temporal Gradient Descent for wind-optimized routing
- **Real-time Wind Grid Integration**
  - Pre-computed wind grids for fast lookups
  - Spatial and temporal interpolation
  - Multi-altitude flight level support (FL050-FL340)
- **Interactive Web Visualization**
  - Real-time flight path visualization on interactive maps
  - Wind vector field display
  - Comparative analysis between actual vs optimized routes
- **Comprehensive Flight Analysis**
  - Flight efficiency metrics
  - Wind benefit calculations
  - Distance and time savings analysis
  - Great circle vs actual path comparison

## 📋 Prerequisites

- Python 3.8+
- 8GB+ RAM recommended for wind grid processing
- Modern web browser for visualization
- ~5GB disk space for data and processed files

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/esgi_nats.git
cd esgi_nats
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
```txt
flask>=2.0.0
pandas>=1.3.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
scikit-learn
pyproj
requests
beautifulsoup4
```

3. Set up the project structure:
```
esgi_nats/                              # Main project directory (set as working directory)
├── data/
│   ├── ECCC/
│   │   └── 2025_JUN_par_met.parquet.gzip  # Meteorological data
│   └── *.parquet                       # Raw flight data files
├── results/
│   ├── processed_flights_*/            # Route-filtered flights
│   ├── filtered_flights_met_bounds_*/  # Met-filtered flights
│   └── wind_grids_cache.pkl           # Pre-computed wind grids
├── src/
│   ├── preprocessing/
│   │   ├── 1_flight_split.py          # Flight route splitting script
│   │   ├── 2_filter_flights.py        # Meteorological filtering script
│   │   └── 3_flight_met_data_optimization.py  # Wind grid pre-calculation
│   ├── app.py                         # Flask web application
│   ├── flight_efficiency_analysis.py  # Flight analysis utilities
│   ├── flight_optimization_gradient.py # Gradient-based optimizer
│   ├── utils.py                       # Helper functions
│   └── wind_service.py                # Wind grid service module
├── templates/
│   └── index.html                     # Web interface template
└── README.md
```

**Important:** Set `esgi_nats` as your working directory before running any scripts. All paths in the code are relative to this root directory.

## 📊 Data Preparation

**⚠️ Note:** The actual flight and meteorological data are proprietary and not included in this repository due to NDA restrictions with NATS. The following instructions assume you have obtained appropriate data access.

The data preparation pipeline consists of three sequential steps:

### Step 1: Flight Route Splitting
**Script:** `src/preprocessing/1_flight_split.py`

This script filters raw flight data by specific routes.

**Configuration:**
```python
ORIGIN_AIRPORT = "CYYZ"        # Origin airport code (Toronto)
DESTINATION_AIRPORT = "LFPG"   # Destination airport code (Paris)
```

**Input:** Raw flight parquet files from `data/` folder

**Output:** `results/processed_flights_CYYZ_LFPG/`

**Usage:**
```bash
python src/preprocessing/1_flight_split.py
```

### Step 2: Meteorological Boundary Filtering
**Script:** `src/preprocessing/2_filter_flights.py`

Applies meteorological boundary filtering to ensure flights are within the available weather data coverage area.

**Input:** `results/processed_flights_CYYZ_LFPG/`

**Output:** `results/filtered_flights_met_bounds_CYYZ_LFPG/`

**Usage:**
```bash
python src/preprocessing/2_filter_flights.py
```

The script filters flights to stay within meteorological data bounds:
- Latitude: 41.25°N to 68.75°N  
- Longitude: 87.50°W to 12.50°W

### Step 3: Wind Grid Pre-calculation
**Script:** `src/preprocessing/3_flight_met_data_optimization.py`

Generates pre-computed wind grids for fast runtime lookups.

**Input:** `data/ECCC/2025_JUN_par_met.parquet.gzip`

**Output:** `results/wind_grids_cache.pkl`

**Usage:**
```bash
python src/preprocessing/3_flight_met_data_optimization.py
```

This creates a cached wind grid with:
- Spatial interpolation at 0.1° resolution
- Temporal coverage for all available timestamps
- Flight levels: FL050, FL100, FL180, FL240, FL300, FL340
- Typical file size: 100-500 MB depending on coverage

### Complete Data Pipeline:
```bash
# Ensure you're in the esgi_nats root directory
cd /path/to/esgi_nats

# Run all preprocessing steps in sequence
python src/preprocessing/1_flight_split.py
python src/preprocessing/2_filter_flights.py  
python src/preprocessing/3_flight_met_data_optimization.py

# Then start the web application
python src/app.py
```

### Data Requirements:
1. **Raw Flight Data**: Place `.parquet` files in `data/` folder
2. **Meteorological Data**: Ensure `data/ECCC/2025_JUN_par_met.parquet.gzip` exists
3. **Storage**: ~2-5 GB for processed data and wind grids

## 🚦 Usage

### Quick Start - Complete Pipeline

**Important:** Ensure you're in the `esgi_nats` root directory before running any commands.

```bash
# Navigate to project root
cd /path/to/esgi_nats

# Step 1: Filter flights by route (CYYZ to LFPG)
python src/preprocessing/1_flight_split.py

# Step 2: Apply meteorological boundary filtering
python src/preprocessing/2_filter_flights.py

# Step 3: Generate wind grid cache
python src/preprocessing/3_flight_met_data_optimization.py

# Step 4: Start the web application
python src/app.py
```

Access the application at: `http://localhost:5000`

### Running the Web Application

2. Start the Flask server:
```bash
python src/app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Select a flight from the dropdown menu to:
   - View actual flight path
   - See optimized routes (both A* and gradient-based)
   - Analyze wind patterns and efficiency metrics
   - Compare distance and time savings

### Using the Optimization Modules Directly

#### Gradient-Based Optimization
```python
import sys
sys.path.append('src')  # Add src to path if running from esgi_nats root

from flight_optimization_gradient import main_fixed_temporal_gradient_search
from wind_service import WindGridOptimizer

# Load wind grids
wind_optimizer = WindGridOptimizer()
wind_optimizer.load_grids('./results/wind_grids_cache.pkl')

# Load flight data
flight_data = pd.read_parquet('results/filtered_flights_met_bounds_CYYZ_LFPG/flight_example.parquet')

# Run optimization
optimal_path, analysis = main_fixed_temporal_gradient_search(
    flight_data, 
    is_visualization=True
)
```

#### A* Heuristic Optimization
```python
import sys
sys.path.append('src')  # Add src to path if running from esgi_nats root

from app import optimize_flight_route
from wind_service import load_wind_grid_from_file

# Load wind grid
wind_grid = load_wind_grid_from_file('./results/wind_grids_cache.pkl')

# Define start and end points
start_lat, start_lon, start_alt = 43.68, -79.63, 35000
end_lat, end_lon, end_alt = 49.01, 2.55, 35000
start_time = pd.Timestamp.now()

# Optimize route
optimized_route = optimize_flight_route(
    start_lat, start_lon, start_alt, start_time,
    end_lat, end_lon, end_alt, wind_grid
)
```

## 🔧 Configuration

Key parameters can be adjusted in the respective files:

### Wind Grid Settings (`src/wind_service.py`)
- `grid_resolution`: Spatial resolution in degrees (default: 0.1°)
- `available_flight_levels`: Supported altitudes

### Optimization Parameters

**Gradient Descent (`src/flight_optimization_gradient.py`):**
- `alpha`: Gradient weight (default: 0.1)
- `beta`: Goal alignment weight (default: 0.5)
- `gamma`: Temporal weight (default: 0.02)

**A* Heuristic (`src/app.py`):**
- `LAT_STEP`: Latitude step size (default: 0.1°)
- `LON_STEP`: Longitude step size (default: 0.5°)
- `ALT_STEP`: Altitude step (default: 2000 ft)
- `TAS_KNOTS`: True airspeed (default: 450 knots)


## 📊 Output Analysis

The system provides comprehensive analysis including:

1. **Route Metrics**
   - Total distance (km/nm)
   - Flight time (hours)
   - Average ground speed
   - Fuel efficiency estimates

2. **Wind Analysis**
   - Average wind benefit/penalty
   - Headwind/tailwind components
   - Optimal altitude recommendations

3. **Comparison Metrics**
   - Actual vs optimized distance
   - Time savings
   - Efficiency improvements
   - Deviation from great circle


## 🙏 Acknowledgments

- **[NATS (National Air Traffic Services)](https://www.nats.aero/about-us/)** - We extend our sincere gratitude to NATS for providing access to real flight data and meteorological data that made this research possible. NATS is the UK's leading air navigation service provider, handling 2.5 million flights and 250 million passengers in UK airspace annually.

- **[ESGI 2025 (European Study Group with Industry)](https://ecmiindmath.org/esgi/)** - Special thanks to ESGI 2025 for facilitating this collaboration and providing the opportunity to work with NATS on this challenging real-world optimization problem. The European Study Groups with Industry bring together academics and industrial partners to work on problems of mutual interest.

### ⚠️ Data Access Notice

**Important:** The flight and meteorological datasets used in this project are proprietary and provided under a Non-Disclosure Agreement (NDA) with NATS. These datasets:
- Are NOT included in this repository
- Cannot be shared or distributed
- Are for authorized research use only
- Require separate licensing agreements with NATS for access

Researchers interested in accessing similar data should contact NATS directly through their official channels:
- Aditya Gaur - Aditya.GAUR@nats.co.uk
- Guillermo Ledo López - Guillermo.LEDOLOPEZ@nats.co.uk

## 🐛 Known Issues

- Wind data interpolation may be less accurate at grid boundaries
- Optimization may take longer for trans-continental flights
- Browser memory usage can be high with many wind vectors displayed



