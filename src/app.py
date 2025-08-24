#!/usr/bin/env python3
"""
Flight-MET Data Interactive Web Application with Automatic Route Optimization

This Flask application provides:
1. Dropdown list of all filtered flight files
2. Dynamic loading of flight data (planned waypoints + actual points)
3. Automatic route optimization using A* algorithm with wind grids
4. Interactive map visualization with flight path, optimized route, and wind arrows
5. Flight duration and distance analysis including optimized route metrics

Setup Instructions:
1. Create project structure:
   your_project/
   ├── app.py                           # This file
   ├── templates/
   │   └── index.html                   # HTML template
   ├── filtered_flights_met_bounds/
   │   └── flight_*.parquet             # Your flight files
   ├── ECCC/
   │   └── 2025_JUN_par_met.parquet.gzip
   └── wind_grids_cache.pkl             # Wind grid cache file

2. Install dependencies:
   pip install flask pandas numpy scipy

3. Run the application:
   python app.py

4. Open browser to: http://localhost:5000
"""

import os

import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request
from datetime import timedelta
import logging
import math
import heapq
import warnings

from wind_service import load_wind_grid_from_file
from flight_optimization_gradient import main_fixed_temporal_gradient_search
from utils import extract_airport_codes, get_airport_info_online

from flight_efficiency_analysis import (
    generate_great_circle_waypoints,
    calculate_path_efficiency_metrics,
    analyze_wind_benefits_along_path
)

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
FILTERED_FLIGHTS_FOLDER = './results/filtered_flights_met_bounds_CYYZ_LFPG'
MET_DATA_FILE = './data/ECCC/2025_JUN_par_met.parquet.gzip'
WIND_GRID_CACHE_FILE = './results/wind_grids_cache.pkl'
MET_BOUNDS = {
    'lat_min': 41.25,
    'lat_max': 68.75,
    'lon_min': -87.50,
    'lon_max': -12.50
}

# Global variables to cache data
met_data_cache = None
wind_grid_cache = None
optimization_status = {}

flight_level_mapping = {
    'FL050': 50,
    'FL100': 100,
    'FL180': 180,
    'FL240': 240,
    'FL300': 300,
    'FL340': 340
}

# =============================================
# ROUTE OPTIMIZATION CLASSES
# =============================================

class OptimizedFlightNode:
    """Optimized node for A* search"""
    __slots__ = ['lat', 'lon', 'alt', 'time', 'parent', 'g_cost', 'h_cost', 'f_cost', 'wind_u', 'wind_v']

    def __init__(self, lat, lon, alt, time, parent=None):
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.time = time
        self.parent = parent
        self.g_cost = 0
        self.h_cost = 0
        self.f_cost = 0
        self.wind_u = 0
        self.wind_v = 0

    def __lt__(self, other):
        return self.f_cost < other.f_cost


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great circle distance between two points"""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing between two points"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = (math.cos(lat1) * math.sin(lat2) -
         math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    bearing = math.atan2(y, x)
    return math.degrees(bearing)


def wind_triangle_calculation(tas_knots, wind_u, wind_v, track_angle_deg):
    """Calculate wind triangle for ground speed"""
    tas_ms = tas_knots * 0.514444
    track_rad = math.radians(track_angle_deg)

    desired_track_x = math.sin(track_rad)
    desired_track_y = math.cos(track_rad)

    ground_speed_x = tas_ms * desired_track_x + wind_u
    ground_speed_y = tas_ms * desired_track_y + wind_v

    ground_speed_ms = math.sqrt(ground_speed_x ** 2 + ground_speed_y ** 2)
    ground_speed_knots = ground_speed_ms / 0.514444

    return ground_speed_knots


def optimize_flight_route(start_lat, start_lon, start_alt, start_time, end_lat, end_lon, end_alt, wind_grid):
    """
    Optimize flight route using A* algorithm with wind grids
    Returns list of waypoints as [lat, lon, alt, time_hours]
    """
    logger.info(
        f"Starting route optimization from ({start_lat:.2f}, {start_lon:.2f}) to ({end_lat:.2f}, {end_lon:.2f})")

    # Algorithm settings
    LAT_STEP = 0.1
    LON_STEP = 0.5
    ALT_STEP = 2000
    TAS_KNOTS = 450.0
    MAX_ITERATIONS = 1000  # Reduced for faster web response

    # Heuristic function
    def heuristic(lat, lon, end_lat, end_lon):
        distance_km = haversine_distance(lat, lon, end_lat, end_lon)
        distance_nm = distance_km * 0.539957
        return distance_nm / TAS_KNOTS

    # Initialize start node
    start_node = OptimizedFlightNode(start_lat, start_lon, start_alt, start_time)
    start_node.h_cost = heuristic(start_lat, start_lon, end_lat, end_lon)
    start_node.f_cost = start_node.h_cost

    open_list = [start_node]
    closed_set = {}
    iterations = 0

    # Possible movements
    movements = [
        (LAT_STEP, 0, 0), (-LAT_STEP, 0, 0),
        (0, LON_STEP, 0), (0, -LON_STEP, 0),
        (LAT_STEP, LON_STEP, 0), (LAT_STEP, -LON_STEP, 0),
        (-LAT_STEP, LON_STEP, 0), (-LAT_STEP, -LON_STEP, 0),
        (0, 0, ALT_STEP), (0, 0, -ALT_STEP)
    ]

    while open_list:
        iterations += 1
        current = heapq.heappop(open_list)

        # Check if already processed
        node_key = (round(current.lat, 1), round(current.lon, 1), current.alt)
        if node_key in closed_set and closed_set[node_key] <= current.g_cost:
            continue
        closed_set[node_key] = current.g_cost

        # Check if goal reached
        dist_to_goal = haversine_distance(current.lat, current.lon, end_lat, end_lon)
        if dist_to_goal < 100:  # 100 km threshold
            logger.info(f"Route optimization completed in {iterations} iterations")

            # Reconstruct path
            path = []
            while current:
                path.append([current.lat, current.lon, current.alt, current.g_cost])
                current = current.parent
            path.reverse()

            logger.info(f"Optimized route has {len(path)} waypoints, flight time: {path[-1][3]:.2f} hours")
            return path

        # Expand neighbors
        for d_lat, d_lon, d_alt in movements:
            new_lat = current.lat + d_lat
            new_lon = current.lon + d_lon
            new_alt = current.alt + d_alt

            # Boundary check
            if not (MET_BOUNDS['lat_min'] <= new_lat <= MET_BOUNDS['lat_max'] and
                    MET_BOUNDS['lon_min'] <= new_lon <= MET_BOUNDS['lon_max'] and
                    28000 <= new_alt <= 42000):
                continue

            neighbor_key = (round(new_lat, 1), round(new_lon, 1), new_alt)
            if neighbor_key in closed_set:
                continue

            # Calculate segment distance
            distance_km = haversine_distance(current.lat, current.lon, new_lat, new_lon)
            if d_alt != 0:
                distance_km += abs(d_alt) * 0.0003048

            if distance_km < 1:
                continue

            # Get wind data
            mid_lat = (current.lat + new_lat) / 2
            mid_lon = (current.lon + new_lon) / 2
            mid_alt = (current.alt + new_alt) / 2 / 100  # Convert to FL

            wind_u, wind_v = wind_grid.get_wind_fast(mid_lat, mid_lon, mid_alt, current.time)

            if np.isnan(wind_u):
                continue

            # Calculate ground speed with wind
            track_angle = calculate_bearing(current.lat, current.lon, new_lat, new_lon)
            ground_speed = wind_triangle_calculation(TAS_KNOTS, wind_u, wind_v, track_angle)

            if ground_speed <= 0:
                continue

            # Calculate time
            distance_nm = distance_km * 0.539957
            segment_time_hours = distance_nm / ground_speed

            # Create neighbor node
            new_time = current.time + timedelta(hours=segment_time_hours)
            neighbor = OptimizedFlightNode(new_lat, new_lon, new_alt, new_time, current)
            neighbor.wind_u = wind_u
            neighbor.wind_v = wind_v
            neighbor.g_cost = current.g_cost + segment_time_hours
            neighbor.h_cost = heuristic(new_lat, new_lon, end_lat, end_lon)
            neighbor.f_cost = neighbor.g_cost + neighbor.h_cost

            heapq.heappush(open_list, neighbor)

    logger.warning(f"Route optimization failed after {iterations} iterations")
    return []


# =============================================
# EXISTING HELPER FUNCTIONS
# =============================================

def load_wind_grid_cache():
    """Load wind grid cache"""
    global wind_grid_cache

    if wind_grid_cache is None:
        if os.path.exists(WIND_GRID_CACHE_FILE):
            wind_grid_cache = load_wind_grid_from_file(WIND_GRID_CACHE_FILE)
            if wind_grid_cache is None:
                logger.warning("Failed to load wind grid cache")
        else:
            logger.warning(f"Wind grid cache file not found: {WIND_GRID_CACHE_FILE}")

    return wind_grid_cache


def calculate_flight_distances(actual_points):
    """Calculate flight distances and efficiency metrics"""
    distance_info = {
        'total_flown_distance': 0,
        'great_circle_distance': 0,
        'efficiency_ratio': 0,
        'has_distance_data': False
    }

    if not actual_points or len(actual_points) < 2:
        logger.warning("Not enough points to calculate distances")
        return distance_info

    try:
        # Calculate total flown distance
        total_distance = 0
        for i in range(1, len(actual_points)):
            lat1, lon1 = actual_points[i - 1][0], actual_points[i - 1][1]
            lat2, lon2 = actual_points[i][0], actual_points[i][1]
            segment_distance = haversine_distance(lat1, lon1, lat2, lon2)
            total_distance += segment_distance

        # Calculate great circle distance
        origin_lat, origin_lon = actual_points[0][0], actual_points[0][1]
        dest_lat, dest_lon = actual_points[-1][0], actual_points[-1][1]
        great_circle_distance = haversine_distance(origin_lat, origin_lon, dest_lat, dest_lon)

        # Calculate efficiency ratio
        efficiency_ratio = total_distance / great_circle_distance if great_circle_distance > 0 else 0

        distance_info = {
            'total_flown_distance': total_distance,
            'great_circle_distance': great_circle_distance,
            'efficiency_ratio': efficiency_ratio,
            'has_distance_data': True
        }

        logger.info(
            f"Flight distances calculated - Flown: {total_distance:.2f} km, Great Circle: {great_circle_distance:.2f} km, Efficiency: {efficiency_ratio:.3f}")

    except Exception as e:
        logger.error(f"Error calculating flight distances: {e}")

    return distance_info


def calculate_optimized_distances(optimized_points):
    """Calculate optimized route distances"""
    distance_info = {
        'total_distance': 0,
        'flight_time': 0,
        'has_data': False
    }

    if not optimized_points or len(optimized_points) < 2:
        return distance_info

    try:
        # Calculate total distance
        total_distance = 0
        for i in range(1, len(optimized_points)):
            lat1, lon1 = optimized_points[i - 1][0], optimized_points[i - 1][1]
            lat2, lon2 = optimized_points[i][0], optimized_points[i][1]
            segment_distance = haversine_distance(lat1, lon1, lat2, lon2)
            total_distance += segment_distance

        # Get flight time from last point
        flight_time = optimized_points[-1][3]  # time_hours is at index 3

        distance_info = {
            'total_distance': total_distance,
            'flight_time': flight_time,
            'has_data': True
        }

        logger.info(f"Optimized route - Distance: {total_distance:.2f} km, Time: {flight_time:.2f} hours")

    except Exception as e:
        logger.error(f"Error calculating optimized distances: {e}")

    return distance_info


def generate_great_circle_points(lat1, lon1, lat2, lon2, num_points=100):
    """Generate points along the great circle path between two coordinates"""
    points = []

    try:
        # Convert to radians
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

        # Calculate the angular distance
        d = math.acos(
            math.sin(lat1_rad) * math.sin(lat2_rad) +
            math.cos(lat1_rad) * math.cos(lat2_rad) * math.cos(lon2_rad - lon1_rad)
        )

        # Generate points along the great circle
        for i in range(num_points + 1):
            f = i / num_points

            # Interpolation along the great circle
            A = math.sin((1 - f) * d) / math.sin(d)
            B = math.sin(f * d) / math.sin(d)

            x = A * math.cos(lat1_rad) * math.cos(lon1_rad) + B * math.cos(lat2_rad) * math.cos(lon2_rad)
            y = A * math.cos(lat1_rad) * math.sin(lon1_rad) + B * math.cos(lat2_rad) * math.sin(lon2_rad)
            z = A * math.sin(lat1_rad) + B * math.sin(lat2_rad)

            # Convert back to lat/lon
            lat = math.atan2(z, math.sqrt(x ** 2 + y ** 2))
            lon = math.atan2(y, x)

            # Convert to degrees
            points.append([math.degrees(lat), math.degrees(lon)])

    except Exception as e:
        logger.error(f"Error generating great circle points: {e}")
        # Fallback to simple linear interpolation
        for i in range(num_points + 1):
            f = i / num_points
            lat = lat1 + f * (lat2 - lat1)
            lon = lon1 + f * (lon2 - lon1)
            points.append([lat, lon])

    return points


def load_met_data():
    """Load and cache MET data"""
    global met_data_cache

    if met_data_cache is None:
        try:
            logger.info(f"Loading MET data from {MET_DATA_FILE}")
            met_data_cache = pd.read_parquet(MET_DATA_FILE)
            logger.info(f"Loaded {len(met_data_cache)} MET data records")
        except Exception as e:
            logger.error(f"Error loading MET data: {e}")
            met_data_cache = pd.DataFrame()

    return met_data_cache


def get_flight_level_from_altitude(altitude_ft):
    """Convert altitude in feet to nearest available flight level"""
    if pd.isna(altitude_ft) or altitude_ft < 1000:
        return 'FL050'

    # Convert to flight level (hundreds of feet)
    fl_numeric = int(altitude_ft / 100)

    # Find closest available flight level
    available_fls = [50, 100, 180, 240, 300, 340]
    closest_fl = min(available_fls, key=lambda x: abs(x - fl_numeric))

    return f"FL{closest_fl:03d}"


def get_flight_files():
    """Get list of all flight files in the filtered folder"""
    flight_files = []

    if not os.path.exists(FILTERED_FLIGHTS_FOLDER):
        logger.error(f"Filtered flights folder not found: {FILTERED_FLIGHTS_FOLDER}")
        return flight_files

    for filename in os.listdir(FILTERED_FLIGHTS_FOLDER):
        if filename.endswith('_filtered.parquet'):
            # Extract flight ID from filename
            flight_id = filename.replace('flight_', '').replace('_filtered.parquet', '')
            flight_id = flight_id.replace('_', ' ').replace('-', ':')

            # if flight_id.__contains__('CYYZ') and flight_id.__contains__('LFPG'):
            flight_files.append({
                'filename': filename,
                'flight_id': flight_id,
                'display_name': flight_id
            })

    # Sort by flight ID
    flight_files.sort(key=lambda x: x['flight_id'])

    logger.info(f"Found {len(flight_files)} flight files")
    return flight_files


def load_flight_data(filename):
    """Load flight data from parquet file"""
    try:
        filepath = os.path.join(FILTERED_FLIGHTS_FOLDER, filename)
        flight_df = pd.read_parquet(filepath)

        logger.info(f"Loaded flight data: {len(flight_df)} records")
        return flight_df
    except Exception as e:
        logger.error(f"Error loading flight data from {filename}: {e}")
        return None


def calculate_flight_duration(flight_df):
    """Calculate flight duration from departure to arrival time"""
    duration_info = {
        'departure_time': None,
        'arrival_time': None,
        'duration_hours': None,
        'duration_minutes': None,
        'duration_formatted': None,
        'estimated_arrival': False,
        'has_timing_data': False
    }

    try:
        # Check if required columns exist
        if 'depau' not in flight_df.columns:
            logger.warning("No departure time column (depau) found")
            return duration_info

        # Get departure time (should be consistent across all records)
        departure_times = flight_df['depau'].dropna()
        if len(departure_times) == 0:
            logger.warning("No valid departure times found")
            return duration_info

        # Use the first valid departure time
        departure_time = departure_times.iloc[0]
        duration_info['departure_time'] = departure_time
        duration_info['has_timing_data'] = True

        # Check for arrival time
        arrival_time = None
        estimated = False

        if 'arrau' in flight_df.columns:
            arrival_times = flight_df['arrau'].dropna()
            if len(arrival_times) > 0:
                arrival_time = arrival_times.iloc[0]
                logger.info("Using actual arrival time from arrau column")
            else:
                logger.info("No arrival time in arrau column, using last record timestamp")
                estimated = True
        else:
            logger.info("No arrau column found, using last record timestamp")
            estimated = True

        # If no arrival time, use timestamp of last record
        if arrival_time is None:
            # Check for timestamp columns (common names)
            timestamp_cols = ['timestamp', 'time', 'ts', 'datetime']
            timestamp_col = None

            for col in timestamp_cols:
                if col in flight_df.columns:
                    timestamp_col = col
                    break

            if timestamp_col:
                timestamps = flight_df[timestamp_col].dropna()
                if len(timestamps) > 0:
                    arrival_time = timestamps.iloc[-1]  # Last timestamp
                    estimated = True
                    logger.info(f"Using last timestamp from {timestamp_col} column as estimated arrival")
            else:
                logger.warning("No timestamp column found for estimated arrival time")
                return duration_info

        if arrival_time is not None:
            duration_info['arrival_time'] = arrival_time
            duration_info['estimated_arrival'] = estimated

            # Calculate duration
            try:
                # Convert to datetime if they're not already
                if isinstance(departure_time, str):
                    departure_dt = pd.to_datetime(departure_time)
                else:
                    departure_dt = departure_time

                if isinstance(arrival_time, str):
                    arrival_dt = pd.to_datetime(arrival_time)
                else:
                    arrival_dt = arrival_time

                # Calculate duration
                duration = arrival_dt - departure_dt
                total_seconds = duration.total_seconds()

                if total_seconds > 0:
                    duration_hours = total_seconds / 3600
                    duration_minutes = (total_seconds % 3600) / 60

                    duration_info['duration_hours'] = duration_hours
                    duration_info['duration_minutes'] = total_seconds / 60

                    # Format duration
                    hours = int(duration_hours)
                    minutes = int(duration_minutes)

                    if estimated:
                        duration_info['duration_formatted'] = f"{hours}h {minutes}m*"
                    else:
                        duration_info['duration_formatted'] = f"{hours}h {minutes}m"

                    logger.info(f"Flight duration calculated: {duration_info['duration_formatted']}")
                else:
                    logger.warning("Negative duration calculated - check time data")

            except Exception as e:
                logger.error(f"Error calculating duration: {e}")

    except Exception as e:
        logger.error(f"Error in calculate_flight_duration: {e}")

    return duration_info


def extract_planned_waypoints(flight_df):
    """Extract planned waypoints from first row with waypoints"""
    planned_waypoints = []

    if 'planned_waypoints' not in flight_df.columns:
        logger.warning("No planned_waypoints column found")
        return planned_waypoints

    planned_waypoints = []
    if 'planned_waypoints' in flight_df.columns:
        unique_waypoints = set()  # To avoid duplicates

        for wp_list in flight_df.planned_waypoints.dropna():
            if isinstance(wp_list, (list, np.ndarray)) and len(wp_list) > 0:
                # Convert numpy array to list if needed
                if isinstance(wp_list, np.ndarray):
                    wp_list = wp_list.tolist()

                # Extract lat/lon pairs
                for wp in wp_list:
                    try:
                        if isinstance(wp, (list, np.ndarray)) and len(wp) >= 2:
                            lat = float(wp[0])
                            lon = float(wp[1])

                            # Validate coordinates
                            if -90 <= lat <= 90 and -180 <= lon <= 180:
                                waypoint_key = (round(lat, 6), round(lon, 6))
                                if waypoint_key not in unique_waypoints:
                                    unique_waypoints.add(waypoint_key)
                                    planned_waypoints.append([lat, lon])
                    except (ValueError, TypeError, IndexError):
                        continue

    if len(planned_waypoints) == 0:
        logger.info("No valid planned waypoints found - will show actual points only")
    else:
        logger.info(f"Extracted {len(planned_waypoints)} planned waypoints")

    return planned_waypoints



def extract_actual_points(flight_df, max_points=200):
    """Extract actual flight points (ADS-B data)"""
    actual_points = []

    if 'la' not in flight_df.columns or 'lo' not in flight_df.columns:
        logger.warning("No latitude/longitude columns found")
        return actual_points

    # Filter valid coordinates
    valid_coords = flight_df[
        (flight_df['la'].notna()) &
        (flight_df['lo'].notna()) &
        (flight_df['la'] != 0) &
        (flight_df['lo'] != 0) &
        (flight_df['la'].between(-90, 90)) &
        (flight_df['lo'].between(-180, 180))
        ]

    if len(valid_coords) > max_points:
        # Sample points evenly across the flight
        indices = np.linspace(0, len(valid_coords) - 1, max_points, dtype=int)
        valid_coords = valid_coords.iloc[indices]

    # Extract coordinates with altitude and time
    for _, row in valid_coords.iterrows():
        try:
            lat = float(row['la'])
            lon = float(row['lo'])
            alt = float(row['alt']) if pd.notna(row['alt']) else 35000
            time_val = row['t'] if 't' in row.index else pd.Timestamp.now()

            actual_points.append([lat, lon, alt, time_val])

        except (ValueError, TypeError):
            continue

    logger.info(f"Extracted {len(actual_points)} actual flight points")
    return actual_points


def process_entire_wind_grid(met_data):
    """Process entire MET data grid to extract wind information for all grid cells"""
    wind_grid_data = []

    logger.info(f"Processing entire wind grid with {len(met_data)} MET data cells")

    # Check if MET data has required columns
    required_cols = ['lat_min', 'lat_max', 'lon_min', 'lon_max']
    if not all(col in met_data.columns for col in required_cols):
        logger.error(f"MET data missing required columns: {required_cols}")
        return wind_grid_data

    for idx, met_cell in met_data.iterrows():
        try:
            # Calculate center coordinates of the grid cell
            lat_center = (met_cell['lat_min'] + met_cell['lat_max']) / 2
            lon_center = (met_cell['lon_min'] + met_cell['lon_max']) / 2

            # Process each available flight level
            for flight_level in flight_level_mapping.keys():
                wind_dir_col = f"{flight_level}_direction"
                wind_speed_col = f"{flight_level}_speed"
                wind_temp_col = f"{flight_level}_temperature"

                # Check if this flight level has wind data
                if wind_dir_col in met_cell.index and wind_speed_col in met_cell.index:
                    wind_direction = met_cell[wind_dir_col]
                    wind_speed = met_cell[wind_speed_col]
                    wind_temp = met_cell[wind_temp_col] if wind_temp_col in met_cell.index else np.nan

                    # Only add if we have valid wind data
                    if not pd.isna(wind_direction) and not pd.isna(wind_speed):
                        wind_grid_data.append({
                            'lat': float(lat_center),
                            'lon': float(lon_center),
                            'altitude': flight_level_mapping[flight_level] * 100,  # Convert FL to feet
                            'flight_level': flight_level,
                            'speed': float(wind_speed),
                            'direction': float(wind_direction),
                            'temperature': float(wind_temp) if not pd.isna(wind_temp) else None,
                            'met_lat_center': float(lat_center),
                            'met_lon_center': float(lon_center),
                            'grid_cell_index': idx
                        })

        except Exception as e:
            logger.warning(f"Error processing grid cell {idx}: {e}")
            continue

    logger.info(f"Processed {len(wind_grid_data)} wind grid points")
    return wind_grid_data

def get_flight_altitude_stats(actual_points):
    """Get altitude statistics from flight points"""
    if not actual_points:
        return {
            'min_altitude': 0,
            'max_altitude': 0,
            'avg_altitude': 0,
            'cruise_altitude': 0,
            'max_flight_level': 'FL050'
        }

    altitudes = [point[2] for point in actual_points if point[2] > 0]

    if not altitudes:
        return {
            'min_altitude': 0,
            'max_altitude': 0,
            'avg_altitude': 0,
            'cruise_altitude': 0,
            'max_flight_level': 'FL050'
        }

    min_alt = min(altitudes)
    max_alt = max(altitudes)
    avg_alt = sum(altitudes) / len(altitudes)

    # Cruise altitude is typically the most common altitude above 10,000 feet
    cruise_altitudes = [alt for alt in altitudes if alt > 10000]
    if cruise_altitudes:
        # Find the most common altitude (mode) for cruise
        from collections import Counter
        altitude_counts = Counter([int(alt / 1000) * 1000 for alt in cruise_altitudes])  # Round to nearest 1000
        cruise_altitude = altitude_counts.most_common(1)[0][0]
    else:
        cruise_altitude = max_alt

    # Get flight level for maximum altitude
    max_flight_level = get_flight_level_from_altitude(max_alt)

    return {
        'min_altitude': min_alt,
        'max_altitude': max_alt,
        'avg_altitude': avg_alt,
        'cruise_altitude': cruise_altitude,
        'max_flight_level': max_flight_level
    }


def get_extended_met_data_for_flight_level(met_data, flight_level, flight_bounds):
    """Get extended MET data for a specific flight level around the flight area"""
    extended_met_data = []

    # Expand the search area around the flight path
    lat_min, lat_max = flight_bounds['lat_min'] - 10, flight_bounds['lat_max'] + 10
    lon_min, lon_max = flight_bounds['lon_min'] - 7, flight_bounds['lon_max'] + 7

    # Ensure we stay within MET bounds
    lat_min = max(lat_min, MET_BOUNDS['lat_min'])
    lat_max = min(lat_max, MET_BOUNDS['lat_max'])
    lon_min = max(lon_min, MET_BOUNDS['lon_min'])
    lon_max = min(lon_max, MET_BOUNDS['lon_max'])

    logger.info(
        f"Getting extended MET data for {flight_level} in area: lat [{lat_min:.1f}, {lat_max:.1f}], lon [{lon_min:.1f}, {lon_max:.1f}]")

    try:
        # Get MET data in the expanded area
        area_met_data = met_data.loc[
            (met_data['latitude_center'] >= lat_min) &
            (met_data['latitude_center'] <= lat_max) &
            (met_data['longitude_center'] >= lon_min) &
            (met_data['longitude_center'] <= lon_max)
            ]

        wind_dir_col = f"{flight_level}_direction"
        wind_speed_col = f"{flight_level}_speed"
        wind_temp_col = f"{flight_level}_temperature"

        for _, met_row in area_met_data.iterrows():
            wind_direction = met_row.get(wind_dir_col, np.nan)
            wind_speed = met_row.get(wind_speed_col, np.nan)
            wind_temp = met_row.get(wind_temp_col, np.nan)

            # If no data for this flight level, try nearest available level
            if pd.isna(wind_direction) or pd.isna(wind_speed):
                for fl_name in flight_level_mapping.keys():
                    if fl_name != flight_level:
                        wind_dir_col_alt = f"{fl_name}_direction"
                        wind_speed_col_alt = f"{fl_name}_speed"

                        if wind_dir_col_alt in met_row.index and wind_speed_col_alt in met_row.index:
                            temp_dir = met_row[wind_dir_col_alt]
                            temp_speed = met_row[wind_speed_col_alt]

                            if not pd.isna(temp_dir) and not pd.isna(temp_speed):
                                wind_direction = temp_dir
                                wind_speed = temp_speed
                                break

            if not pd.isna(wind_direction) and not pd.isna(wind_speed):
                extended_met_data.append({
                    'lat': float(met_row.get('latitude_center', 0)),
                    'lon': float(met_row.get('longitude_center', 0)),
                    'altitude': 0,  # Will be updated based on flight level
                    'flight_level': flight_level,
                    'speed': float(wind_speed),
                    'direction': float(wind_direction),
                    'temperature': float(wind_temp) if not pd.isna(wind_temp) else None,
                    'met_lat_center': float(met_row.get('latitude_center', 0)),
                    'met_lon_center': float(met_row.get('longitude_center', 0)),
                    'point_index': -1,  # Indicates this is extended MET data
                    'source': 'extended_met'
                })

    except Exception as e:
        logger.warning(f"Error getting extended MET data: {e}")

    logger.info(f"Found {len(extended_met_data)} extended MET data points for {flight_level}")
    return extended_met_data


def match_flight_to_met_data(actual_points, met_data):
    """Match flight points to MET data by coordinates and flight level"""
    matched_data = []

    logger.info(f"Matching {len(actual_points)} flight points to MET data")

    # Check if MET data has required columns
    required_cols = ['lat_min', 'lat_max', 'lon_min', 'lon_max']
    if not all(col in met_data.columns for col in required_cols):
        logger.error(f"MET data missing required columns: {required_cols}")
        return matched_data

    for point_idx, point in enumerate(actual_points):
        lat, lon, alt = point[0], point[1], point[2]

        # Get flight level from altitude
        flight_level = get_flight_level_from_altitude(alt)

        # Find MET data cells that contain this flight point
        try:
            met_matches = met_data.loc[
                (met_data['lat_min'] <= lat) &
                (met_data['lat_max'] >= lat) &
                (met_data['lon_min'] <= lon) &
                (met_data['lon_max'] >= lon)
                ]

            if len(met_matches) > 0:
                # Take the first match
                met_match = met_matches.iloc[0]

                # Extract wind data for this flight level
                wind_direction = np.nan
                wind_speed = np.nan
                wind_temp = np.nan

                # Try to get wind data for calculated flight level
                wind_dir_col = f"{flight_level}_direction"
                wind_speed_col = f"{flight_level}_speed"
                wind_temp_col = f"{flight_level}_temperature"

                if wind_dir_col in met_match.index:
                    wind_direction = met_match[wind_dir_col]
                if wind_speed_col in met_match.index:
                    wind_speed = met_match[wind_speed_col]
                if wind_temp_col in met_match.index:
                    wind_temp = met_match[wind_temp_col]

                # If no data for this flight level, find nearest available level
                if pd.isna(wind_direction) or pd.isna(wind_speed):
                    for fl_name in flight_level_mapping.keys():
                        if fl_name != flight_level:
                            wind_dir_col = f"{fl_name}_direction"
                            wind_speed_col = f"{fl_name}_speed"

                            if wind_dir_col in met_match.index and wind_speed_col in met_match.index:
                                temp_dir = met_match[wind_dir_col]
                                temp_speed = met_match[wind_speed_col]

                                if not pd.isna(temp_dir) and not pd.isna(temp_speed):
                                    wind_direction = temp_dir
                                    wind_speed = temp_speed
                                    break

                # Only add if we have valid wind data
                if not pd.isna(wind_direction) and not pd.isna(wind_speed):
                    try:
                        matched_data.append({
                            'lat': float(lat),
                            'lon': float(lon),
                            'altitude': float(alt),
                            'flight_level': flight_level,
                            'speed': float(wind_speed),
                            'direction': float(wind_direction),
                            'temperature': float(wind_temp) if not pd.isna(wind_temp) else None,
                            'met_lat_center': float(met_match.get('latitude_center', lat)),
                            'met_lon_center': float(met_match.get('longitude_center', lon)),
                            'point_index': point_idx
                        })
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error converting wind data for point {point_idx}: {e}")
                        continue

        except Exception as e:
            logger.warning(f"Error matching point {point_idx} at ({lat}, {lon}): {e}")
            continue

    logger.info(f"Matched {len(matched_data)} flight points to MET data")
    return matched_data


# =============================================
# API ROUTES
# =============================================

@app.route('/')
def index():
    """Main page - serves the HTML template"""
    return render_template('index.html')


@app.route('/api/flights')
def get_flights():
    """API endpoint to get list of available flights"""
    try:
        flight_files = get_flight_files()
        return jsonify({
            'success': True,
            'flights': flight_files
        })
    except Exception as e:
        logger.error(f"Error getting flights: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/flight-data')
def get_flight_data():
    """API endpoint to get flight data and automatically optimize route"""
    try:
        filename = request.args.get('filename')
        if not filename:
            return jsonify({
                'success': False,
                'error': 'No filename provided'
            }), 400

        origin, destination = extract_airport_codes(filename)
        print(f"Origin: {origin}")
        print(f"Destination: {destination}")

        if not origin or not destination:
            print("Could not extract airport codes from filename")
            return None

        print(f"Extracted codes: {origin} -> {destination}")

        results = {}
        print("Looking up airports online...")
        origin_info = get_airport_info_online(origin)
        dest_info = get_airport_info_online(destination)

        results['origin'] = origin_info
        results['destination'] = dest_info

        # Load flight data
        flight_df = load_flight_data(filename)
        if flight_df is None:
            return jsonify({
                'success': False,
                'error': 'Could not load flight data'
            }), 500

        # Calculate flight duration
        duration_info = calculate_flight_duration(flight_df)

        # Extract planned waypoints and actual points
        planned_waypoints = extract_planned_waypoints(flight_df)

        # Extract actual flight points
        actual_points = extract_actual_points(flight_df)

        # Calculate flight distances
        distance_info = calculate_flight_distances(actual_points)

        # Generate great circle path for actual flight
        great_circle_points = []
        great_circle_for_actual = []
        if len(actual_points) >= 2:
            origin_point = actual_points[0]
            destination_point = actual_points[-1]

            # Original great circle (for display)
            great_circle_points = generate_great_circle_points(
                origin_point[0], origin_point[1], destination_point[0], destination_point[1]
            )

            # Enhanced great circle with more points (for analysis)
            great_circle_for_actual = generate_great_circle_waypoints(
                origin_point[0], origin_point[1], destination_point[0], destination_point[1],
                num_points=100
            )

        # Get flight altitude statistics
        altitude_stats = get_flight_altitude_stats(actual_points)

        # Calculate flight bounds for extended MET data
        if actual_points:
            flight_bounds = {
                'lat_min': min(point[0] for point in actual_points),
                'lat_max': max(point[0] for point in actual_points),
                'lon_min': min(point[1] for point in actual_points),
                'lon_max': max(point[1] for point in actual_points)
            }
        else:
            flight_bounds = {
                'lat_min': MET_BOUNDS['lat_min'],
                'lat_max': MET_BOUNDS['lat_max'],
                'lon_min': MET_BOUNDS['lon_min'],
                'lon_max': MET_BOUNDS['lon_max']
            }

        # Load MET data
        met_data = load_met_data()

        # Match flight points to MET data (original functionality)
        matched_met_data = []
        if len(met_data) > 0:
            matched_met_data = match_flight_to_met_data(actual_points, met_data)

        # Get extended MET data for maximum flight level
        extended_met_data = []
        if len(met_data) > 0 and altitude_stats['max_flight_level']:
            extended_met_data = get_extended_met_data_for_flight_level(
                met_data,
                altitude_stats['max_flight_level'],
                flight_bounds
            )

        # Combine both types of MET data
        all_met_data = extended_met_data

        # Load wind grid cache
        wind_grid = load_wind_grid_cache()

        # Combine both types of MET data
        all_met_data = extended_met_data

        # **ENHANCED: Route optimization with comprehensive analysis**
        optimized_route_heuristic = []
        optimized_heuristic_distance_info = {'has_data': False}
        optimized_route_gradient = []
        optimized_gradient_distance_info = {'has_data': False}

        # A* Heuristic optimization
        if len(actual_points) >= 2 and wind_grid is not None:
            logger.info("Starting A* heuristic route optimization...")
            optimized_heuristic_distance_info, optimized_route_heuristic = a_star_heuristic(
                actual_points, optimized_route_heuristic, wind_grid
            )

        # Gradient optimization
        try:
            logger.info("Starting gradient-based route optimization...")
            optimized_route_gradient, optimized_gradient_distance_info = main_fixed_temporal_gradient_search(wind_grid, flight_df)
            # optimized_route_gradient = None
            if optimized_route_gradient is not None and len(optimized_route_gradient) > 0:
                optimized_route_gradient = optimized_route_gradient[['lat', 'lon', 'alt', 'f_cost']].values.tolist()
            else:
                optimized_route_gradient = []
        except Exception as e:
            logger.warning(f"Gradient optimization failed: {e}")
            optimized_route_gradient = []

        # **NEW: Comprehensive Efficiency Analysis**
        efficiency_analysis = {}

        if wind_grid is not None:
            logger.info("Calculating comprehensive efficiency analysis...")

            try:
                # Analyze actual flight path
                actual_efficiency = calculate_path_efficiency_metrics(actual_points, "actual_flight")
                actual_wind_analysis = analyze_wind_benefits_along_path(actual_points, wind_grid)

                # Analyze optimized routes
                heuristic_efficiency = {}
                heuristic_wind_analysis = {}
                if optimized_route_heuristic:
                    heuristic_efficiency = calculate_path_efficiency_metrics(optimized_route_heuristic,
                                                                             "optimized_heuristic")
                    heuristic_wind_analysis = analyze_wind_benefits_along_path(optimized_route_heuristic, wind_grid)

                gradient_efficiency = {}
                gradient_wind_analysis = {}
                if optimized_route_gradient:
                    gradient_efficiency = calculate_path_efficiency_metrics(optimized_route_gradient,
                                                                            "optimized_gradient")
                    gradient_wind_analysis = analyze_wind_benefits_along_path(optimized_route_gradient, wind_grid)

                # Analyze great circle
                great_circle_efficiency = {}
                if great_circle_for_actual:
                    great_circle_efficiency = calculate_path_efficiency_metrics(great_circle_for_actual, "great_circle")

                # Calculate comprehensive comparison
                efficiency_analysis = {
                    'actual_flight': {
                        'path_metrics': actual_efficiency,
                        'wind_analysis': actual_wind_analysis
                    },
                    'optimized_heuristic': {
                        'path_metrics': heuristic_efficiency,
                        'wind_analysis': heuristic_wind_analysis
                    },
                    'optimized_gradient': {
                        'path_metrics': gradient_efficiency,
                        'wind_analysis': gradient_wind_analysis
                    },
                    'great_circle': {
                        'path_metrics': great_circle_efficiency
                    }
                }

                # Calculate comparative savings
                if actual_efficiency.get('has_data') and heuristic_efficiency.get('has_data'):
                    actual_dist = actual_efficiency['total_distance_km']
                    heuristic_dist = heuristic_efficiency['total_distance_km']

                    efficiency_analysis['heuristic_savings'] = {
                        'distance_savings_km': actual_dist - heuristic_dist,
                        'distance_savings_nm': (actual_dist - heuristic_dist) * 0.539957,
                        'distance_savings_percent': (
                                    (actual_dist - heuristic_dist) / actual_dist * 100) if actual_dist > 0 else 0,
                        'efficiency_improvement': (heuristic_efficiency['efficiency_ratio'] - actual_efficiency[
                            'efficiency_ratio']) * 100
                    }

                    # Add wind benefit comparison
                    if actual_wind_analysis.get('has_data') and heuristic_wind_analysis.get('has_data'):
                        actual_wind_benefit = actual_wind_analysis['wind_statistics']['average_wind_benefit_knots']
                        heuristic_wind_benefit = heuristic_wind_analysis['wind_statistics'][
                            'average_wind_benefit_knots']

                        efficiency_analysis['heuristic_savings'].update({
                            'wind_benefit_improvement_knots': heuristic_wind_benefit - actual_wind_benefit,
                            'actual_avg_wind_benefit_knots': actual_wind_benefit,
                            'optimized_avg_wind_benefit_knots': heuristic_wind_benefit
                        })

                        # Calculate time savings if available
                        actual_time = actual_wind_analysis['wind_statistics'].get('total_flight_time_hours')
                        heuristic_time = heuristic_wind_analysis['wind_statistics'].get('total_flight_time_hours')

                        if actual_time and heuristic_time:
                            efficiency_analysis['heuristic_savings'].update({
                                'time_savings_hours': actual_time - heuristic_time,
                                'time_savings_minutes': (actual_time - heuristic_time) * 60,
                                'time_savings_percent': ((
                                                                     actual_time - heuristic_time) / actual_time * 100) if actual_time > 0 else 0
                            })

                # Similar analysis for gradient optimization
                if actual_efficiency.get('has_data') and gradient_efficiency.get('has_data'):
                    actual_dist = actual_efficiency['total_distance_km']
                    gradient_dist = gradient_efficiency['total_distance_km']

                    efficiency_analysis['gradient_savings'] = {
                        'distance_savings_km': actual_dist - gradient_dist,
                        'distance_savings_nm': (actual_dist - gradient_dist) * 0.539957,
                        'distance_savings_percent': (
                                    (actual_dist - gradient_dist) / actual_dist * 100) if actual_dist > 0 else 0,
                        'efficiency_improvement': (gradient_efficiency['efficiency_ratio'] - actual_efficiency[
                            'efficiency_ratio']) * 100
                    }

                logger.info("Comprehensive efficiency analysis completed successfully")

            except Exception as e:
                logger.error(f"Error in comprehensive efficiency analysis: {e}")
                efficiency_analysis = {'error': str(e)}
        else:
            logger.warning("Wind grid not available - efficiency analysis limited")
            efficiency_analysis = {'error': 'Wind grid not available'}

        # Prepare enhanced response
        response = {
            'success': True,
            'flight_data': {
                'filename': filename,
                'origin_country': origin_info.get('country', 'Unknown'),
                'destination_country': dest_info.get('country', 'Unknown'),
                'planned_waypoints': planned_waypoints,
                'actual_points': actual_points,
                'great_circle_points': great_circle_points,
                'great_circle_for_actual': great_circle_for_actual,
                'total_records': len(flight_df),
                'met_matches': len(matched_met_data),
                'altitude_stats': altitude_stats,
                'flight_bounds': flight_bounds,
                'duration_info': duration_info,
                'distance_info': distance_info,
                'optimized_route': optimized_route_heuristic,
                'optimized_distance_info': optimized_heuristic_distance_info,
                'optimized_route_gradient': optimized_route_gradient,
                'optimized_gradient_distance_info': optimized_gradient_distance_info
            },
            'met_data': all_met_data,
            'extended_met_data': extended_met_data,
            'met_bounds': MET_BOUNDS,
            'efficiency_analysis': efficiency_analysis,
            'airport_info': {
                'origin': origin_info,
                'destination': dest_info
            }
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing flight data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Helper function to format efficiency summary for display
def format_efficiency_summary(efficiency_analysis):
    """
    Format efficiency analysis into a human-readable summary
    """
    summary = {
        'has_data': False,
        'summary_text': 'No efficiency data available'
    }

    try:
        if 'heuristic_savings' in efficiency_analysis:
            savings = efficiency_analysis['heuristic_savings']

            distance_saved = savings.get('distance_savings_km', 0)
            distance_percent = savings.get('distance_savings_percent', 0)
            wind_improvement = savings.get('wind_benefit_improvement_knots', 0)
            time_saved_minutes = savings.get('time_savings_minutes', 0)

            summary_parts = []

            if distance_saved > 0:
                summary_parts.append(f"Distance saved: {distance_saved:.1f} km ({distance_percent:.1f}%)")

            if wind_improvement > 0:
                summary_parts.append(f"Wind benefit improved by {wind_improvement:.1f} knots")
            elif wind_improvement < 0:
                summary_parts.append(f"Wind benefit reduced by {abs(wind_improvement):.1f} knots")

            if time_saved_minutes and time_saved_minutes > 0:
                if time_saved_minutes >= 60:
                    hours = int(time_saved_minutes // 60)
                    minutes = int(time_saved_minutes % 60)
                    summary_parts.append(f"Time saved: {hours}h {minutes}m")
                else:
                    summary_parts.append(f"Time saved: {time_saved_minutes:.0f} minutes")

            if summary_parts:
                summary['summary_text'] = "; ".join(summary_parts)
                summary['has_data'] = True
            else:
                summary['summary_text'] = "Optimized route shows minimal improvement over actual flight"
                summary['has_data'] = True

    except Exception as e:
        summary['summary_text'] = f"Error formatting efficiency summary: {str(e)}"

    return summary

def a_star_heuristic(actual_points, optimized_route, wind_grid):
    optimized_distance_info = {'has_data': False}
    if len(actual_points) >= 2:

        if wind_grid is not None:
            logger.info("Starting automatic route optimization...")

            # Extract start and end points
            start_point = actual_points[0]
            end_point = actual_points[-1]

            start_lat, start_lon, start_alt = start_point[0], start_point[1], start_point[2]
            end_lat, end_lon, end_alt = end_point[0], end_point[1], end_point[2]
            start_time = start_point[3] if len(start_point) > 3 else pd.Timestamp.now()

            # Run optimization
            optimized_route = optimize_flight_route(
                start_lat, start_lon, start_alt, start_time,
                end_lat, end_lon, end_alt, wind_grid
            )

            # Calculate optimized route metrics
            if optimized_route:
                optimized_distance_info = calculate_optimized_distances(optimized_route)
                logger.info(f"Optimized route completed with {len(optimized_route)} waypoints")
            else:
                logger.warning("Route optimization failed")
        else:
            logger.warning("Wind grid cache not available for route optimization")
    return optimized_distance_info, optimized_route

def create_templates_folder():
    """Create templates folder and instructions if it doesn't exist"""
    templates_dir = 'templates'
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)

        # Create instruction file
        instruction_file = os.path.join(templates_dir, 'README.txt')
        with open(instruction_file, 'w') as f:
            f.write("""
IMPORTANT: You need to create the HTML template file here!

1. Create a file named 'index.html' in this templates folder
2. Copy the HTML template code from the artifacts
3. Save it as templates/index.html

The Flask app is looking for: templates/index.html

After creating the file, restart the Flask app:
python app.py

Then open: http://localhost:5000
""")

        print(f"Created templates folder: {templates_dir}")
        print(f"Please create templates/index.html with the HTML template code")
        print(f"See {instruction_file} for instructions")
        return False

    # Check if index.html exists
    index_file = os.path.join(templates_dir, 'index.html')
    if not os.path.exists(index_file):
        print(f"Missing file: {index_file}")
        print("Please create templates/index.html with the HTML template code")
        return False

    return True


if __name__ == '__main__':
    print("=" * 60)
    print("FLIGHT-MET DATA WEB APPLICATION WITH AUTOMATIC ROUTE OPTIMIZATION")
    print("=" * 60)

    # Check if required files exist
    if not os.path.exists(FILTERED_FLIGHTS_FOLDER):
        print(f"❌ Error: Filtered flights folder not found: {FILTERED_FLIGHTS_FOLDER}")
        print("Please ensure your flight files are in the correct folder")
        exit(1)

    if not os.path.exists(MET_DATA_FILE):
        print(f"❌ Error: MET data file not found: {MET_DATA_FILE}")
        print("Please ensure your MET data file is in the correct location")
        exit(1)

    # Check for wind grid cache file
    if not os.path.exists(WIND_GRID_CACHE_FILE):
        print(f"⚠️  Warning: Wind grid cache file not found: {WIND_GRID_CACHE_FILE}")
        print("Route optimization will not be available until you create the wind grid cache.")
        print("Please run the wind grid preprocessing script first.")
    else:
        print(f"✅ Wind grid cache file found: {WIND_GRID_CACHE_FILE}")

    # Create templates folder and check for HTML file
    if not create_templates_folder():
        exit(1)

    # Count flight files
    flight_files = get_flight_files()
    print(f"✅ Found {len(flight_files)} flight files")
    print(f"✅ MET data file: {MET_DATA_FILE}")
    print(f"✅ Templates ready")

    print("\n" + "=" * 60)
    print("STARTING FLASK SERVER...")
    print("=" * 60)
    print("🚀 Open your browser and go to: http://localhost:5000")
    print("📝 DO NOT open the HTML file directly!")
    print("⚠️  The app must be accessed through Flask server")
    print("🔧 Automatic route optimization will run when loading flight data")
    print("=" * 60)

    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\n👋 Flask server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting Flask server: {e}")
