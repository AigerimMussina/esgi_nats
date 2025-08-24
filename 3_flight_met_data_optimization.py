import pandas as pd
import numpy as np
import datetime
from scipy.interpolate import griddata, RegularGridInterpolator
import pickle
import os
from typing import Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

# Algorithm settings
LAT_STEP = 0.1  # Increased step size for faster search
LON_STEP = 0.5  # Increased step size for faster search
ALT_STEP = 2000  # feet
TAS_KNOTS = 450.0
MAX_ITERATIONS = 3000  # Reduced for faster search

class WindGridOptimizer:
    """
    Optimized wind grid system for fast wind lookups during flight path optimization
    """

    def __init__(self, met_data: pd.DataFrame, grid_resolution: float = 0.01):
        """
        Initialize the wind grid optimizer

        Args:
            met_data: Processed meteorological data with valid_forecast_time
            grid_resolution: Grid resolution in degrees (smaller = more accurate but slower)
        """
        self.met_data = met_data
        self.grid_resolution = grid_resolution
        self.time_grids = {}
        self.spatial_bounds = self._calculate_bounds()
        self.available_flight_levels = self._get_available_flight_levels()
        self.grid_coords = self._create_grid_coordinates()

        print(f"Initializing WindGridOptimizer:")
        print(f"  Grid resolution: {grid_resolution}°")
        print(f"  Spatial bounds: {self.spatial_bounds}")
        print(f"  Available flight levels: {self.available_flight_levels}")
        print(
            f"  Grid dimensions: {len(self.grid_coords['lat'])} x {len(self.grid_coords['lon'])} x {len(self.available_flight_levels)}")

    def _calculate_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Calculate spatial bounds of meteorological data"""
        return {
            'lat': (self.met_data['latitude_center'].min(), self.met_data['latitude_center'].max()),
            'lon': (self.met_data['longitude_center'].min(), self.met_data['longitude_center'].max())
        }

    def _get_available_flight_levels(self) -> list:
        """Extract available flight levels from meteorological data"""
        fl_cols = [col for col in self.met_data.columns if col.endswith('_speed')]
        flight_levels = []

        for col in fl_cols:
            fl_str = col.split('_')[0]
            if fl_str.startswith('FL'):
                fl_number = int(fl_str.replace('FL', ''))
                flight_levels.append(fl_number)

        return sorted(flight_levels)

    def _create_grid_coordinates(self) -> Dict[str, np.ndarray]:
        """Create regular grid coordinates"""
        lat_min, lat_max = self.spatial_bounds['lat']
        lon_min, lon_max = self.spatial_bounds['lon']

        # Extend bounds slightly to avoid edge effects
        lat_buffer = self.grid_resolution
        lon_buffer = self.grid_resolution

        lat_grid = np.arange(lat_min - lat_buffer, lat_max + lat_buffer + self.grid_resolution, self.grid_resolution)
        lon_grid = np.arange(lon_min - lon_buffer, lon_max + lon_buffer + self.grid_resolution, self.grid_resolution)

        return {
            'lat': lat_grid,
            'lon': lon_grid,
            'fl': np.array(self.available_flight_levels)
        }

    def _interpolate_wind_for_time_fl(self, time_snapshot: pd.DataFrame, flight_level: int) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Interpolate wind components for a specific time and flight level

        Returns:
            Tuple of (u_wind_grid, v_wind_grid) as 2D arrays
        """
        fl_str = f'FL{flight_level:03d}'
        speed_col = f'{fl_str}_speed'
        dir_col = f'{fl_str}_direction'

        if speed_col not in time_snapshot.columns or dir_col not in time_snapshot.columns:
            # Return zero wind if flight level not available
            grid_shape = (len(self.grid_coords['lat']), len(self.grid_coords['lon']))
            return np.zeros(grid_shape), np.zeros(grid_shape)

        # Get wind data
        lats = time_snapshot['latitude_center'].values
        lons = time_snapshot['longitude_center'].values
        speeds = time_snapshot[speed_col].values
        directions_deg = time_snapshot[dir_col].values * 10  # Decode direction

        # Convert to u, v components
        directions_rad = np.radians(directions_deg)
        u_wind = speeds * np.sin(directions_rad)
        v_wind = speeds * np.cos(directions_rad)

        # Create mesh grid for interpolation
        lon_mesh, lat_mesh = np.meshgrid(self.grid_coords['lon'], self.grid_coords['lat'])

        # Interpolate to regular grid
        try:
            u_grid = griddata(
                points=np.column_stack([lons, lats]),
                values=u_wind,
                xi=(lon_mesh, lat_mesh),
                method='linear',
                fill_value=0.0
            )

            v_grid = griddata(
                points=np.column_stack([lons, lats]),
                values=v_wind,
                xi=(lon_mesh, lat_mesh),
                method='linear',
                fill_value=0.0
            )

            # Fill NaN values with nearest neighbor
            if np.any(np.isnan(u_grid)) or np.any(np.isnan(v_grid)):
                u_grid_nn = griddata(
                    points=np.column_stack([lons, lats]),
                    values=u_wind,
                    xi=(lon_mesh, lat_mesh),
                    method='nearest'
                )
                v_grid_nn = griddata(
                    points=np.column_stack([lons, lats]),
                    values=v_wind,
                    xi=(lon_mesh, lat_mesh),
                    method='nearest'
                )

                u_grid = np.where(np.isnan(u_grid), u_grid_nn, u_grid)
                v_grid = np.where(np.isnan(v_grid), v_grid_nn, v_grid)

            return u_grid, v_grid

        except Exception as e:
            print(f"Warning: Interpolation failed for FL{flight_level}: {e}")
            grid_shape = (len(self.grid_coords['lat']), len(self.grid_coords['lon']))
            return np.zeros(grid_shape), np.zeros(grid_shape)

    def precompute_wind_grids(self, max_time_intervals: int = 100) -> None:
        """
        Precompute wind grids for all time intervals and flight levels

        Args:
            max_time_intervals: Maximum number of time intervals to process
        """
        print("Starting wind grid precomputation...")

        # Get unique time intervals
        unique_times = sorted(self.met_data['valid_forecast_time'].unique())

        if len(unique_times) > max_time_intervals:
            print(f"Limiting to {max_time_intervals} time intervals out of {len(unique_times)}")
            # Sample time intervals evenly
            indices = np.linspace(0, len(unique_times) - 1, max_time_intervals, dtype=int)
            unique_times = [unique_times[i] for i in indices]

        total_grids = len(unique_times) * len(self.available_flight_levels)
        processed = 0

        for time_idx, time_point in enumerate(unique_times):
            print(f"Processing time {time_idx + 1}/{len(unique_times)}: {time_point}")

            # Get data for this time point
            time_snapshot = self.met_data[self.met_data['valid_forecast_time'] == time_point]

            if time_snapshot.empty:
                continue

            # Create storage for this time point
            time_grids = {
                'u_wind': {},
                'v_wind': {},
                'flight_levels': self.available_flight_levels
            }

            # Process each flight level
            for fl in self.available_flight_levels:
                u_grid, v_grid = self._interpolate_wind_for_time_fl(time_snapshot, fl)
                time_grids['u_wind'][fl] = u_grid
                time_grids['v_wind'][fl] = v_grid

                processed += 1
                if processed % 10 == 0:
                    print(f"  Processed {processed}/{total_grids} grids ({processed / total_grids * 100:.1f}%)")

            # Store grids for this time point
            self.time_grids[time_point] = time_grids

        print(f"Wind grid precomputation complete! Generated {processed} grids.")
        print(f"Memory usage: ~{self._estimate_memory_usage():.1f} MB")

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        if not self.time_grids:
            return 0

        # Calculate size of one grid
        grid_size = len(self.grid_coords['lat']) * len(self.grid_coords['lon'])
        bytes_per_grid = grid_size * 8  # 8 bytes per float64

        # Total grids = time_points * flight_levels * 2 (u and v components)
        total_grids = len(self.time_grids) * len(self.available_flight_levels) * 2

        return (total_grids * bytes_per_grid) / (1024 * 1024)

    def get_wind_fast(self, lat: float, lon: float, alt_fl: int, timestamp: datetime.datetime) -> Tuple[float, float]:
        """
        Fast wind lookup using precomputed grids

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            alt_fl: Flight level (e.g., 350 for FL350)
            timestamp: Time for wind lookup

        Returns:
            Tuple of (u_wind, v_wind) in m/s
        """
        # Check if point is within bounds
        if not (self.spatial_bounds['lat'][0] <= lat <= self.spatial_bounds['lat'][1] and
                self.spatial_bounds['lon'][0] <= lon <= self.spatial_bounds['lon'][1]):
            return np.nan, np.nan

        # Find surrounding time points
        available_times = list(self.time_grids.keys())
        if not available_times:
            return np.nan, np.nan

        # Find closest time points
        time_before = None
        time_after = None

        for time_point in available_times:
            if time_point <= timestamp:
                time_before = time_point
            elif time_point > timestamp and time_after is None:
                time_after = time_point
                break

        if time_before is None and time_after is None:
            return np.nan, np.nan

        # If only one time point available, use it
        if time_before is None:
            time_before = time_after
        elif time_after is None:
            time_after = time_before

        # Get wind at both time points
        u_wind_before, v_wind_before = self._get_wind_at_time_point(lat, lon, alt_fl, time_before)

        if time_before == time_after:
            return u_wind_before, v_wind_before

        u_wind_after, v_wind_after = self._get_wind_at_time_point(lat, lon, alt_fl, time_after)

        # Check for NaN values
        if np.isnan(u_wind_before) or np.isnan(u_wind_after):
            return np.nan, np.nan

        # Temporal interpolation
        time_delta = (time_after - time_before).total_seconds()
        if time_delta == 0:
            return u_wind_before, v_wind_before

        time_ratio = (timestamp - time_before).total_seconds() / time_delta
        time_ratio = np.clip(time_ratio, 0, 1)  # Ensure ratio is between 0 and 1

        u_wind_interp = u_wind_before + time_ratio * (u_wind_after - u_wind_before)
        v_wind_interp = v_wind_before + time_ratio * (v_wind_after - v_wind_before)

        return u_wind_interp, v_wind_interp

    def _get_wind_at_time_point(self, lat: float, lon: float, alt_fl: int, time_point: datetime.datetime) -> Tuple[
        float, float]:
        """Get wind at a specific time point with spatial and altitude interpolation"""

        if time_point not in self.time_grids:
            return np.nan, np.nan

        time_data = self.time_grids[time_point]

        # Find surrounding flight levels
        available_fls = time_data['flight_levels']
        lower_fl = max([fl for fl in available_fls if fl <= alt_fl], default=alt_fl)
        upper_fl = min([fl for fl in available_fls if fl > alt_fl], default=alt_fl)

        # Get wind at lower flight level
        u_lower, v_lower = self._interpolate_spatial_wind(lat, lon, lower_fl, time_data)

        if lower_fl == upper_fl:
            return u_lower, v_lower

        # Get wind at upper flight level
        u_upper, v_upper = self._interpolate_spatial_wind(lat, lon, upper_fl, time_data)

        if np.isnan(u_lower) or np.isnan(u_upper):
            return np.nan, np.nan

        # Altitude interpolation
        alt_ratio = (alt_fl - lower_fl) / (upper_fl - lower_fl)
        alt_ratio = np.clip(alt_ratio, 0, 1)

        u_interp = u_lower + alt_ratio * (u_upper - u_lower)
        v_interp = v_lower + alt_ratio * (v_upper - v_lower)

        return u_interp, v_interp

    def _interpolate_spatial_wind(self, lat: float, lon: float, flight_level: int, time_data: Dict) -> Tuple[
        float, float]:
        """Interpolate wind spatially using precomputed grids"""

        if flight_level not in time_data['u_wind']:
            return np.nan, np.nan

        u_grid = time_data['u_wind'][flight_level]
        v_grid = time_data['v_wind'][flight_level]

        # Create interpolators
        try:
            u_interpolator = RegularGridInterpolator(
                (self.grid_coords['lat'], self.grid_coords['lon']),
                u_grid,
                method='linear',
                bounds_error=False,
                fill_value=np.nan
            )

            v_interpolator = RegularGridInterpolator(
                (self.grid_coords['lat'], self.grid_coords['lon']),
                v_grid,
                method='linear',
                bounds_error=False,
                fill_value=np.nan
            )

            # Interpolate at the point
            point = np.array([[lat, lon]])
            u_wind = u_interpolator(point)[0]
            v_wind = v_interpolator(point)[0]

            return u_wind, v_wind

        except Exception as e:
            print(f"Warning: Spatial interpolation failed: {e}")
            return np.nan, np.nan

    def save_grids(self, filename: str) -> None:
        """Save precomputed grids to file"""
        save_data = {
            'time_grids': self.time_grids,
            'grid_coords': self.grid_coords,
            'spatial_bounds': self.spatial_bounds,
            'available_flight_levels': self.available_flight_levels,
            'grid_resolution': self.grid_resolution
        }

        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"Wind grids saved to {filename}")
        print(f"File size: {os.path.getsize(filename) / (1024 * 1024):.1f} MB")

    def load_grids(self, filename: str) -> None:
        """Load precomputed grids from file"""
        with open(filename, 'rb') as f:
            save_data = pickle.load(f)

        self.time_grids = save_data['time_grids']
        self.grid_coords = save_data['grid_coords']
        self.spatial_bounds = save_data['spatial_bounds']
        self.available_flight_levels = save_data['available_flight_levels']
        self.grid_resolution = save_data['grid_resolution']

        print(f"Wind grids loaded from {filename}")
        print(f"Loaded {len(self.time_grids)} time intervals")
        print(f"Memory usage: ~{self._estimate_memory_usage():.1f} MB")

    def benchmark_performance(self, num_tests: int = 1000) -> None:
        """Benchmark wind lookup performance"""
        import time

        print(f"Benchmarking wind lookup performance with {num_tests} tests...")

        # Generate random test points
        lat_range = self.spatial_bounds['lat']
        lon_range = self.spatial_bounds['lon']

        test_points = []
        for _ in range(num_tests):
            lat = np.random.uniform(lat_range[0], lat_range[1])
            lon = np.random.uniform(lon_range[0], lon_range[1])
            alt_fl = np.random.choice(self.available_flight_levels)
            time_point = np.random.choice(list(self.time_grids.keys()))
            test_points.append((lat, lon, alt_fl, time_point))

        # Benchmark fast lookup
        start_time = time.time()
        for lat, lon, alt_fl, time_point in test_points:
            u_wind, v_wind = self.get_wind_fast(lat, lon, alt_fl, time_point)
        fast_time = time.time() - start_time

        print(f"Fast lookup: {fast_time:.3f} seconds ({fast_time / num_tests * 1000:.3f} ms per lookup)")
        print(f"Throughput: {num_tests / fast_time:.0f} lookups/second")


def optimize_astar_with_wind_grid(start_info, end_info, wind_grid: WindGridOptimizer):
    """
    Optimized A* algorithm using precomputed wind grids
    """
    import heapq
    import datetime

    print("=== OPTIMIZED A* WITH WIND GRIDS ===")
    start_time = datetime.datetime.now()

    # Heuristic function
    def heuristic(lat, lon, end_lat, end_lon):
        from math import sqrt, radians, cos, sin, asin

        # Haversine distance
        R = 6371  # Earth radius in km
        dlat = radians(end_lat - lat)
        dlon = radians(end_lon - lon)
        a = (sin(dlat / 2) ** 2 + cos(radians(lat)) * cos(radians(end_lat)) * sin(dlon / 2) ** 2)
        c = 2 * asin(sqrt(a))
        distance_km = R * c
        distance_nm = distance_km * 0.539957
        return distance_nm / TAS_KNOTS  # Minimum time in hours

    # Initialize start node
    class FastNode:
        def __init__(self, lat, lon, alt, time, parent=None):
            self.lat = lat
            self.lon = lon
            self.alt = alt
            self.time = time
            self.parent = parent
            self.g_cost = 0
            self.h_cost = 0
            self.f_cost = 0

        def __lt__(self, other):
            return self.f_cost < other.f_cost

    start_node = FastNode(start_info['la'], start_info['lo'], start_info['alt'], start_info['t'])
    start_node.h_cost = heuristic(start_node.lat, start_node.lon, end_info['la'], end_info['lo'])
    start_node.f_cost = start_node.h_cost

    open_list = [start_node]
    closed_set = set()
    iterations = 0

    print(f"Search from ({start_node.lat:.2f}, {start_node.lon:.2f}) to ({end_info['la']:.2f}, {end_info['lo']:.2f})")
    print(f"Initial time estimate: {start_node.h_cost:.2f} hours")

    # Possible movements
    movements = [
        (LAT_STEP, 0, 0), (-LAT_STEP, 0, 0),
        (0, LON_STEP, 0), (0, -LON_STEP, 0),
        (LAT_STEP, LON_STEP, 0), (LAT_STEP, -LON_STEP, 0),
        (-LAT_STEP, LON_STEP, 0), (-LAT_STEP, -LON_STEP, 0),
        (0, 0, ALT_STEP), (0, 0, -ALT_STEP)
    ]

    while open_list and iterations < MAX_ITERATIONS:
        iterations += 1
        current = heapq.heappop(open_list)

        # Check if already processed
        node_key = (round(current.lat, 1), round(current.lon, 1), current.alt)
        if node_key in closed_set:
            continue
        closed_set.add(node_key)

        if iterations % 100 == 0:
            elapsed = datetime.datetime.now() - start_time
            print(f"Iteration {iterations:4d} | Queue: {len(open_list):4d} | "
                  f"Position: ({current.lat:.2f}, {current.lon:.2f}) | "
                  f"Time: {elapsed}")

        # Check if goal reached
        from math import sqrt, radians, cos, sin, asin
        R = 6371
        dlat = radians(end_info['la'] - current.lat)
        dlon = radians(end_info['lo'] - current.lon)
        a = (sin(dlat / 2) ** 2 + cos(radians(current.lat)) * cos(radians(end_info['la'])) * sin(dlon / 2) ** 2)
        c = 2 * asin(sqrt(a))
        dist_to_goal = R * c

        if dist_to_goal < 100:  # 100 km threshold
            print(f"✅ Goal reached in {iterations} iterations!")

            # Reconstruct path
            path = []
            while current:
                path.append(current)
                current = current.parent
            path.reverse()

            total_time = datetime.datetime.now() - start_time
            print(f"Search time: {total_time}")
            print(f"Flight time: {path[-1].g_cost:.2f} hours")

            return path

        # Expand neighbors
        for d_lat, d_lon, d_alt in movements:
            new_lat = current.lat + d_lat
            new_lon = current.lon + d_lon
            new_alt = current.alt + d_alt

            # Boundary check
            if not (40 <= new_lat <= 70 and -85 <= new_lon <= -10 and 28000 <= new_alt <= 42000):
                continue

            neighbor_key = (round(new_lat, 1), round(new_lon, 1), new_alt)
            if neighbor_key in closed_set:
                continue

            # Calculate segment distance
            dlat = radians(new_lat - current.lat)
            dlon = radians(new_lon - current.lon)
            a = (sin(dlat / 2) ** 2 + cos(radians(current.lat)) * cos(radians(new_lat)) * sin(dlon / 2) ** 2)
            c = 2 * asin(sqrt(a))
            distance_km = R * c

            if d_alt != 0:
                distance_km += abs(d_alt) * 0.0003048  # Altitude penalty

            if distance_km < 1:
                continue

            # FAST WIND LOOKUP using precomputed grids
            mid_lat = (current.lat + new_lat) / 2
            mid_lon = (current.lon + new_lon) / 2
            mid_alt = (current.alt + new_alt) / 2 / 100  # Convert to FL

            wind_u, wind_v = wind_grid.get_wind_fast(mid_lat, mid_lon, mid_alt, current.time)

            if np.isnan(wind_u):
                continue

            # Wind triangle calculation
            import math
            track_angle = math.degrees(math.atan2(new_lon - current.lon, new_lat - current.lat))

            # Simple ground speed calculation
            tas_ms = TAS_KNOTS * 0.514444
            track_rad = math.radians(track_angle)

            ground_speed_x = tas_ms * math.sin(track_rad) + wind_u
            ground_speed_y = tas_ms * math.cos(track_rad) + wind_v
            ground_speed_ms = math.sqrt(ground_speed_x ** 2 + ground_speed_y ** 2)
            ground_speed_knots = ground_speed_ms / 0.514444

            if ground_speed_knots <= 0:
                continue

            # Calculate time
            distance_nm = distance_km * 0.539957
            segment_time_hours = distance_nm / ground_speed_knots

            # Create neighbor node
            new_time = current.time + datetime.timedelta(hours=segment_time_hours)
            neighbor = FastNode(new_lat, new_lon, new_alt, new_time, current)
            neighbor.g_cost = current.g_cost + segment_time_hours
            neighbor.h_cost = heuristic(new_lat, new_lon, end_info['la'], end_info['lo'])
            neighbor.f_cost = neighbor.g_cost + neighbor.h_cost

            heapq.heappush(open_list, neighbor)

    print(f"❌ Path not found in {iterations} iterations")
    return None


# Example usage and integration with existing code
def main_optimization_with_wind_grid():
    """
    Main function to demonstrate wind grid optimization
    """
    print("=== WIND GRID OPTIMIZATION DEMO ===")

    met_file_path = "./data/ECCC/2025_JUN_par_met.parquet.gzip"


    try:
        df_met = pd.read_parquet(met_file_path)
        print("- MET data loaded.")
    except Exception as e:
        df_met = pd.DataFrame()
        print(f"- MET data loading error: {e}")

    met_fl_cols = sorted([col.split('_')[0] for col in df_met.columns if col.endswith('_speed')], reverse=True)
    met_fl_map = {int(fl.replace('FL', '')): fl for fl in met_fl_cols}


    df_met_processed = df_met.copy()
    forecast_time = pd.to_timedelta(df_met_processed['forecast'].str.slice(0, 2).astype(int), unit='h') + \
                    pd.to_timedelta(df_met_processed['forecast'].str.slice(2, 4).astype(int), unit='m')
    base_date = df_met_processed['date_time'].dt.normalize()
    df_met_processed['valid_forecast_time'] = base_date + forecast_time
    df_met_processed.loc[df_met_processed['observation'] == '1200', 'valid_forecast_time'] += pd.Timedelta(days=1)
    df_met_processed.sort_values('valid_forecast_time', inplace=True)

    # Create wind grid optimizer
    wind_optimizer = WindGridOptimizer(df_met_processed, grid_resolution=0.1)

    # Precompute wind grids
    wind_optimizer.precompute_wind_grids(max_time_intervals=50)  # Limit for demo

    # Benchmark performance
    wind_optimizer.benchmark_performance(num_tests=1000)

    # Save grids for reuse
    wind_optimizer.save_grids("./results/wind_grids_cache.pkl")

    return wind_optimizer


if __name__ == "__main__":
    wind_optimizer = main_optimization_with_wind_grid()