"""
Wind Grid Service Module
========================
Consolidated wind grid optimization service for flight path optimization.
Supports both gradient-based and heuristic A* search algorithms.
"""

import pickle
import numpy as np
import pandas as pd
import datetime
import logging
from scipy.interpolate import RegularGridInterpolator
from typing import Dict, Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class WindGridOptimizer:
    """
    Unified Wind Grid Optimizer for fast wind data lookups.
    Supports both gradient and heuristic flight optimization methods.
    """

    def __init__(self, met_data: pd.DataFrame = None, grid_resolution: float = 0.1):
        """
        Initialize the Wind Grid Optimizer.

        Args:
            met_data: Meteorological data DataFrame (optional)
            grid_resolution: Resolution of the grid in degrees (default: 0.1)
        """
        self.met_data = met_data
        self.grid_resolution = grid_resolution
        self.time_grids = {}
        self.spatial_bounds = {}
        self.available_flight_levels = []
        self.grid_coords = {}
        self.times = []  # For index lookup

        if met_data is not None:
            self.spatial_bounds = self._calculate_bounds()
            self.available_flight_levels = self._get_available_flight_levels()
            self.grid_coords = self._create_grid_coordinates()

    def _calculate_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Calculate spatial bounds from meteorological data."""
        return {
            'lat': (self.met_data['latitude_center'].min(), self.met_data['latitude_center'].max()),
            'lon': (self.met_data['longitude_center'].min(), self.met_data['longitude_center'].max())
        }

    def _get_available_flight_levels(self) -> list:
        """Extract available flight levels from meteorological data columns."""
        fl_cols = [col for col in self.met_data.columns if col.endswith('_speed')]
        flight_levels = []

        for col in fl_cols:
            fl_str = col.split('_')[0]
            if fl_str.startswith('FL'):
                fl_number = int(fl_str.replace('FL', ''))
                flight_levels.append(fl_number)

        return sorted(flight_levels)

    def _create_grid_coordinates(self) -> Dict[str, np.ndarray]:
        """Create grid coordinate arrays for interpolation."""
        lat_min, lat_max = self.spatial_bounds['lat']
        lon_min, lon_max = self.spatial_bounds['lon']

        lat_buffer = self.grid_resolution
        lon_buffer = self.grid_resolution

        lat_grid = np.arange(lat_min - lat_buffer, lat_max + lat_buffer + self.grid_resolution, self.grid_resolution)
        lon_grid = np.arange(lon_min - lon_buffer, lon_max + lon_buffer + self.grid_resolution, self.grid_resolution)

        return {
            'lat': lat_grid,
            'lon': lon_grid,
            'fl': np.array(self.available_flight_levels)
        }

    def load_grids(self, filename: str) -> bool:
        """
        Load precomputed wind grids from pickle file.

        Args:
            filename: Path to the pickle file containing wind grids

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(filename, 'rb') as f:
                save_data = pickle.load(f)

            self.time_grids = save_data['time_grids']
            self.grid_coords = save_data['grid_coords']
            self.spatial_bounds = save_data['spatial_bounds']
            self.available_flight_levels = save_data['available_flight_levels']
            self.grid_resolution = save_data['grid_resolution']

            # Extract times for index lookup
            self.times = sorted(list(self.time_grids.keys()))

            logger.info(f"✅ Wind grids loaded from {filename}")
            logger.info(f"Number of time grids: {len(self.time_grids)}")
            logger.info(f"Memory usage: ~{self._estimate_memory_usage():.1f} MB")

            return True

        except Exception as e:
            logger.error(f"Error loading wind grids: {e}")
            return False

    def save_grids(self, filename: str) -> bool:
        """
        Save wind grids to pickle file.

        Args:
            filename: Path to save the pickle file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            save_data = {
                'time_grids': self.time_grids,
                'grid_coords': self.grid_coords,
                'spatial_bounds': self.spatial_bounds,
                'available_flight_levels': self.available_flight_levels,
                'grid_resolution': self.grid_resolution
            }

            with open(filename, 'wb') as f:
                pickle.dump(save_data, f)

            logger.info(f"✅ Wind grids saved to {filename}")
            return True

        except Exception as e:
            logger.error(f"Error saving wind grids: {e}")
            return False

    def get_wind_fast(self, lat: float, lon: float, alt_fl: int,
                      time_input) -> Tuple[float, float]:
        """
        Fast wind lookup using precomputed grids.
        Supports both time index (int) and timestamp (datetime) inputs.

        Args:
            lat: Latitude
            lon: Longitude
            alt_fl: Altitude in flight level (hundreds of feet)
            time_input: Either time index (int) or timestamp (datetime)

        Returns:
            Tuple[float, float]: Wind u and v components (m/s)
        """
        # Check spatial bounds
        if not (self.spatial_bounds['lat'][0] <= lat <= self.spatial_bounds['lat'][1] and
                self.spatial_bounds['lon'][0] <= lon <= self.spatial_bounds['lon'][1]):
            return np.nan, np.nan

        # Handle different time input types
        if isinstance(time_input, int):
            # Time index provided (for gradient optimization)
            time_idx = time_input
            if time_idx < 0 or time_idx >= len(self.times):
                return np.nan, np.nan
            time_point = self.times[time_idx]

        elif isinstance(time_input, (datetime.datetime, pd.Timestamp)):
            # Timestamp provided (for heuristic optimization)
            timestamp = time_input
            time_point = self._find_closest_time_point(timestamp)
            if time_point is None:
                return np.nan, np.nan
        else:
            logger.warning(f"Invalid time input type: {type(time_input)}")
            return np.nan, np.nan

        if time_point not in self.time_grids:
            return np.nan, np.nan

        return self._get_wind_at_time_point(lat, lon, alt_fl, time_point)

    def get_wind_gradient(self, lat: float, lon: float, alt_fl: int, time_idx: int,
                          goal_lat: float, goal_lon: float, delta: float = 0.05) -> np.ndarray:
        """
        Calculate wind gradient for gradient-based optimization.

        Args:
            lat: Current latitude
            lon: Current longitude
            alt_fl: Current altitude in flight level
            time_idx: Current time index
            goal_lat: Goal latitude
            goal_lon: Goal longitude
            delta: Gradient calculation step size

        Returns:
            np.ndarray: Gradient array [grad_lat, grad_lon, grad_alt, temporal_grad]
        """
        import math

        goal_bearing = math.atan2(goal_lon - lon, goal_lat - lat)

        u_current, v_current = self.get_wind_fast(lat, lon, alt_fl, time_idx)
        if np.isnan(u_current):
            return np.array([0, 0, 0, 0])

        grad_lat = self._calculate_spatial_gradient(lat, lon, alt_fl, time_idx, goal_bearing, 'lat', delta)
        grad_lon = self._calculate_spatial_gradient(lat, lon, alt_fl, time_idx, goal_bearing, 'lon', delta)
        grad_alt = self._calculate_spatial_gradient(lat, lon, alt_fl, time_idx, goal_bearing, 'alt', delta)
        temporal_grad = self._calculate_temporal_gradient(lat, lon, alt_fl, time_idx, goal_bearing, 1)

        return np.array([grad_lat, grad_lon, grad_alt, temporal_grad])

    def _calculate_spatial_gradient(self, lat: float, lon: float, alt_fl: int, time_idx: int,
                                    goal_bearing: float, direction: str, delta: float) -> float:
        """Calculate spatial gradient component."""
        import math

        if direction == 'lat':
            u_plus, v_plus = self.get_wind_fast(lat + delta, lon, alt_fl, time_idx)
            u_minus, v_minus = self.get_wind_fast(lat - delta, lon, alt_fl, time_idx)
        elif direction == 'lon':
            u_plus, v_plus = self.get_wind_fast(lat, lon + delta, alt_fl, time_idx)
            u_minus, v_minus = self.get_wind_fast(lat, lon - delta, alt_fl, time_idx)
        elif direction == 'alt':
            alt_plus = min(alt_fl + 20, max(self.available_flight_levels))
            alt_minus = max(alt_fl - 20, min(self.available_flight_levels))
            u_plus, v_plus = self.get_wind_fast(lat, lon, alt_plus, time_idx)
            u_minus, v_minus = self.get_wind_fast(lat, lon, alt_minus, time_idx)
            delta = (alt_plus - alt_minus) * 100
        else:
            return 0

        if np.isnan(u_plus) or np.isnan(u_minus):
            return 0

        headwind_plus = u_plus * math.sin(goal_bearing) + v_plus * math.cos(goal_bearing)
        headwind_minus = u_minus * math.sin(goal_bearing) + v_minus * math.cos(goal_bearing)

        if delta != 0:
            return (headwind_plus - headwind_minus) / (2 * delta)
        return 0

    def _calculate_temporal_gradient(self, lat: float, lon: float, alt_fl: int, time_idx: int,
                                     goal_bearing: float, delta_idx: int = 1) -> float:
        """Calculate temporal gradient component."""
        import math

        u_future, v_future = self.get_wind_fast(lat, lon, alt_fl, time_idx + delta_idx)
        u_past, v_past = self.get_wind_fast(lat, lon, alt_fl, time_idx - delta_idx)

        if np.isnan(u_future) or np.isnan(u_past):
            return 0

        headwind_future = u_future * math.sin(goal_bearing) + v_future * math.cos(goal_bearing)
        headwind_past = u_past * math.sin(goal_bearing) + v_past * math.cos(goal_bearing)

        return (headwind_future - headwind_past) / (2 * delta_idx)

    def _find_closest_time_point(self, timestamp: datetime.datetime) -> Optional[datetime.datetime]:
        """Find the closest time point in the grid for a given timestamp."""
        if not self.times:
            return None

        # For temporal interpolation between grid points
        time_before = None
        time_after = None

        for time_point in self.times:
            if time_point <= timestamp:
                time_before = time_point
            elif time_point > timestamp and time_after is None:
                time_after = time_point
                break

        if time_before is None and time_after is None:
            return None

        if time_before is None:
            return time_after
        elif time_after is None:
            return time_before

        # Return the closer time point
        delta_before = abs((timestamp - time_before).total_seconds())
        delta_after = abs((time_after - timestamp).total_seconds())

        return time_before if delta_before <= delta_after else time_after

    def _get_wind_at_time_point(self, lat: float, lon: float, alt_fl: int,
                                time_point: datetime.datetime) -> Tuple[float, float]:
        """Get wind at a specific time point with spatial and altitude interpolation."""
        if time_point not in self.time_grids:
            return np.nan, np.nan

        time_data = self.time_grids[time_point]
        available_fls = time_data['flight_levels']

        # Find bracketing flight levels
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

        # Interpolate between flight levels
        alt_ratio = (alt_fl - lower_fl) / (upper_fl - lower_fl)
        alt_ratio = np.clip(alt_ratio, 0, 1)

        u_interp = u_lower + alt_ratio * (u_upper - u_lower)
        v_interp = v_lower + alt_ratio * (v_upper - v_lower)

        return u_interp, v_interp

    def _interpolate_spatial_wind(self, lat: float, lon: float, flight_level: int,
                                  time_data: Dict) -> Tuple[float, float]:
        """Interpolate wind spatially using precomputed grids."""
        if flight_level not in time_data['u_wind']:
            return np.nan, np.nan

        u_grid = time_data['u_wind'][flight_level]
        v_grid = time_data['v_wind'][flight_level]

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

            point = np.array([[lat, lon]])
            u_wind = u_interpolator(point)[0]
            v_wind = v_interpolator(point)[0]

            return u_wind, v_wind

        except Exception as e:
            logger.warning(f"Spatial interpolation failed: {e}")
            return np.nan, np.nan

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of loaded grids in MB."""
        if not self.time_grids:
            return 0

        grid_size = len(self.grid_coords['lat']) * len(self.grid_coords['lon'])
        bytes_per_grid = grid_size * 8  # 8 bytes per float64
        total_grids = len(self.time_grids) * len(self.available_flight_levels) * 2  # u and v components

        return (total_grids * bytes_per_grid) / (1024 * 1024)

    def get_temporal_interpolated_wind(self, lat: float, lon: float, alt_fl: int,
                                       timestamp: datetime.datetime) -> Tuple[float, float]:
        """
        Get wind with full temporal interpolation between grid points.
        Used for high-precision applications.

        Args:
            lat: Latitude
            lon: Longitude
            alt_fl: Altitude in flight level
            timestamp: Exact timestamp

        Returns:
            Tuple[float, float]: Temporally interpolated wind components
        """
        if not self.times:
            return np.nan, np.nan

        # Find bracketing time points
        time_before = None
        time_after = None

        for time_point in self.times:
            if time_point <= timestamp:
                time_before = time_point
            elif time_point > timestamp and time_after is None:
                time_after = time_point
                break

        if time_before is None and time_after is None:
            return np.nan, np.nan

        if time_before is None:
            time_before = time_after
        elif time_after is None:
            time_after = time_before

        # Get wind at both time points
        u_before, v_before = self._get_wind_at_time_point(lat, lon, alt_fl, time_before)

        if time_before == time_after:
            return u_before, v_before

        u_after, v_after = self._get_wind_at_time_point(lat, lon, alt_fl, time_after)

        if np.isnan(u_before) or np.isnan(u_after):
            return np.nan, np.nan

        # Temporal interpolation
        time_delta = (time_after - time_before).total_seconds()
        if time_delta == 0:
            return u_before, v_before

        time_ratio = (timestamp - time_before).total_seconds() / time_delta
        time_ratio = np.clip(time_ratio, 0, 1)

        u_interp = u_before + time_ratio * (u_after - u_before)
        v_interp = v_before + time_ratio * (v_after - v_before)

        return u_interp, v_interp

    def get_grid_info(self) -> Dict:
        """Get information about the loaded wind grid."""
        return {
            'grid_resolution': self.grid_resolution,
            'spatial_bounds': self.spatial_bounds,
            'available_flight_levels': self.available_flight_levels,
            'num_time_points': len(self.times),
            'time_range': {
                'start': self.times[0] if self.times else None,
                'end': self.times[-1] if self.times else None
            },
            'grid_dimensions': {
                'lat_points': len(self.grid_coords.get('lat', [])),
                'lon_points': len(self.grid_coords.get('lon', [])),
                'fl_points': len(self.available_flight_levels)
            },
            'estimated_memory_mb': self._estimate_memory_usage()
        }


# Backward compatibility aliases
WindGridOptimizerGradient = WindGridOptimizer
WindGridOptimizerHeuristic = WindGridOptimizer


def create_wind_grid_optimizer(met_data: pd.DataFrame = None,
                               grid_resolution: float = 0.1) -> WindGridOptimizer:
    """
    Factory function to create a WindGridOptimizer instance.

    Args:
        met_data: Meteorological data DataFrame (optional)
        grid_resolution: Resolution of the grid in degrees

    Returns:
        WindGridOptimizer: Configured wind grid optimizer instance
    """
    return WindGridOptimizer(met_data, grid_resolution)


def load_wind_grid_from_file(filename: str) -> Optional[WindGridOptimizer]:
    """
    Load a wind grid optimizer from a saved file.

    Args:
        filename: Path to the saved wind grid file

    Returns:
        WindGridOptimizer: Loaded optimizer instance, or None if failed
    """
    try:
        optimizer = WindGridOptimizer()
        if optimizer.load_grids(filename):
            return optimizer
        return None
    except Exception as e:
        logger.error(f"Failed to load wind grid from {filename}: {e}")
        return None