#!/usr/bin/env python3
"""
Flight Efficiency Analysis with Wind Benefits

This module adds efficiency calculations including:
1. Wind speed benefits analysis
2. Great circle distance calculations for all paths
3. Efficiency ratios and comparative metrics
4. Detailed wind impact analysis
"""

import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# =============================================
# ENHANCED EFFICIENCY CALCULATION FUNCTIONS
# =============================================

def calculate_wind_component(wind_u: float, wind_v: float, track_angle_deg: float) -> Tuple[float, float]:
    """
    Calculate headwind/tailwind and crosswind components

    Args:
        wind_u: East-west wind component (m/s)
        wind_v: North-south wind component (m/s)
        track_angle_deg: Track angle in degrees (0 = North, 90 = East)

    Returns:
        Tuple of (tailwind_component, crosswind_component) in m/s
        Positive tailwind = helping, negative = headwind
    """
    if np.isnan(wind_u) or np.isnan(wind_v):
        return 0.0, 0.0

    # Convert track to radians
    track_rad = math.radians(track_angle_deg)

    # Track unit vector
    track_x = math.sin(track_rad)  # East component
    track_y = math.cos(track_rad)  # North component

    # Wind vector components
    wind_along_track = wind_u * track_x + wind_v * track_y  # Tailwind component
    wind_cross_track = wind_u * track_y - wind_v * track_x  # Crosswind component

    return wind_along_track, wind_cross_track


def calculate_wind_speed_benefit(wind_u: float, wind_v: float, track_angle_deg: float) -> Dict[str, float]:
    """
    Calculate comprehensive wind speed benefits

    Returns:
        Dictionary with wind analysis including speed benefit in knots
    """
    tailwind, crosswind = calculate_wind_component(wind_u, wind_v, track_angle_deg)

    # Convert to knots (1 m/s = 1.94384 knots)
    tailwind_knots = tailwind * 1.94384
    crosswind_knots = crosswind * 1.94384
    total_wind_knots = math.sqrt(wind_u ** 2 + wind_v ** 2) * 1.94384

    return {
        'tailwind_ms': tailwind,
        'crosswind_ms': crosswind,
        'tailwind_knots': tailwind_knots,
        'crosswind_knots': crosswind_knots,
        'total_wind_speed_knots': total_wind_knots,
        'wind_direction_deg': math.degrees(math.atan2(wind_u, wind_v)) % 360,
        'is_tailwind': bool(tailwind > 0),
        'wind_benefit_knots': tailwind_knots  # Primary benefit metric
    }


def calculate_great_circle_distance_and_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[
    float, float]:
    """
    Calculate great circle distance and initial bearing between two points

    Returns:
        Tuple of (distance_km, initial_bearing_deg)
    """
    # Convert to radians
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

    # Calculate distance
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    distance_km = 6371 * c  # Earth radius in km

    # Calculate initial bearing
    y = math.sin(dlon) * math.cos(lat2_rad)
    x = (math.cos(lat1_rad) * math.sin(lat2_rad) -
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon))
    bearing_rad = math.atan2(y, x)
    bearing_deg = (math.degrees(bearing_rad) + 360) % 360

    return distance_km, bearing_deg


def calculate_path_efficiency_metrics(path_points: List[List], path_type: str = "unknown") -> Dict[str, any]:
    """
    Calculate comprehensive efficiency metrics for any flight path

    Args:
        path_points: List of [lat, lon, alt, ...] points
        path_type: String identifier for the path type

    Returns:
        Dictionary with efficiency metrics
    """
    if not path_points or len(path_points) < 2:
        return {
            'path_type': path_type,
            'has_data': False,
            'error': 'Insufficient points for analysis'
        }

    try:
        # Calculate total path distance
        total_distance_km = 0.0
        segments = []

        for i in range(1, len(path_points)):
            lat1, lon1 = path_points[i - 1][0], path_points[i - 1][1]
            lat2, lon2 = path_points[i][0], path_points[i][1]

            segment_distance, segment_bearing = calculate_great_circle_distance_and_bearing(lat1, lon1, lat2, lon2)
            total_distance_km += segment_distance

            segments.append({
                'distance_km': segment_distance,
                'bearing_deg': segment_bearing,
                'from_lat': lat1, 'from_lon': lon1,
                'to_lat': lat2, 'to_lon': lon2
            })

        # Calculate great circle distance (direct route)
        origin_lat, origin_lon = path_points[0][0], path_points[0][1]
        dest_lat, dest_lon = path_points[-1][0], path_points[-1][1]
        great_circle_distance_km, great_circle_bearing = calculate_great_circle_distance_and_bearing(
            origin_lat, origin_lon, dest_lat, dest_lon
        )

        # Calculate efficiency metrics
        efficiency_ratio = great_circle_distance_km / total_distance_km if total_distance_km > 0 else 0
        distance_overhead_km = total_distance_km - great_circle_distance_km
        distance_overhead_percent = (
                    distance_overhead_km / great_circle_distance_km * 100) if great_circle_distance_km > 0 else 0

        # Convert to nautical miles
        total_distance_nm = total_distance_km * 0.539957
        great_circle_distance_nm = great_circle_distance_km * 0.539957
        distance_overhead_nm = distance_overhead_km * 0.539957

        # Calculate flight time if available (for optimized routes)
        flight_time_hours = None
        if len(path_points[0]) > 3 and len(path_points[-1]) > 3:
            try:
                if isinstance(path_points[-1][3], (int, float)):
                    flight_time_hours = path_points[-1][3]  # Assuming time is in hours
                elif hasattr(path_points[-1][3], 'total_seconds'):
                    flight_time_hours = path_points[-1][3].total_seconds() / 3600
            except:
                pass

        return {
            'path_type': path_type,
            'has_data': True,
            'total_distance_km': total_distance_km,
            'total_distance_nm': total_distance_nm,
            'great_circle_distance_km': great_circle_distance_km,
            'great_circle_distance_nm': great_circle_distance_nm,
            'efficiency_ratio': efficiency_ratio,
            'efficiency_percent': efficiency_ratio * 100,
            'distance_overhead_km': distance_overhead_km,
            'distance_overhead_nm': distance_overhead_nm,
            'distance_overhead_percent': distance_overhead_percent,
            'great_circle_bearing_deg': great_circle_bearing,
            'flight_time_hours': flight_time_hours,
            'segments': segments,
            'num_waypoints': len(path_points)
        }

    except Exception as e:
        logger.error(f"Error calculating path efficiency for {path_type}: {e}")
        return {
            'path_type': path_type,
            'has_data': False,
            'error': str(e)
        }


def analyze_wind_benefits_along_path(path_points: List[List], wind_grid, tas_knots: float = 450.0) -> Dict[str, any]:
    """
    Analyze wind benefits along the entire flight path

    Args:
        path_points: List of [lat, lon, alt, time] points
        wind_grid: Wind grid optimizer object
        tas_knots: True airspeed in knots

    Returns:
        Dictionary with comprehensive wind analysis
    """
    if not path_points or len(path_points) < 2 or wind_grid is None:
        return {
            'has_data': False,
            'error': 'Insufficient data for wind analysis'
        }

    try:
        segment_analyses = []
        total_wind_benefit_knots = 0.0
        total_distance_km = 0.0
        total_time_hours = 0.0

        for i in range(1, len(path_points)):
            # Get segment points
            p1, p2 = path_points[i - 1], path_points[i]
            lat1, lon1, alt1 = p1[0], p1[1], p1[2]
            lat2, lon2, alt2 = p2[0], p2[1], p2[2]

            # Calculate segment properties
            segment_distance, track_bearing = calculate_great_circle_distance_and_bearing(lat1, lon1, lat2, lon2)

            # Get wind at midpoint of segment
            mid_lat = (lat1 + lat2) / 2
            mid_lon = (lon1 + lon2) / 2
            mid_alt_fl = (alt1 + alt2) / 2 / 100  # Convert feet to flight level

            # Get time for wind lookup
            segment_time = p1[3] if len(p1) > 3 else datetime.now()
            if isinstance(segment_time, (int, float)):
                # If time is in hours, create datetime
                base_time = datetime.now()
                segment_time = base_time + timedelta(hours=segment_time)

            # Get wind data
            wind_u, wind_v = wind_grid.get_wind_fast(mid_lat, mid_lon, mid_alt_fl, segment_time)

            if not np.isnan(wind_u) and not np.isnan(wind_v):
                # Calculate wind benefits
                wind_analysis = calculate_wind_speed_benefit(wind_u, wind_v, track_bearing)

                # Calculate ground speed with wind
                ground_speed_knots = wind_triangle_calculation(tas_knots, wind_u, wind_v, track_bearing)

                # Calculate time for this segment
                segment_distance_nm = segment_distance * 0.539957
                segment_time_hours = segment_distance_nm / ground_speed_knots if ground_speed_knots > 0 else 0

                segment_analysis = {
                    'segment_index': i,
                    'distance_km': segment_distance,
                    'distance_nm': segment_distance_nm,
                    'track_bearing_deg': track_bearing,
                    'wind_u_ms': wind_u,
                    'wind_v_ms': wind_v,
                    'wind_benefit_knots': wind_analysis['wind_benefit_knots'],
                    'tailwind_knots': wind_analysis['tailwind_knots'],
                    'crosswind_knots': wind_analysis['crosswind_knots'],
                    'total_wind_speed_knots': wind_analysis['total_wind_speed_knots'],
                    'wind_direction_deg': wind_analysis['wind_direction_deg'],
                    'ground_speed_knots': ground_speed_knots,
                    'segment_time_hours': segment_time_hours,
                    'is_tailwind': wind_analysis['is_tailwind']
                }

                # Weight wind benefit by segment distance
                weighted_wind_benefit = wind_analysis['wind_benefit_knots'] * segment_distance_nm
                total_wind_benefit_knots += weighted_wind_benefit
                total_distance_km += segment_distance
                total_time_hours += segment_time_hours

                segment_analyses.append(segment_analysis)

        # Calculate overall wind benefit (distance-weighted average)
        total_distance_nm = total_distance_km * 0.539957
        average_wind_benefit_knots = total_wind_benefit_knots / total_distance_nm if total_distance_nm > 0 else 0

        # Calculate statistics
        if segment_analyses:
            wind_benefits = [seg['wind_benefit_knots'] for seg in segment_analyses]
            tailwind_segments = [seg for seg in segment_analyses if seg['is_tailwind']]
            headwind_segments = [seg for seg in segment_analyses if not seg['is_tailwind']]

            stats = {
                'average_wind_benefit_knots': average_wind_benefit_knots,
                'max_wind_benefit_knots': max(wind_benefits),
                'min_wind_benefit_knots': min(wind_benefits),
                'tailwind_segments_count': len(tailwind_segments),
                'headwind_segments_count': len(headwind_segments),
                'tailwind_percentage': len(tailwind_segments) / len(segment_analyses) * 100,
                'total_flight_time_hours': total_time_hours,
                'average_ground_speed_knots': total_distance_nm / total_time_hours if total_time_hours > 0 else 0
            }
        else:
            stats = {
                'average_wind_benefit_knots': 0,
                'max_wind_benefit_knots': 0,
                'min_wind_benefit_knots': 0,
                'tailwind_segments_count': 0,
                'headwind_segments_count': 0,
                'tailwind_percentage': 0,
                'total_flight_time_hours': 0,
                'average_ground_speed_knots': 0
            }

        return {
            'has_data': True,
            'segment_analyses': segment_analyses,
            'wind_statistics': stats,
            'total_distance_km': total_distance_km,
            'total_distance_nm': total_distance_nm
        }

    except Exception as e:
        logger.error(f"Error analyzing wind benefits: {e}")
        return {
            'has_data': False,
            'error': str(e)
        }


def wind_triangle_calculation(tas_knots: float, wind_u: float, wind_v: float, track_angle_deg: float) -> float:
    """
    Calculate ground speed using wind triangle (existing function - keeping for compatibility)
    """
    tas_ms = tas_knots * 0.514444
    track_rad = math.radians(track_angle_deg)

    desired_track_x = math.sin(track_rad)
    desired_track_y = math.cos(track_rad)

    ground_speed_x = tas_ms * desired_track_x + wind_u
    ground_speed_y = tas_ms * desired_track_y + wind_v

    ground_speed_ms = math.sqrt(ground_speed_x ** 2 + ground_speed_y ** 2)
    ground_speed_knots = ground_speed_ms / 0.514444

    return ground_speed_knots


def calculate_comprehensive_flight_analysis(actual_points: List[List],
                                            optimized_route: List[List],
                                            planned_waypoints: List[List],
                                            wind_grid,
                                            tas_knots: float = 450.0) -> Dict[str, any]:
    """
    Calculate comprehensive analysis comparing all flight paths

    Args:
        actual_points: Actual flight path [lat, lon, alt, time]
        optimized_route: Optimized route [lat, lon, alt, time]
        planned_waypoints: Planned waypoints [lat, lon]
        wind_grid: Wind grid for analysis
        tas_knots: True airspeed in knots

    Returns:
        Dictionary with comprehensive flight analysis
    """

    analysis = {
        'timestamp': datetime.now().isoformat(),
        'analysis_parameters': {
            'tas_knots': tas_knots
        }
    }

    # Analyze actual flight path
    logger.info("Analyzing actual flight path...")
    analysis['actual_flight'] = calculate_path_efficiency_metrics(actual_points, "actual_flight")
    analysis['actual_flight_wind'] = analyze_wind_benefits_along_path(actual_points, wind_grid, tas_knots)

    # Analyze optimized route
    logger.info("Analyzing optimized route...")
    analysis['optimized_route'] = calculate_path_efficiency_metrics(optimized_route, "optimized_route")
    analysis['optimized_route_wind'] = analyze_wind_benefits_along_path(optimized_route, wind_grid, tas_knots)

    # Analyze planned waypoints (if available)
    if planned_waypoints and len(planned_waypoints) >= 2:
        logger.info("Analyzing planned waypoints...")
        analysis['planned_route'] = calculate_path_efficiency_metrics(planned_waypoints, "planned_route")
        # Note: Can't analyze wind for planned waypoints without altitude/time data

    # Calculate great circle reference
    if actual_points and len(actual_points) >= 2:
        origin = actual_points[0]
        destination = actual_points[-1]
        great_circle_distance_km, great_circle_bearing = calculate_great_circle_distance_and_bearing(
            origin[0], origin[1], destination[0], destination[1]
        )

        analysis['great_circle_reference'] = {
            'distance_km': great_circle_distance_km,
            'distance_nm': great_circle_distance_km * 0.539957,
            'bearing_deg': great_circle_bearing,
            'theoretical_time_hours': (great_circle_distance_km * 0.539957) / tas_knots
        }

    # Calculate comparative metrics
    try:
        if (analysis['actual_flight']['has_data'] and
                analysis['optimized_route']['has_data']):
            actual_distance = analysis['actual_flight']['total_distance_km']
            optimized_distance = analysis['optimized_route']['total_distance_km']
            great_circle_distance = analysis['great_circle_reference']['distance_km']

            actual_time = analysis['actual_flight_wind']['wind_statistics']['total_flight_time_hours']
            optimized_time = analysis['optimized_route_wind']['wind_statistics']['total_flight_time_hours']

            analysis['comparative_metrics'] = {
                'distance_savings_km': actual_distance - optimized_distance,
                'distance_savings_nm': (actual_distance - optimized_distance) * 0.539957,
                'distance_savings_percent': ((
                                                         actual_distance - optimized_distance) / actual_distance * 100) if actual_distance > 0 else 0,
                'time_savings_hours': actual_time - optimized_time if actual_time and optimized_time else None,
                'time_savings_minutes': (actual_time - optimized_time) * 60 if actual_time and optimized_time else None,
                'time_savings_percent': ((
                                                     actual_time - optimized_time) / actual_time * 100) if actual_time and optimized_time and actual_time > 0 else None,
                'optimized_vs_great_circle_efficiency': optimized_distance / great_circle_distance if great_circle_distance > 0 else None,
                'actual_vs_great_circle_efficiency': actual_distance / great_circle_distance if great_circle_distance > 0 else None,
                'wind_benefit_improvement_knots': (
                        analysis['optimized_route_wind']['wind_statistics']['average_wind_benefit_knots'] -
                        analysis['actual_flight_wind']['wind_statistics']['average_wind_benefit_knots']
                ) if (analysis['optimized_route_wind']['has_data'] and analysis['actual_flight_wind'][
                    'has_data']) else None
            }

    except Exception as e:
        logger.error(f"Error calculating comparative metrics: {e}")
        analysis['comparative_metrics'] = {'error': str(e)}

    return analysis


# =============================================
# INTEGRATION FUNCTIONS FOR EXISTING CODE
# =============================================

def enhance_flight_data_response(response: Dict, wind_grid) -> Dict:
    """
    Enhance the existing flight data response with comprehensive efficiency analysis

    This function can be called from your existing get_flight_data() route
    to add the new efficiency metrics without breaking existing functionality.
    """

    try:
        flight_data = response.get('flight_data', {})
        actual_points = flight_data.get('actual_points', [])
        optimized_route = flight_data.get('optimized_route', [])
        planned_waypoints = flight_data.get('planned_waypoints', [])

        if actual_points or optimized_route:
            logger.info("Calculating comprehensive flight efficiency analysis...")

            # Calculate comprehensive analysis
            efficiency_analysis = calculate_comprehensive_flight_analysis(
                actual_points, optimized_route, planned_waypoints, wind_grid
            )

            # Add to response
            response['efficiency_analysis'] = efficiency_analysis

            # Also update individual metrics for backward compatibility
            if efficiency_analysis.get('actual_flight', {}).get('has_data'):
                flight_data['actual_flight_efficiency'] = efficiency_analysis['actual_flight']

            if efficiency_analysis.get('optimized_route', {}).get('has_data'):
                flight_data['optimized_route_efficiency'] = efficiency_analysis['optimized_route']

            if efficiency_analysis.get('comparative_metrics'):
                flight_data['comparative_metrics'] = efficiency_analysis['comparative_metrics']

            logger.info("Enhanced flight data with comprehensive efficiency analysis")

    except Exception as e:
        logger.error(f"Error enhancing flight data with efficiency analysis: {e}")
        response['efficiency_analysis_error'] = str(e)

    return response


def generate_great_circle_waypoints(lat1: float, lon1: float, lat2: float, lon2: float,
                                    num_points: int = 50) -> List[List[float]]:
    """
    Generate waypoints along the great circle path

    Returns:
        List of [lat, lon] waypoints along the great circle
    """
    waypoints = []

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
            A = math.sin((1 - f) * d) / math.sin(d) if math.sin(d) != 0 else 1 - f
            B = math.sin(f * d) / math.sin(d) if math.sin(d) != 0 else f

            x = A * math.cos(lat1_rad) * math.cos(lon1_rad) + B * math.cos(lat2_rad) * math.cos(lon2_rad)
            y = A * math.cos(lat1_rad) * math.sin(lon1_rad) + B * math.cos(lat2_rad) * math.sin(lon2_rad)
            z = A * math.sin(lat1_rad) + B * math.sin(lat2_rad)

            # Convert back to lat/lon
            lat = math.atan2(z, math.sqrt(x ** 2 + y ** 2))
            lon = math.atan2(y, x)

            # Convert to degrees and add to waypoints
            waypoints.append([math.degrees(lat), math.degrees(lon)])

    except Exception as e:
        logger.error(f"Error generating great circle waypoints: {e}")
        # Fallback to simple linear interpolation
        for i in range(num_points + 1):
            f = i / num_points
            lat = lat1 + f * (lat2 - lat1)
            lon = lon1 + f * (lon2 - lon1)
            waypoints.append([lat, lon])

    return waypoints