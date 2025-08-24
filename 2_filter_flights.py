import json
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Meteorological data bounds
MET_LAT = [41.25, 68.75]  # [min_lat, max_lat]
MET_LON = [-87.50, -12.50]  # [min_lon, max_lon]


def is_within_met_bounds(lat, lon):
    """
    Check if a point is within the meteorological data bounds.
    """
    return (MET_LAT[0] <= lat <= MET_LAT[1] and
            MET_LON[0] <= lon <= MET_LON[1])


def load_flight_from_parquet(filepath):
    """
    Load flight data directly from a parquet file.
    """
    try:
        return pd.read_parquet(filepath)
    except Exception as e:
        print(f'Failed to load flight from {filepath}: {e}')
        return None


def extract_flight_id_from_filename(filename):
    """
    Extract flight ID from parquet filename.
    Assumes format: flight_<unique_id>.parquet
    """
    if filename.startswith('flight_') and filename.endswith('.parquet'):
        # Remove 'flight_' prefix and '.parquet' suffix
        flight_id = filename[7:-8]  # flight_ = 7 chars, .parquet = 8 chars
        # Convert back from safe filename format
        flight_id = flight_id.replace('_', ' ').replace('-', ':')
        return flight_id
    return filename  # Fallback to filename if format doesn't match


def filter_flight_data_by_met_bounds(loaded_flight):
    """
    Filter flight data to only include points within meteorological bounds.
    Returns filtered dataframe and statistics.
    """
    if loaded_flight is None:
        return None, {}

    original_count = len(loaded_flight)

    # Filter ADS-B data points within met bounds
    if 'la' in loaded_flight.columns and 'lo' in loaded_flight.columns:
        # Create mask for points within met bounds
        within_bounds_mask = (
                (loaded_flight['la'].between(MET_LAT[0], MET_LAT[1])) &
                (loaded_flight['lo'].between(MET_LON[0], MET_LON[1])) &
                (loaded_flight['la'].notna()) &
                (loaded_flight['lo'].notna())
        )

        filtered_flight = loaded_flight[within_bounds_mask].copy()

        # Filter planned waypoints to only include those within bounds
        if 'planned_waypoints' in filtered_flight.columns:
            def filter_waypoints(wp_list):
                if not isinstance(wp_list, (list, np.ndarray)) or len(wp_list) == 0:
                    return wp_list

                filtered_wps = []
                for wp in wp_list:
                    if isinstance(wp, (list, np.ndarray)) and len(wp) >= 2:
                        try:
                            lat, lon = float(wp[0]), float(wp[1])
                            if is_within_met_bounds(lat, lon):
                                filtered_wps.append(wp)
                        except (ValueError, TypeError, IndexError):
                            continue

                return filtered_wps if filtered_wps else None

            filtered_flight['planned_waypoints'] = filtered_flight['planned_waypoints'].apply(filter_waypoints)
    else:
        filtered_flight = pd.DataFrame()  # Empty dataframe if no coordinate columns

    filtered_count = len(filtered_flight)
    removed_count = original_count - filtered_count

    stats = {
        'original_points': original_count,
        'filtered_points': filtered_count,
        'removed_points': removed_count,
        'retention_percentage': (filtered_count / original_count * 100) if original_count > 0 else 0
    }

    return filtered_flight, stats


def extract_flight_data_for_map(loaded_flight, max_adsb_points=100):
    """
    Extract and format flight data for the interactive map.
    Returns properly formatted JSON-serializable data.
    """
    # Extract ADS-B data (actual flight path)
    adsb_data = []
    if 'la' in loaded_flight.columns and 'lo' in loaded_flight.columns:
        # Remove any rows with invalid coordinates
        valid_coords = loaded_flight[
            (loaded_flight['la'].notna()) &
            (loaded_flight['lo'].notna()) &
            (loaded_flight['la'] != 0) &
            (loaded_flight['lo'] != 0) &
            (loaded_flight['la'].between(-90, 90)) &
            (loaded_flight['lo'].between(-180, 180))
            ]

        if len(valid_coords) > max_adsb_points:
            # Sample points evenly across the flight
            indices = np.linspace(0, len(valid_coords) - 1, max_adsb_points, dtype=int)
            valid_coords = valid_coords.iloc[indices]

        if 'alt' in loaded_flight.columns:
            # Include altitude if available, convert to regular Python types
            for _, row in valid_coords.iterrows():
                try:
                    lat = float(row['la'])
                    lon = float(row['lo'])
                    alt = float(row['alt']) if pd.notna(row['alt']) else 0
                    adsb_data.append([lat, lon, alt])
                except (ValueError, TypeError):
                    continue
        else:
            # Just lat/lon
            for _, row in valid_coords.iterrows():
                try:
                    lat = float(row['la'])
                    lon = float(row['lo'])
                    adsb_data.append([lat, lon])
                except (ValueError, TypeError):
                    continue

    # Extract planned waypoints
    waypoints_data = []
    if 'planned_waypoints' in loaded_flight.columns:
        unique_waypoints = set()  # To avoid duplicates

        for wp_list in loaded_flight.planned_waypoints.dropna():
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
                                    waypoints_data.append([lat, lon])
                    except (ValueError, TypeError, IndexError):
                        continue

    return adsb_data, waypoints_data


def validate_and_save_json(data, filename):
    """
    Validate JSON data and save to file if valid.
    """
    try:
        # Test JSON serialization
        json_str = json.dumps(data)

        # Save to file
        with open(filename, 'w') as f:
            f.write(json_str)

        print(f"✓ Data saved to {filename}")
        return True
    except Exception as e:
        print(f"✗ JSON validation failed: {e}")
        return False


def save_filtered_flight(filtered_flight, output_dir, unique_id):
    """
    Save filtered flight data to parquet file.
    """
    if filtered_flight is None or len(filtered_flight) == 0:
        print(f"No data to save for flight {unique_id}")
        return False

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create safe filename
    safe_uid = unique_id.replace(' ', '_').replace(':', '-').replace('/', '-')
    flight_filename = f"flight_{safe_uid}_filtered.parquet"
    flight_path = os.path.join(output_dir, flight_filename)

    try:
        filtered_flight.to_parquet(flight_path, index=False)
        print(f"✓ Filtered flight saved to: {flight_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to save filtered flight: {e}")
        return False


def plot_planned_wps(cell):
    """Plot planned waypoints on matplotlib"""
    waypoints_2d = np.array([wp for wp in cell if len(wp) >= 2])
    if len(waypoints_2d) > 0:
        lats, lons = waypoints_2d[:, 0], waypoints_2d[:, 1]
        plt.scatter(lons, lats, color='green', marker=".", s=2.5)


# Main execution
if __name__ == "__main__":
    print("\n=== FLIGHT DATA FILTERING FOR MET BOUNDS ===")
    print(f"Met bounds: Lat {MET_LAT}, Lon {MET_LON}")

    PROCESSED_FOLDER = './results/processed_flights_CYYZ_LFPG'
    FILTERED_FOLDER = './results/filtered_flights_met_bounds_CYYZ_LFPG'

    # Create filtered output directory
    os.makedirs(FILTERED_FOLDER, exist_ok=True)
    os.makedirs(FILTERED_FOLDER + '/plots', exist_ok=True)
    os.makedirs(FILTERED_FOLDER + '/json_data', exist_ok=True)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    # Get all parquet files in the processed folder
    parquet_files = []
    for filename in os.listdir(PROCESSED_FOLDER):
        if filename.endswith('.parquet'):
            filepath = os.path.join(PROCESSED_FOLDER, filename)
            # Get file size for sorting (optional - you can remove this if you don't need sorting)
            file_size = os.path.getsize(filepath)
            parquet_files.append((filename, filepath, file_size))

    # Sort by file size in descending order (largest files first)
    # Remove this sorting if you want to process files in alphabetical order
    parquet_files.sort(key=lambda x: x[2], reverse=True)

    print(f'Found {len(parquet_files)} parquet files to process')

    # Statistics tracking
    all_stats = []

    # Process each parquet file directly
    for idx, (filename, filepath, file_size) in enumerate(parquet_files):
        print(f"\n--- Processing File {idx + 1}/{len(parquet_files)}: {filename} ---")
        print(f"File size: {file_size / (1024*1024):.2f} MB")

        # Extract flight ID from filename
        flight_id = extract_flight_id_from_filename(filename)
        print(f"Flight ID: {flight_id}")

        # Load flight data directly from parquet file
        loaded_flight = load_flight_from_parquet(filepath)
        if loaded_flight is None:
            print(f"Failed to load flight data from {filepath}")
            continue

        print(f"Loaded {len(loaded_flight)} rows from {filename}")

        # Filter flight data by meteorological bounds
        filtered_flight, stats = filter_flight_data_by_met_bounds(loaded_flight)

        if filtered_flight is None or len(filtered_flight) == 0:
            print(f"No points within met bounds for flight {flight_id}")
            continue

        # Save statistics
        stats['flight_id'] = flight_id
        stats['filename'] = filename
        stats['file_size_mb'] = file_size / (1024*1024)
        all_stats.append(stats)

        print(f"Original points: {stats['original_points']}")
        print(f"Points within met bounds: {stats['filtered_points']}")
        print(f"Points removed: {stats['removed_points']}")
        print(f"Retention: {stats['retention_percentage']:.1f}%")

        # Save filtered flight data
        save_filtered_flight(filtered_flight, FILTERED_FOLDER, flight_id)

        # Extract data for map visualization
        adsb_data, waypoints_data = extract_flight_data_for_map(filtered_flight, max_adsb_points=100)

        # Save JSON data for map tool
        flight_safe_name = flight_id.replace(' ', '_').replace(':', '-').replace('/', '-')
        json_output_dir = FILTERED_FOLDER + '/json_data'

        # Save ADS-B data
        adsb_file = os.path.join(json_output_dir, f"{flight_safe_name}_adsb_filtered.json")
        validate_and_save_json(adsb_data, adsb_file)

        # Save waypoints data
        waypoints_file = os.path.join(json_output_dir, f"{flight_safe_name}_waypoints_filtered.json")
        validate_and_save_json(waypoints_data, waypoints_file)

        # Create complete dataset
        simplified_data = {
            "flight_id": flight_id,
            "filename": filename,
            "adsb_data": adsb_data,
            "waypoints_data": waypoints_data,
            "met_bounds": {
                "lat_range": MET_LAT,
                "lon_range": MET_LON
            },
            "filtering_stats": stats,
            "summary": {
                "total_adsb_points": len(adsb_data),
                "total_waypoints": len(waypoints_data),
                "origin": adsb_data[0][:2] if len(adsb_data) > 0 else None,
                "destination": adsb_data[-1][:2] if len(adsb_data) > 0 else None
            }
        }

        # Save complete dataset
        complete_file = os.path.join(json_output_dir, f"{flight_safe_name}_complete_filtered.json")
        validate_and_save_json(simplified_data, complete_file)

    # Save overall statistics
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_file = os.path.join(FILTERED_FOLDER, 'filtering_statistics.csv')
        stats_df.to_csv(stats_file, index=False)

        print(f"\n=== OVERALL FILTERING STATISTICS ===")
        print(f"Total flights processed: {len(all_stats)}")
        print(f"Average retention rate: {stats_df['retention_percentage'].mean():.1f}%")
        print(f"Total original points: {stats_df['original_points'].sum()}")
        print(f"Total filtered points: {stats_df['filtered_points'].sum()}")
        print(f"Statistics saved to: {stats_file}")

    print(f"\n=== FILTERING COMPLETE ===")
    print(f"Filtered data saved to: {FILTERED_FOLDER}")
    print(f"Met bounds used: Lat {MET_LAT}, Lon {MET_LON}")