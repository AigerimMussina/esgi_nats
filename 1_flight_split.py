"""
+++++ Flight Split Script +++++

This script processes data parquet files one by one and creates separate files
for each unique flight to avoid memory issues. Each flight is saved as a
separate parquet file for efficient storage and retrieval.

Use flights between two airports to minimize processing time.
"""

import pandas as pd
import numpy as np
import os
import gc


def format_flight_data(df_data):
    """
    Format flight data by applying the same transformations as the original script.
    """
    print('  - Removing non-ADS-B datapoints...')
    df_data = df_data[df_data['so'] == 'ADSB']

    if len(df_data) == 0:
        return df_data

    print('  - Implementing datetime format...')
    df_data.loc[:, 't'] = pd.to_datetime(df_data['t'])
    df_data = df_data.sort_values('t')

    df_data = df_data[((df_data['aporgic']==ORIGIN_AIRPORT) & (df_data['apdstic']==DESTINATION_AIRPORT)) |
                          ((df_data['apdstic']==ORIGIN_AIRPORT) & (df_data['aporgic']==DESTINATION_AIRPORT))]

    print('  - Reformatting planned waypoints...')
    def get_waypoints_wp(cell):
        if not isinstance(cell, str):
            return cell
        list_coords_float = []
        list_pairs = cell.split(";")
        filtered_list_pairs = [item for item in list_pairs if item != ""]
        for pair in filtered_list_pairs:
            try:
                list_coords = list(np.float16(pair.split(",")))
                list_coords_float.append(list_coords)
            except ValueError:
                continue
        return list_coords_float

    df_data['planned_waypoints'] = df_data['wp'].apply(lambda x: get_waypoints_wp(x) if isinstance(x, str) else x)

    print('  - Generating unique flight identifiers...')
    df_data['unique_id'] = (df_data['acr'] + ' ' + df_data['cs'] + ' ' +
                           df_data['aporgic'] + ' ' + df_data['apdstic'] + ' ' +
                           df_data['depau'])

    print('  - Replacing NoneType cells with NaN...')
    df_data = df_data.fillna(value=np.nan)

    print('  - Deleting unnecessary columns...')
    df_data.drop(columns=['wp'], inplace=True)

    return df_data


def process_single_file(file_path, output_dir, processed_flights=None):
    """
    Process a single parquet file and extract individual flights.
    If a flight already exists, merge the new data with existing data.

    Args:
        file_path: Path to the input parquet file
        output_dir: Directory to save individual flight files
        processed_flights: Set of already processed flight IDs to track progress

    Returns:
        Set of unique flight IDs found in this file
    """
    if processed_flights is None:
        processed_flights = set()

    print(f'Processing file: {file_path}')

    try:
        # Load the file
        df_data = pd.read_parquet(file_path)
        print(f'  Loaded {len(df_data)} rows')

        # Format the data
        df_data = format_flight_data(df_data)

        if len(df_data) == 0:
            print('  No valid data after filtering')
            return processed_flights

        # Group by unique flight ID
        print('  - Grouping data by unique ID...')
        gb_uid = df_data.groupby('unique_id')

        current_file_flights = set()

        # Process each unique flight
        for uid, new_flight_data in gb_uid:
            current_file_flights.add(uid)

            # Create filename-safe version of unique ID
            safe_uid = uid.replace(' ', '_').replace(':', '-').replace('/', '-')
            flight_filename = f"flight_{safe_uid}.parquet"
            flight_path = os.path.join(output_dir, flight_filename)

            # Check if flight already exists
            if os.path.exists(flight_path):
                print(f'  - Merging additional data for existing flight: {uid}')

                # Load existing flight data
                existing_flight_data = pd.read_parquet(flight_path)

                # Combine new and existing data
                combined_flight_data = pd.concat([existing_flight_data, new_flight_data])

                # Remove duplicates based on timestamp and position
                # (in case the same data point appears in multiple files)
                combined_flight_data = combined_flight_data.drop_duplicates(
                    subset=['t', 'lo', 'la'], keep='first'
                )

                # Sort by timestamp
                combined_flight_data = combined_flight_data.sort_values('t').reset_index(drop=True)

                # Save merged data
                combined_flight_data.to_parquet(flight_path, compression='gzip')
                print(f'  - Updated flight {uid}: {len(existing_flight_data)} + {len(new_flight_data)} -> {len(combined_flight_data)} rows')

                flight_data_for_summary = combined_flight_data

            else:
                print(f'  - Creating new flight file: {uid}')

                # Save new flight data
                new_flight_data.to_parquet(flight_path, compression='gzip')
                print(f'  - Saved flight {uid} ({len(new_flight_data)} rows) to {flight_filename}')

                flight_data_for_summary = new_flight_data

            # Get the row where planned_waypoints has the maximum length
            try:
                longest_row = flight_data_for_summary.loc[
                    flight_data_for_summary['planned_waypoints'].str.len().idxmax()
                ]
                nbr_rows_with_wps = len(longest_row['planned_waypoints'])
            except:
                longest_row = flight_data_for_summary.iloc[0]
                nbr_rows_with_wps = 0

            # Create/update summary info for this flight
            summary_info = {
                'unique_id': uid,
                'fnia': flight_data_for_summary.aporgic.iloc[0] if 'fnia' in flight_data_for_summary.columns else None,
                'nbr_rows': len(flight_data_for_summary),
                'nbr_rows_with_wps': nbr_rows_with_wps,
                'departure_time': flight_data_for_summary.depau.iloc[0] if 'depau' in flight_data_for_summary.columns else None,
                'arrival_time': flight_data_for_summary.arrau.iloc[-1] if 'arrau' in flight_data_for_summary.columns else None,
                'last_timestamp': flight_data_for_summary.t.iloc[-1],
                'first_timestamp': flight_data_for_summary.t.iloc[0],
                'origin_airport': flight_data_for_summary.aporgic.iloc[0] if 'aporgic' in flight_data_for_summary.columns else None,
                'destination_airport': flight_data_for_summary.apdstic.iloc[0] if 'apdstic' in flight_data_for_summary.columns else None,
                'aircraft_registration': flight_data_for_summary.acr.iloc[0] if 'acr' in flight_data_for_summary.columns else None,
                'callsign': flight_data_for_summary.cs.iloc[0] if 'cs' in flight_data_for_summary.columns else None,
                'file_path': flight_path,
                'last_updated': pd.Timestamp.now().isoformat()
            }

            # Save summary info
            summary_filename = f"flight_{safe_uid}_summary.json"
            summary_path = os.path.join(output_dir, "summaries", summary_filename)
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)

            import json
            with open(summary_path, 'w') as f:
                json.dump(summary_info, f, indent=2, default=str)

        print(f'  Processed {len(current_file_flights)} unique flights from this file')
        processed_flights.update(current_file_flights)

        # Clean up memory
        del df_data, gb_uid
        gc.collect()

        return processed_flights

    except Exception as e:
        print(f'  ****** Failed to process file {file_path} -> {e}')
        return processed_flights


def process_all_files(data_dir, output_dir, file_range=None, route_filter=None):
    """
    Process all files in the specified range and extract individual flights.

    Args:
        data_dir: Directory containing input parquet files
        output_dir: Directory to save individual flight files
        file_range: Range of file numbers to process (e.g., range(4000, 4010))
        route_filter: Optional dict with 'origin' and 'destination' to filter routes
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "summaries"), exist_ok=True)

    # Set default file range if not provided
    if file_range is None:
        file_range = range(4000, 4010)  # Process files 4000-4009 by default

    processed_flights = set()
    total_flights = 0

    # Process each file
    for file_number in file_range:
        file_path = os.path.join(data_dir, f"AirNav-Kafka-{file_number}.parquet.gzip")

        if not os.path.exists(file_path):
            print(f'File not found: {file_path}')
            continue

        # Process the file
        processed_flights = process_single_file(file_path, output_dir, processed_flights)

        current_total = len(processed_flights)
        new_flights = current_total - total_flights
        total_flights = current_total

        print(f'File {file_number} complete. New flights: {new_flights}, Total flights: {total_flights}')
        print('-' * 50)

    print(f'Processing complete! Total unique flights processed: {total_flights}')

    # Create overall summary
    create_overall_summary(output_dir, total_flights)


def create_overall_summary(output_dir, total_flights):
    """Create an overall summary of all processed flights."""
    summary_dir = os.path.join(output_dir, "summaries")
    overall_summary = {
        'total_flights': total_flights,
        'summary_files': [],
        'processing_date': pd.Timestamp.now().isoformat()
    }

    # List all summary files
    if os.path.exists(summary_dir):
        for file in os.listdir(summary_dir):
            if file.endswith('_summary.json'):
                overall_summary['summary_files'].append(file)

    # Save overall summary
    import json
    with open(os.path.join(output_dir, "overall_summary.json"), 'w') as f:
        json.dump(overall_summary, f, indent=2)

    print(f'Overall summary saved to: {os.path.join(output_dir, "overall_summary.json")}')


def load_specific_flight(output_dir, unique_id):
    """
    Load a specific flight's data from the processed files.

    Args:
        output_dir: Directory containing processed flight files
        unique_id: Unique ID of the flight to load

    Returns:
        DataFrame with the flight data
    """
    safe_uid = unique_id.replace(' ', '_').replace(':', '-').replace('/', '-')
    flight_filename = f"flight_{safe_uid}.parquet"
    flight_path = os.path.join(output_dir, flight_filename)

    if os.path.exists(flight_path):
        return pd.read_parquet(flight_path)
    else:
        print(f'Flight file not found: {flight_path}')
        return None


def process_additional_files(data_dir, output_dir, new_file_range):
    """
    Process additional files and merge data with existing flights.

    Args:
        data_dir: Directory containing input parquet files
        output_dir: Directory with existing processed flight files
        new_file_range: Range of new file numbers to process
    """
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} doesn't exist. Run process_all_files first.")
        return

    # Get list of already processed flights
    processed_flights = set()
    summary_dir = os.path.join(output_dir, "summaries")
    if os.path.exists(summary_dir):
        for summary_file in os.listdir(summary_dir):
            if summary_file.endswith('_summary.json'):
                import json
                with open(os.path.join(summary_dir, summary_file), 'r') as f:
                    summary = json.load(f)
                processed_flights.add(summary['unique_id'])

    print(f"Found {len(processed_flights)} existing flights")

    # Process new files
    initial_count = len(processed_flights)

    for file_number in new_file_range:
        file_path = os.path.join(data_dir, f"data-Kafka-{file_number}.parquet.gzip")

        if not os.path.exists(file_path):
            print(f'File not found: {file_path}')
            continue

        # Process the file (will merge with existing flights)
        processed_flights = process_single_file(file_path, output_dir, processed_flights)

        current_total = len(processed_flights)
        print(f'File {file_number} complete. Total flights: {current_total}')
        print('-' * 50)

    final_count = len(processed_flights)
    new_flights = final_count - initial_count

    print(f'Additional processing complete!')
    print(f'New flights discovered: {new_flights}')
    print(f'Total flights now: {final_count}')

    # Update overall summary
    create_overall_summary(output_dir, final_count)


def get_flight_statistics(output_dir):
    """Get statistics about processed flights."""
    summary_dir = os.path.join(output_dir, "summaries")
    if not os.path.exists(summary_dir):
        print("No processed flights found.")
        return

    import json

    total_flights = 0
    total_rows = 0
    flights_with_waypoints = 0
    airports = set()
    aircraft_registrations = set()

    for summary_file in os.listdir(summary_dir):
        if summary_file.endswith('_summary.json'):
            with open(os.path.join(summary_dir, summary_file), 'r') as f:
                summary = json.load(f)

            total_flights += 1
            total_rows += summary.get('nbr_rows', 0)

            if int(summary.get('nbr_rows_with_wps', 0)) > 0:
                flights_with_waypoints += 1

            if summary.get('origin_airport'):
                airports.add(summary['origin_airport'])
            if summary.get('destination_airport'):
                airports.add(summary['destination_airport'])
            if summary.get('aircraft_registration'):
                aircraft_registrations.add(summary['aircraft_registration'])

    print(f"Flight Statistics:")
    print(f"  Total flights: {total_flights}")
    print(f"  Total data rows: {total_rows:,}")
    print(f"  Average rows per flight: {total_rows/total_flights:.1f}")
    print(f"  Flights with waypoint data: {flights_with_waypoints}")
    print(f"  Unique airports: {len(airports)}")
    print(f"  Unique aircraft: {len(aircraft_registrations)}")

    return {
        'total_flights': total_flights,
        'total_rows': total_rows,
        'flights_with_waypoints': flights_with_waypoints,
        'unique_airports': len(airports),
        'unique_aircraft': len(aircraft_registrations)
    }


def list_available_flights(output_dir):
    """List all available flights in the output directory."""
    summary_path = os.path.join(output_dir, "overall_summary.json")

    if os.path.exists(summary_path):
        import json
        with open(summary_path, 'r') as f:
            summary = json.load(f)

        print(f"Total flights available: {summary['total_flights']}")

        # List first few flights as examples
        summary_dir = os.path.join(output_dir, "summaries")
        if os.path.exists(summary_dir):
            summary_files = [f for f in os.listdir(summary_dir) if f.endswith('_summary.json')][:10]
            print("Sample flights:")
            for summary_file in summary_files:
                with open(os.path.join(summary_dir, summary_file), 'r') as f:
                    flight_summary = json.load(f)
                print(f"  - {flight_summary['unique_id']} ({flight_summary['nbr_rows']} rows)")
    else:
        print("No summary file found. Run process_all_files first.")


# Example usage
if __name__ == "__main__":
    # Configuration
    DATA_DIR = "data"  # Directory containing your parquet files
    ORIGIN_AIRPORT = "CYYZ"
    DESTINATION_AIRPORT = "LFPG"
    OUTPUT_DIR = f"./results/processed_flights_{ORIGIN_AIRPORT}_{DESTINATION_AIRPORT}"  # Directory to save individual flight files

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    # INITIAL PROCESSING - Process first batch of files
    print("=== INITIAL PROCESSING ===")
    FILE_RANGE = range(4000, 6575)
    process_all_files(DATA_DIR, OUTPUT_DIR, FILE_RANGE)

    # Get statistics
    print("\n=== STATISTICS AFTER INITIAL PROCESSING ===")
    get_flight_statistics(OUTPUT_DIR)

    # Get updated statistics
    print("\n=== FINAL STATISTICS ===")
    get_flight_statistics(OUTPUT_DIR)

    # List available flights
    print("\n=== AVAILABLE FLIGHTS ===")
    list_available_flights(OUTPUT_DIR)

    # Example: Load a specific flight
    print("\n=== EXAMPLE: LOADING SPECIFIC FLIGHT ===")
    summary_dir = os.path.join(OUTPUT_DIR, "summaries")
    if os.path.exists(summary_dir):
        summary_files = [f for f in os.listdir(summary_dir) if f.endswith('_summary.json')]
        if summary_files:
            import json
            with open(os.path.join(summary_dir, summary_files[0]), 'r') as f:
                first_flight = json.load(f)

            flight_data = load_specific_flight(OUTPUT_DIR, first_flight['unique_id'])
            if flight_data is not None:
                print(f"Loaded flight {first_flight['unique_id']} with {len(flight_data)} rows")
                print(f"Time range: {flight_data['t'].min()} to {flight_data['t'].max()}")
                print(flight_data[['t', 'lo', 'la', 'acr', 'cs']].head())