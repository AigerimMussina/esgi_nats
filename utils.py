import re

import requests


def extract_airport_codes(filename):
    """Extract airport codes from flight filename pattern"""
    # Pattern: flight_callsign_flightnum_origin_destination_date_time_filtered.parquet
    parts = filename.split('_')

    if len(parts) >= 5:
        origin = parts[3]  # LFPG
        destination = parts[4]  # CYYZ
        return origin, destination
    else:
        # Fallback regex pattern
        pattern = r'_([A-Z]{4})_([A-Z]{4})_'
        match = re.search(pattern, filename)
        if match:
            return match.group(1), match.group(2)

    return None, None

def get_airport_info_online(icao_code):
    """Get airport info using online API"""
    try:
        # Using airport-data.com API (free)
        url = f"https://airport-data.com/api/ap_info.json?icao={icao_code}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            return {
                'icao': icao_code,
                'iata': data.get('iata', 'N/A'),
                'name': data.get('name', 'Unknown'),
                'city': data.get('city', 'Unknown'),
                'country': data.get('country', 'Unknown')
            }
    except Exception as e:
        print(f"Error fetching data for {icao_code}: {e}")

    return None

