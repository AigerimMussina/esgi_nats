import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import heapq
import datetime
import math
import pickle
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.spatial import cKDTree
from typing import Dict, Tuple, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from wind_service import WindGridOptimizer, create_wind_grid_optimizer

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)


# ==================== UTILITY FUNCTIONS ====================

def haversine_km(lon1, lat1, lon2, lat2):

    R = 6371  # Earth R in rm

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)

    c = 2 * math.asin(math.sqrt(a))
    return R * c


def calculate_bearing(lat1, lon1, lat2, lon2):

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1

    y = math.sin(dlon) * math.cos(lat2)
    x = (math.cos(lat1) * math.sin(lat2) -
         math.sin(lat1) * math.cos(lat2) * math.cos(dlon))

    bearing = math.atan2(y, x)
    return math.degrees(bearing)


def calculate_ground_speed_wind_triangle(lat1, lon1, lat2, lon2, tas_knots, wind_u, wind_v):


    track_angle = calculate_bearing(lat1, lon1, lat2, lon2)
    track_rad = math.radians(track_angle)


    tas_ms = tas_knots * 0.514444


    ground_speed_x = tas_ms * math.sin(track_rad) + wind_u
    ground_speed_y = tas_ms * math.cos(track_rad) + wind_v


    ground_speed_ms = math.sqrt(ground_speed_x ** 2 + ground_speed_y ** 2)
    ground_speed_knots = ground_speed_ms / 0.514444

    return ground_speed_knots


def find_closest_time_index(target_time, times_list):

    if not times_list:
        return 0


    target_ts = target_time.timestamp() if hasattr(target_time, 'timestamp') else target_time

    min_diff = float('inf')
    closest_idx = 0

    for i, time_point in enumerate(times_list):
        time_ts = time_point.timestamp() if hasattr(time_point, 'timestamp') else time_point
        diff = abs(time_ts - target_ts)

        if diff < min_diff:
            min_diff = diff
            closest_idx = i

    return closest_idx


def calculate_heuristic_with_epsilon(lat, lon, end_info, epsilon=0.1):

    distance_km = haversine_km(lon, lat, end_info['lo'], end_info['la'])
    distance_nm = distance_km * 0.539957
    return distance_nm / 450.0 * (1 + epsilon)


def calculate_tie_breaker(lat, lon, end_info):

    return haversine_km(lon, lat, end_info['lo'], end_info['la']) * 0.001


# ==================== FIXED GRADIENT NODE CLASS ====================

class GradientNode:

    __slots__ = ['lat', 'lon', 'alt', 'time_idx', 'real_time', 'parent', 'g_cost', 'h_cost', 'f_cost', 'gradient_info']

    def __init__(self, lat, lon, alt, time_idx, real_time, parent=None):
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.time_idx = time_idx
        self.real_time = real_time
        self.parent = parent
        self.g_cost = 0
        self.h_cost = 0
        self.f_cost = 0
        self.gradient_info = {}

    def __lt__(self, other):
        return self.f_cost < other.f_cost

    def __eq__(self, other):
        return (self.lat == other.lat and self.lon == other.lon and
                self.alt == other.alt and self.time_idx == other.time_idx)

    def __hash__(self):
        return hash((round(self.lat, 2), round(self.lon, 2), self.alt, self.time_idx))


# ==================== FIXED GRADIENT FUNCTIONS ====================

def calculate_wind_gradient(node, goal_info, wind_grid):

    return wind_grid.get_wind_gradient(
        node.lat, node.lon, node.alt // 100, node.time_idx,
        goal_info['la'], goal_info['lo']
    )


def calculate_gradient_cost_FIXED(current_node, neighbor_node, goal_info, wind_grid,
                                  alpha=0.05, beta=0.1, gamma=0.02, tas_knots=450.0):
    """
    FIXED: Calculate cost using temporal gradient descent with proper scaling
    """
    # Basic flight time cost
    dist_km = haversine_km(current_node.lon, current_node.lat,
                           neighbor_node.lon, neighbor_node.lat)

    # Add altitude change penalty
    if neighbor_node.alt != current_node.alt:
        dist_km += 5

    dist_nm = dist_km * 0.539957

    # Get wind at neighbor position
    u_wind, v_wind = wind_grid.get_wind_fast(neighbor_node.lat, neighbor_node.lon,
                                             neighbor_node.alt // 100, neighbor_node.time_idx)

    if np.isnan(u_wind) or np.isnan(v_wind):
        return float('inf')

    # Calculate ground speed
    ground_speed = calculate_ground_speed_wind_triangle(
        current_node.lat, current_node.lon, neighbor_node.lat, neighbor_node.lon,
        tas_knots, u_wind, v_wind
    )

    if ground_speed <= 50:
        return float('inf')

    base_time_cost = dist_nm / ground_speed

    # Calculate wind gradient at neighbor position
    wind_gradient = calculate_wind_gradient(neighbor_node, goal_info, wind_grid)

    # Movement direction vector
    movement_vector = np.array([
        neighbor_node.lon - current_node.lon,
        neighbor_node.lat - current_node.lat,
        (neighbor_node.alt - current_node.alt) / 1000
    ])

    movement_length = np.linalg.norm(movement_vector)
    if movement_length > 0:
        movement_unit = movement_vector / movement_length
    else:
        movement_unit = np.zeros(3)

    # FIXED: Normalized gradient bonus
    spatial_gradient = wind_gradient[:3]
    raw_gradient_bonus = np.dot(movement_unit, spatial_gradient)

    # Scale gradient bonus to reasonable range [-1, +1]
    gradient_bonus = np.tanh(raw_gradient_bonus / 5.0)  # Sigmoid scaling

    # FIXED: Normalized goal alignment bonus
    goal_vector = np.array([
        goal_info['lo'] - neighbor_node.lon,
        goal_info['la'] - neighbor_node.lat,
        0
    ])

    goal_length = np.linalg.norm(goal_vector)
    if goal_length > 0:
        goal_unit = goal_vector / goal_length
        raw_goal_bonus = np.dot(movement_unit, goal_unit)
        goal_bonus = raw_goal_bonus  # Already in [-1, +1]
    else:
        goal_bonus = 0

    # FIXED: Normalized temporal bonus
    raw_temporal_bonus = wind_gradient[3]
    temporal_bonus = np.tanh(raw_temporal_bonus / 2.0)  # Sigmoid scaling

    # FIXED: Conservative coefficients to prevent negative costs
    total_cost = (base_time_cost
                  * (1.0 - alpha * gradient_bonus - beta * goal_bonus - gamma * temporal_bonus))

    # Ensure positive cost
    final_cost = max(0.01 * base_time_cost, total_cost)

    # Store gradient info for debugging
    neighbor_node.gradient_info = {
        'base_cost': base_time_cost,
        'raw_gradient_bonus': raw_gradient_bonus,
        'gradient_bonus': gradient_bonus,
        'goal_bonus': goal_bonus,
        'temporal_bonus': temporal_bonus,
        'wind_gradient': wind_gradient,
        'ground_speed': ground_speed,
        'final_cost': final_cost
    }

    return final_cost


def evaluate_neighbors_gradient_FIXED(current_node, end_info, wind_grid,
                                      alpha=0.05, beta=0.1, gamma=0.02, max_workers=8):
    """Evaluate neighbors using FIXED temporal gradient descent"""

    moves = [
        (1.0, 0, 0),  # N
        (-1.0, 0, 0),  # S
        (0, 2.0, 0),  # E
        (0, -2.0, 0),  # W
        (1.0, 2.0, 0),  # NE
        (1.0, -2.0, 0),  # NW
        (-1.0, 2.0, 0),  # SE
        (-1.0, -2.0, 0),  # SW
        (0, 0, 4000),  # UP
        (0, 0, -4000)  # DOWN
    ]

    def evaluate_single_neighbor_gradient_fixed(move):
        d_lat, d_lon, d_alt = move

        new_lat = current_node.lat + d_lat
        new_lon = current_node.lon + d_lon
        new_alt = current_node.alt + d_alt

        # Bounds check
        if not (40 <= new_lat <= 70 and -80 <= new_lon <= 0 and
                28000 <= new_alt <= 42000):
            return None

        # Calculate time cost with FIXED function
        try:
            time_cost = calculate_gradient_cost_FIXED(current_node,
                                                      GradientNode(new_lat, new_lon, new_alt,
                                                                   current_node.time_idx + 1,
                                                                   current_node.real_time),
                                                      end_info, wind_grid, alpha, beta, gamma)

            if time_cost == float('inf'):
                return None

            # Real time progression
            new_real_time = current_node.real_time + pd.Timedelta(hours=time_cost)
            new_time_idx = find_closest_time_index(new_real_time, wind_grid.times)

            if new_time_idx >= len(wind_grid.times):
                return None

            # Create neighbor with gradient info
            neighbor = GradientNode(new_lat, new_lon, new_alt, new_time_idx,
                                    new_real_time, current_node)
            neighbor.g_cost = current_node.g_cost + time_cost
            neighbor.h_cost = calculate_heuristic_with_epsilon(new_lat, new_lon, end_info, 0.1)

            tie_breaker = calculate_tie_breaker(new_lat, new_lon, end_info)
            neighbor.f_cost = neighbor.g_cost + neighbor.h_cost + tie_breaker

            return neighbor

        except Exception as e:
            return None

    # Evaluate all moves in parallel
    valid_neighbors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_move = {executor.submit(evaluate_single_neighbor_gradient_fixed, move): move
                          for move in moves}
        for future in as_completed(future_to_move):
            try:
                neighbor = future.result()
                if neighbor is not None:
                    valid_neighbors.append(neighbor)
            except Exception:
                continue

    return valid_neighbors


# ==================== FIXED TEMPORAL GRADIENT A* ALGORITHM ====================

def temporal_gradient_astar_FIXED(start_info, end_info, wind_grid,
                                  alpha=0.05, beta=0.1, gamma=0.02, max_workers=8):
    """FIXED A* search with temporal gradient descent cost function"""

    print(f"🌊 Starting FIXED TEMPORAL GRADIENT A* search...")
    print(f"   🎯 FIXED Parameters: α={alpha} (gradient), β={beta} (goal), γ={gamma} (temporal)")
    print(f"   🔧 Key fixes: Sigmoid scaling, multiplicative cost, positive constraints")
    print(f"   📍 Start: {start_info['t']} at ({start_info['la']:.2f}, {start_info['lo']:.2f})")
    print(f"   📍 End: {end_info['t']} at ({end_info['la']:.2f}, {end_info['lo']:.2f})")

    # Initialize start node
    start_time_idx = find_closest_time_index(start_info['t'], wind_grid.times)
    start_node = GradientNode(start_info['la'], start_info['lo'], start_info['alt'],
                              start_time_idx, start_info['t'])
    start_node.h_cost = calculate_heuristic_with_epsilon(start_node.lat, start_node.lon, end_info, 0.1)
    start_node.f_cost = start_node.h_cost

    # Initialize search
    open_list = []
    heapq.heappush(open_list, start_node)
    closed_set = set()

    iterations = 0
    best_distance = float('inf')

    gc_distance = haversine_km(start_info['lo'], start_info['la'], end_info['lo'], end_info['la'])

    print(f"   📏 Great circle distance: {gc_distance:.0f} km")
    print(f"🚀 Fixed temporal gradient search started...")

    while open_list and iterations < 15000:
        iterations += 1
        current_node = heapq.heappop(open_list)

        # Enhanced progress reporting
        if iterations % 50 == 0:
            dist_to_goal = haversine_km(current_node.lon, current_node.lat,
                                        end_info['lo'], end_info['la'])
            if dist_to_goal < best_distance:
                best_distance = dist_to_goal

            progress = (gc_distance - dist_to_goal) / gc_distance * 100
            progress = max(0, min(100, progress))

            # Show FIXED gradient info
            gradient_info = getattr(current_node, 'gradient_info', {})
            gradient_bonus = gradient_info.get('gradient_bonus', 0)
            goal_bonus = gradient_info.get('goal_bonus', 0)
            temporal_bonus = gradient_info.get('temporal_bonus', 0)
            final_cost = gradient_info.get('final_cost', 0)

            print(f"   Iteration {iterations}: progress={progress:.1f}%, "
                  f"dist={dist_to_goal:.0f}km, "
                  f"g={current_node.g_cost:.3f}h, "
                  f"∇W={gradient_bonus:.3f}, goal={goal_bonus:.3f}, "
                  f"∂t={temporal_bonus:.3f}, cost={final_cost:.3f}")

        # Goal check
        dist_to_goal = haversine_km(current_node.lon, current_node.lat,
                                    end_info['lo'], end_info['la'])
        if dist_to_goal < 75:
            print(f"\n🎯 Goal reached in {iterations} iterations!")
            print(f"   Final distance: {dist_to_goal:.1f} km")
            print(f"   Total flight time: {current_node.g_cost:.2f} hours")

            # Show final gradient statistics
            total_gradient_bonus = 0
            total_goal_bonus = 0
            total_temporal_bonus = 0
            valid_nodes = 0

            current = current_node
            while current and hasattr(current, 'gradient_info'):
                info = current.gradient_info
                total_gradient_bonus += info.get('gradient_bonus', 0)
                total_goal_bonus += info.get('goal_bonus', 0)
                total_temporal_bonus += info.get('temporal_bonus', 0)
                valid_nodes += 1
                current = current.parent

            if valid_nodes > 0:
                print(f"   Average gradient bonus: {total_gradient_bonus / valid_nodes:.3f}")
                print(f"   Average goal bonus: {total_goal_bonus / valid_nodes:.3f}")
                print(f"   Average temporal bonus: {total_temporal_bonus / valid_nodes:.3f}")

            # Reconstruct path
            path = []
            current = current_node
            while current:
                path.append(current)
                current = current.parent

            return path[::-1]

        # Add to closed set
        closed_set.add(current_node)

        # Get neighbors using FIXED evaluation
        try:
            neighbors = evaluate_neighbors_gradient_FIXED(current_node, end_info, wind_grid,
                                                          alpha, beta, gamma, max_workers)

            for neighbor in neighbors:
                if neighbor not in closed_set:
                    heapq.heappush(open_list, neighbor)

        except Exception as e:
            print(f"   Warning: Neighbor evaluation failed: {e}")
            continue

    print("❌ Search completed without finding goal")
    return None


# ==================== ANALYSIS AND VISUALIZATION ====================

def create_flight_dataframe_FIXED(path):

    if not path:
        return pd.DataFrame()

    data = {
        'lat': [node.lat for node in path],
        'lon': [node.lon for node in path],
        'alt': [node.alt for node in path],
        'time_idx': [node.time_idx for node in path],
        'real_time': [node.real_time for node in path],
        'g_cost_hours': [node.g_cost for node in path],
        'f_cost': [node.f_cost for node in path]
    }

    # Add gradient info if available
    gradient_info_keys = ['gradient_bonus', 'goal_bonus', 'temporal_bonus', 'ground_speed', 'final_cost']
    for key in gradient_info_keys:
        data[key] = [getattr(node, 'gradient_info', {}).get(key, 0) for node in path]

    return pd.DataFrame(data)


def analyze_fixed_gradient_path(path, wind_grid):

    if not path:
        return None

    df = create_flight_dataframe_FIXED(path)

    results = {
        'waypoints': len(path),
        'total_time': path[-1].g_cost,
        'total_distance': 0,
        'avg_gradient_bonus': df['gradient_bonus'].mean(),
        'avg_goal_bonus': df['goal_bonus'].mean(),
        'avg_temporal_bonus': df['temporal_bonus'].mean(),
        'avg_ground_speed': df['ground_speed'].mean(),
        'gradient_utilization': (df['gradient_bonus'] > 0).sum() / len(df) * 100
    }

    total_distance = 0
    for i in range(1, len(path)):
        prev_node = path[i - 1]
        curr_node = path[i]
        segment_distance = haversine_km(prev_node.lon, prev_node.lat, curr_node.lon, curr_node.lat)
        total_distance += segment_distance

    results['total_distance'] = total_distance
    results['avg_speed_kmh'] = total_distance / path[-1].g_cost if path[-1].g_cost > 0 else 0

    return results


def create_fixed_gradient_visualization(df_path, analysis, start_info, end_info):


    plt.figure(figsize=(20, 16))

    # 1. Route comparison
    plt.subplot(3, 4, 1)
    plt.plot(df_path['lon'], df_path['lat'], 'r-', linewidth=3, label='Fixed Gradient Route')
    plt.plot(start_info['lo'], start_info['la'], 'go', markersize=12, label='Start')
    plt.plot(end_info['lo'], end_info['la'], 'ro', markersize=12, label='End')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('FIXED Temporal Gradient Route')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Gradient bonuses
    plt.subplot(3, 4, 2)
    distances = [0]
    for i in range(1, len(df_path)):
        dist = haversine_km(df_path.iloc[i - 1]['lon'], df_path.iloc[i - 1]['lat'],
                            df_path.iloc[i]['lon'], df_path.iloc[i]['lat'])
        distances.append(distances[-1] + dist)

    plt.plot(distances, df_path['gradient_bonus'], 'purple', linewidth=2, label='Gradient Bonus')
    plt.plot(distances, df_path['goal_bonus'], 'orange', linewidth=2, label='Goal Bonus')
    plt.plot(distances, df_path['temporal_bonus'], 'cyan', linewidth=2, label='Temporal Bonus')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Distance (km)')
    plt.ylabel('Bonus Value')
    plt.title('FIXED Gradient Bonuses')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Ground speed
    plt.subplot(3, 4, 3)
    plt.plot(distances, df_path['ground_speed'], 'green', linewidth=2, label='Ground Speed')
    plt.axhline(y=450, color='red', linestyle='--', alpha=0.5, label='TAS = 450 kts')
    plt.xlabel('Distance (km)')
    plt.ylabel('Speed (knots)')
    plt.title('Ground Speed Profile')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Altitude profile
    plt.subplot(3, 4, 4)
    plt.plot(distances, df_path['alt'], 'blue', linewidth=2, label='Altitude')
    plt.xlabel('Distance (km)')
    plt.ylabel('Altitude (feet)')
    plt.title('Altitude Profile')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5. Cost evolution
    plt.subplot(3, 4, 5)
    plt.plot(distances, df_path['final_cost'], 'red', linewidth=2, label='Final Cost')
    plt.xlabel('Distance (km)')
    plt.ylabel('Cost (hours)')
    plt.title('Cost Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6. Cumulative time
    plt.subplot(3, 4, 6)
    plt.plot(distances, df_path['g_cost_hours'], 'navy', linewidth=2, label='Cumulative Time')
    plt.xlabel('Distance (km)')
    plt.ylabel('Time (hours)')
    plt.title('Cumulative Flight Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 7. Gradient utilization histogram
    plt.subplot(3, 4, 7)
    plt.hist(df_path['gradient_bonus'], bins=20, alpha=0.7, color='purple', edgecolor='black')
    plt.axvline(analysis['avg_gradient_bonus'], color='red', linestyle='--',
                label=f'Mean: {analysis["avg_gradient_bonus"]:.3f}')
    plt.xlabel('Gradient Bonus')
    plt.ylabel('Frequency')
    plt.title('Gradient Bonus Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 8. Time progression
    plt.subplot(3, 4, 8)
    time_hours = [(t - df_path['real_time'].iloc[0]).total_seconds() / 3600
                  for t in df_path['real_time']]
    plt.plot(time_hours, distances, 'orange', linewidth=2, label='Distance vs Time')
    plt.xlabel('Time (hours)')
    plt.ylabel('Distance (km)')
    plt.title('Distance vs Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Statistics panel
    plt.subplot(3, 4, 9)
    plt.axis('off')

    stats_text = f"""
    FIXED GRADIENT DESCENT RESULTS:

    📍 Waypoints: {analysis['waypoints']}
    ⏱️ Total time: {analysis['total_time']:.2f} h
    📏 Distance: {analysis['total_distance']:.1f} km
    🚀 Avg speed: {analysis['avg_speed_kmh']:.1f} km/h

    🌪️ Gradient bonus: {analysis['avg_gradient_bonus']:.3f}
    🎯 Goal bonus: {analysis['avg_goal_bonus']:.3f}
    ⏰ Temporal bonus: {analysis['avg_temporal_bonus']:.3f}

    ✈️ Avg ground speed: {analysis['avg_ground_speed']:.1f} kts
    📊 Gradient utilization: {analysis['gradient_utilization']:.1f}%

    ✅ FIXED algorithm completed!
    🔧 Sigmoid scaling applied
    📈 Positive cost constraints enforced
    """

    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()
    plt.show()


def load_specific_flight(output_dir, unique_id):
    """
    Load a specific flight's data from the processed files.
    """
    safe_uid = unique_id.replace(' ', '_').replace(':', '-').replace('/', '-')
    flight_filename = f"flight_{safe_uid}_filtered.parquet"
    print(f'filename: {flight_filename}')
    flight_path = os.path.join(output_dir, flight_filename)

    if os.path.exists(flight_path):
        return pd.read_parquet(flight_path)
    else:
        print(f'Flight file not found: {flight_path}')
        return None

# ==================== MAIN EXECUTION ====================

def main_fixed_temporal_gradient_search(loaded_flight, is_visualization=False):
    print("🚀 RUN FIXED TEMPORAL GRADIENT DESCENT FLIGHT OPTIMIZER")
    print("=" * 70)
    wind_grid_file = "./results/wind_grids_cache.pkl"

    if not os.path.exists(wind_grid_file):
        print(f"❌ File {wind_grid_file} not found!")
        print("Please run 3_flight_met_data_optimization.py first to get wind grid cache")
        return None

    wind_optimizer = WindGridOptimizer()
    wind_optimizer.load_grids(wind_grid_file)
    print("\n📊 Load data...")

    if loaded_flight is None:
        print("❌ No data could be loaded")
        return None

    start_info = loaded_flight.iloc[0]
    end_info = loaded_flight.iloc[-1]

    print(f"🛫 Start: ({start_info['la']:.2f}, {start_info['lo']:.2f}) on FL{start_info['alt'] / 100:.0f}")
    print(f"🛬 Finish: ({end_info['la']:.2f}, {end_info['lo']:.2f}) on FL{end_info['alt'] / 100:.0f}")
    print("\n🔍 Run FIXED Temporal Gradient Descent ...")

    alpha = 0.1  # Gradient weight
    beta = 0.5  # Goal weight
    gamma = 0.02  # Temporal weight

    optimal_path = temporal_gradient_astar_FIXED(start_info, end_info, wind_optimizer, alpha, beta, gamma, max_workers=4)
    if optimal_path:
        print("\n🎉 FIXED TEMPORAL GRADIENT DESCENT FINISHED SUCCESSFULLY!")
        print("=" * 70)

        df_optimal_path = create_flight_dataframe_FIXED(optimal_path)
        analysis = analyze_fixed_gradient_path(optimal_path, wind_optimizer)

        print(f"📊 Results:")
        print(f"   • Number of route points: {analysis['waypoints']}")
        print(f"   • Total time: {analysis['total_time']:.2f} h")
        print(f"   • Total distance: {analysis['total_distance']:.1f} km")
        print(f"   • Avg velocity: {analysis['avg_speed_kmh']:.1f} km/h")
        print(f"   • Average gradient bonus: {analysis['avg_gradient_bonus']:.3f}")
        print(f"   • Average goal bonus: {analysis['avg_goal_bonus']:.3f}")
        print(f"   • Average temporal bonus: {analysis['avg_temporal_bonus']:.3f}")
        print(f"   • Gradient utilization: {analysis['gradient_utilization']:.1f}%")


        print("\n📈 Create visualization...")
        if is_visualization:
            create_fixed_gradient_visualization(df_optimal_path, analysis, start_info, end_info)

        return df_optimal_path, analysis

    else:
        print("❌ Fixed Temporal Gradient Descent could not find optimal route")
        return None, None

if __name__ == "__main__":


    loaded_flight = load_specific_flight('filtered_flights_met_bounds_CYYZ_LFPG',
                                         'F:HUVR AFR355J CYYZ LFPG 2025:06:23 22:29:00')
    result_df, result_analysis = main_fixed_temporal_gradient_search(loaded_flight, is_visualization=True)
