import math
import random
from qubots.base_problem import BaseProblem
import os

# --- Helper functions for instance reading ---

def read_elem(filename):

    # Resolve relative path with respect to this module’s directory.
    if not os.path.isabs(instance_file):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        instance_file = os.path.join(base_dir, instance_file)

    with open(filename) as f:
        return f.read().split()

def compute_dist(xi, xj, yi, yj):
    return math.sqrt((xi - xj)**2 + (yi - yj)**2)

def get_profile(dist, distance_levels, nb_distance_levels):
    idx = 0
    while idx < nb_distance_levels and dist > distance_levels[idx]:
        idx += 1
    return idx

def compute_distance_matrices(customers_x, customers_y, max_horizon, travel_time_profile_matrix,
                              time_interval_steps, nb_time_intervals, distance_levels, nb_distance_levels):
    nb_customers = len(customers_x)
    distance_matrix = [[0 for _ in range(nb_customers)] for _ in range(nb_customers)]
    travel_time = [[[0 for _ in range(nb_time_intervals)] for _ in range(nb_customers)] for _ in range(nb_customers)]
    time_to_matrix_idx = [0 for _ in range(max_horizon)]
    for i in range(nb_customers):
        for j in range(nb_customers):
            d = compute_dist(customers_x[i], customers_x[j], customers_y[i], customers_y[j])
            distance_matrix[i][j] = d
            profile_idx = get_profile(d, distance_levels, nb_distance_levels)
            for k in range(nb_time_intervals):
                travel_time[i][j][k] = travel_time_profile_matrix[profile_idx][k] * d
    for i in range(nb_time_intervals):
        time_step_start = int(round(time_interval_steps[i] * max_horizon))
        time_step_end = int(round(time_interval_steps[i+1] * max_horizon))
        for t in range(time_step_start, time_step_end):
            if t < max_horizon:
                time_to_matrix_idx[t] = i
    return distance_matrix, travel_time, time_to_matrix_idx

def compute_distance_depots(depot_x, depot_y, customers_x, customers_y, travel_time_profile_matrix,
                            nb_time_intervals, distance_levels, nb_distance_levels):
    nb_customers = len(customers_x)
    distance_depots = [0 for _ in range(nb_customers)]
    travel_time_warehouse = [[0 for _ in range(nb_time_intervals)] for _ in range(nb_customers)]
    for i in range(nb_customers):
        d = compute_dist(depot_x, customers_x[i], depot_y, customers_y[i])
        distance_depots[i] = d
        profile_idx = get_profile(d, distance_levels, nb_distance_levels)
        for k in range(nb_time_intervals):
            travel_time_warehouse[i][k] = travel_time_profile_matrix[profile_idx][k] * d
    return distance_depots, travel_time_warehouse

def read_input_tdcvrptw(filename):
    # Read instance data from a Solomon-like file
    tokens = read_elem(filename)
    idx = 0
    instance_name = tokens[idx]; idx += 1  # instance name
    # Skip next 4 tokens.
    idx += 4
    nb_trucks = int(tokens[idx]); idx += 1
    truck_capacity = int(tokens[idx]); idx += 1
    # Skip next 13 tokens.
    idx += 13
    depot_x = int(tokens[idx]); idx += 1
    depot_y = int(tokens[idx]); idx += 1
    # Skip next 2 tokens.
    idx += 2
    max_horizon = int(tokens[idx]); idx += 1
    idx += 1  # skip one token
    customers_x = []
    customers_y = []
    demands = []
    earliest_start = []
    latest_end = []
    service_time = []
    while idx < len(tokens):
        val = tokens[idx]
        idx += 1
        if val is None:
            break
        # Customer index (we ignore it, but use it to count)
        cust_idx = int(val) - 1
        customers_x.append(int(tokens[idx])); idx += 1
        customers_y.append(int(tokens[idx])); idx += 1
        demands.append(int(tokens[idx])); idx += 1
        ready = int(tokens[idx]); idx += 1
        due = int(tokens[idx]); idx += 1
        stime = int(tokens[idx]); idx += 1
        earliest_start.append(ready)
        latest_end.append(due + stime)
        service_time.append(stime)
    nb_customers = len(customers_x)
    # Define travel time profiles.
    short_profile = [1.00, 2.50, 1.75, 2.50, 1.00]
    medium_profile = [1.00, 2.00, 1.50, 2.00, 1.00]
    long_profile = [1.00, 1.60, 1.10, 1.60, 1.00]
    travel_time_profile_matrix = [short_profile, medium_profile, long_profile]
    distance_levels = [10, 25]
    time_interval_steps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    nb_time_intervals = len(time_interval_steps) - 1
    nb_distance_levels = len(distance_levels)
    distance_matrix, travel_time, time_to_matrix_idx = compute_distance_matrices(
        customers_x, customers_y, max_horizon, travel_time_profile_matrix, time_interval_steps,
        nb_time_intervals, distance_levels, nb_distance_levels)
    distance_depots, travel_time_warehouse = compute_distance_depots(
        depot_x, depot_y, customers_x, customers_y, travel_time_profile_matrix,
        nb_time_intervals, distance_levels, nb_distance_levels)
    return (nb_customers, nb_trucks, truck_capacity, distance_matrix, travel_time,
            time_to_matrix_idx, distance_depots, travel_time_warehouse, demands,
            service_time, earliest_start, latest_end, max_horizon)

# --- Qubot Problem Class ---

class TDCVRPTWProblem(BaseProblem):
    """
    Time Dependent Capacitated Vehicle Routing with Time Windows (TDCVRPTW)

    A fleet of vehicles (each with the same capacity) must serve a set of customers,
    each with a demand and a time window (given by an earliest start and a latest end time).
    The vehicles start and end at a depot. Travel times between customers are time–dependent:
    the time to travel from customer i to customer j is determined by a discretized time profile
    (via the time_to_matrix_idx mapping). Service times are fixed per customer.
    
    A candidate solution is a list of truck routes (one route per truck). Each route is a list of customer indices.
    Every customer must be assigned to exactly one truck.
    
    For each truck route, the simulation computes:
      • The total demand (which must not exceed the truck capacity).
      • The distance traveled (from depot to first customer, between customers, and from the last customer back to the depot).
      • The end times of each customer visit are computed recursively as:
          
          T[0] = max( earliest[route[0]], travel_time_warehouse[route[0]][ time_to_matrix_idx[0] ] ) + service_time[route[0]]
          For i ≥ 1:
            T[i] = max( earliest[route[i]], T[i-1] + travel_time[ route[i-1] ][ route[i] ][ time_to_matrix_idx[ round(T[i-1]) ] ] ) + service_time[route[i]]
      
      • “Home lateness” is computed as max(0, T[last] + travel_time_warehouse[route[last]][ time_to_matrix_idx[ round(T[last]) ] ] - max_horizon).
      • For each visit, lateness is max(0, T[i] – latest[route[i]]).
      • The truck’s lateness is the sum of home lateness and the per-visit lateness.
    
    The lexicographic objective is to minimize (in order):
      1. Total lateness (sum over all trucks)
      2. Number of trucks used (nonempty routes)
      3. Total distance traveled
      
    Here we combine them into one weighted objective.
    """
    def __init__(self, instance_file):
        self.instance_file = instance_file
        (self.nb_customers, self.nb_trucks, self.truck_capacity, self.distance_matrix, self.travel_time,
         self.time_to_matrix_idx, self.distance_depots, self.travel_time_warehouse, self.demands,
         self.service_time, self.earliest, self.latest, self.max_horizon) = read_input_tdcvrptw(instance_file)
    
    def evaluate_solution(self, candidate) -> float:
        penalty = 0
        total_lateness = 0
        total_distance = 0
        trucks_used = 0
        
        # Check that candidate is a valid partition of customers.
        assigned = []
        for route in candidate:
            assigned.extend(route)
        if sorted(assigned) != list(range(self.nb_customers)):
            penalty += 1e9
        
        for route in candidate:
            if not route:
                continue
            trucks_used += 1
            # Capacity check.
            route_demand = sum(self.demands[j] for j in route)
            if route_demand > self.truck_capacity:
                penalty += 1e9 * (route_demand - self.truck_capacity)
            # Compute route distance.
            route_distance = self.distance_depots[route[0]] + self.distance_depots[route[-1]]
            for i in range(1, len(route)):
                route_distance += self.distance_matrix[route[i-1]][route[i]]
            total_distance += route_distance
            # Simulate the route: compute end times T[i] for each visit.
            T = []
            # First customer:
            idx0 = self.time_to_matrix_idx[0]  # time 0 maps to first interval
            t0 = max(self.earliest[route[0]], self.travel_time_warehouse[route[0]][idx0]) + self.service_time[route[0]]
            T.append(t0)
            for i in range(1, len(route)):
                prev = T[-1]
                # Ensure index is within range:
                time_index = self.time_to_matrix_idx[min(int(round(prev)), self.max_horizon-1)]
                travel = self.travel_time[route[i-1]][route[i]][time_index]
                start = max(self.earliest[route[i]], prev + travel)
                t_i = start + self.service_time[route[i]]
                T.append(t_i)
            # Home lateness.
            last = T[-1]
            idx_last = self.time_to_matrix_idx[min(int(round(last)), self.max_horizon-1)]
            home_time = last + self.travel_time_warehouse[route[-1]][idx_last]
            home_lateness = max(0, home_time - self.max_horizon)
            # Lateness at visits.
            lateness_visits = sum(max(0, T[i] - self.latest[route[i]]) for i in range(len(route)))
            route_lateness = home_lateness + lateness_visits
            total_lateness += route_lateness
        
        nb_trucks_used = trucks_used
        
        # Combine objectives lexicographically using large weights.
        objective = total_lateness + nb_trucks_used * 1e9 + total_distance
        return objective + penalty
    
    def random_solution(self):
        # Generate a random partition of customers among trucks.
        customers = list(range(self.nb_customers))
        random.shuffle(customers)
        candidate = [[] for _ in range(self.nb_trucks)]
        for i, cust in enumerate(customers):
            candidate[i % self.nb_trucks].append(cust)
        for route in candidate:
            random.shuffle(route)
        return candidate
