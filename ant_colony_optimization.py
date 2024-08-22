'''
    CMSC 170 - ST1L | Introduction to Artificial Intelligence
    Ant Colony Optimization that Solves for the Travelling Sales Person Problem
    Programmer: John Rommel B. Octavo
'''

import random


# This will serve as the primary function in solving the ant_colony_optimization
def ant_colony_optimization(adjacency_matrix, num_ants, num_iterations, evaporation_rate, initial_pheromone):
    num_cities = len(adjacency_matrix)
    
    # Initialize pheromone levels on all edges in the graph.
    pheromone = [[initial_pheromone] * num_cities for _ in range(num_cities)]
    
    # Initialize the best path to none and minimum travel cost to positive infinity
    best_path = None
    min_travel_cost = float('inf')
    
    # Run the algorithm for the specified number of iterations
    for _ in range(num_iterations):
        
        # Initialize the paths taken by each ant and their distances
        paths = [[] for _ in range(num_ants)]
        path_distances = [0] * num_ants
        

        # Each ant will have take random paths to traverse the given graph
        for ant in range(num_ants):
            current_city = random.randint(0, num_cities-1)
            paths[ant].append(current_city)
            
            # Move to the next city until all cities have been visited
            while len(paths[ant]) < num_cities: 
                # Choose the next city based on pheromone levels and distance
                next_city = choose_next_city(adjacency_matrix, pheromone, paths[ant], current_city)
                paths[ant].append(next_city)
                path_distances[ant] += adjacency_matrix[current_city][next_city]
                current_city = next_city
                
            # Add the distance between the last and first cities
            path_distances[ant] += adjacency_matrix[current_city][paths[ant][0]]
            
            # Check if the the current path distance is lesser than the value of the min_travel_cost
            if path_distances[ant] < min_travel_cost:
                best_path = paths[ant]
                min_travel_cost = path_distances[ant]
                
        # This is to update the pheromone levels in the path
        update_pheromone(pheromone, paths, path_distances, evaporation_rate)
        
    best_path = [best_path[i] + 1 for i in range(len(best_path))]
    return best_path, min_travel_cost


# This function is responsible in choosing the next city the ant will traverse
def choose_next_city(graph, pheromone, path, current_city):
    
    # Calculate the distances to each neighboring city and the pheromone levels on the edges connecting them
    distances = [graph[current_city][neighbor] for neighbor in range(len(graph)) if neighbor not in path]
    pheromone_levels = [pheromone[current_city][neighbor] for neighbor in range(len(graph)) if neighbor not in path]
    
    # Calculate the probability of moving to a neighboring city based on the pheromone level and distance
    probabilities = [((pheromone_levels[i]) * ((1.0 / distances[i]))) for i in range(len(distances))]
    total_probability = sum(probabilities)
    probabilities = [probability / total_probability for probability in probabilities]
    
    # This is to choose a random city based on the probability
    next_city = random.choices([neighbor for neighbor in range(len(graph)) if neighbor not in path], probabilities)[0]
    return next_city


# This function is for updating the pheromone levels of each edges in the graph
# All pheromone levels will decreased based on the evaporation rate
# Paths taken by the ants will have an increased pheromone levels
def update_pheromone(pheromone, paths, path_distances, evaporation_rate):
    num_cities = len(pheromone)
    
    # Evaporate pheromone on all edges
    for i in range(num_cities):
        for j in range(num_cities):
            # Decrease the pheromone level of each edges by the evaporation rate.
            pheromone[i][j] *= (1 - evaporation_rate)
    
    # Increase pheromone levels on edges that the ants traversed
    for ant in range(len(paths)):
        for i in range(len(paths[ant])):
            # j is the next city from i. 
            j = (i + 1) % num_cities
            pheromone[paths[ant][i]][paths[ant][j]] += (1.0 / path_distances[ant])


# This is the main function of the program

#==============================================
adjacency_matrix = [
    [0, 20, 21, 22, 23, 24],
    [20, 0, 25, 26, 27, 28],
    [21, 25, 0, 29, 30, 31],
    [22, 26, 29, 0, 32, 33],
    [23, 27, 30, 32, 0, 34],
    [24, 28, 31, 33, 34, 0]
]
#==============================================


best_path, min_travel_cost = ant_colony_optimization(adjacency_matrix, num_ants=10, num_iterations=100, evaporation_rate=0.5, initial_pheromone=1)
print('Best solution found: nodes = ', best_path)
print('Travel Cost (From city A back to A) = ', min_travel_cost)