'''
    CMSC 170 - ST1L | Introduction to Artificial Intelligence
    Ant Colony Optimization that Solves for the Schedulling Aircraft Landings Problem
    Programmer: John Rommel B. Octavo
'''

import random
import time


def solve_aircraft_landing(aircrafts, num_ants, num_iterations, alpha, beta, rho, sigma):
    num_aircrafts = len(aircrafts)
    pheromone = [[1] * num_aircrafts for _ in range(num_aircrafts)]  # Initialize pheromone matrix
    separation_time = calculate_separation_time(aircrafts)  # Calculate separation time matrix
    best_solution = []  # Initialize variables to track best solution and cost
    best_cost = float('inf')

    for iteration in range(num_iterations):
        solutions = []  # Track solutions and costs for each ant in this iteration
        costs = []

        for ant in range(num_ants):
            solution = construct_solution(aircrafts, pheromone, separation_time, alpha, beta, sigma)  # Construct ant solution
            cost = calculate_cost(solution, aircrafts)  # Calculate cost of the solution
            solutions.append(solution)  # Add solution and cost to the respective lists
            costs.append(cost)

            if cost < best_cost:  # Update best solution if the current solution is better
                best_solution = solution
                best_cost = cost

        pheromone = update_pheromone(pheromone, solutions, costs, rho)  # Update pheromone matrix

    return best_solution, best_cost


def construct_solution(aircrafts, pheromone, separation_time, alpha, beta, sigma):
    num_aircrafts = len(aircrafts)
    solution = []  # Initialize solution list
    unvisited = list(range(num_aircrafts))  # List of unvisited aircrafts

    start_aircraft = random.choice(unvisited)  # Randomly choose a start aircraft
    solution.append(start_aircraft)  # Add the start aircraft to the solution
    unvisited.remove(start_aircraft)  # Remove start aircraft from unvisited list

    while unvisited:  # Continue until all aircrafts are visited
        current_aircraft = solution[-1]  # Get the last visited aircraft
        probabilities = calculate_probabilities(current_aircraft, unvisited, pheromone, separation_time, alpha, beta)  # Calculate probabilities to choose the next aircraft
        next_aircraft = choose_next_aircraft(probabilities, unvisited)  # Choose the next aircraft based on the probabilities
        solution.append(next_aircraft)  # Add the next aircraft to the solution
        unvisited.remove(next_aircraft)  # Remove the next aircraft from the unvisited list

    return solution


def calculate_probabilities(current_aircraft, unvisited, pheromone, separation_time, alpha, beta):
    pheromone_sum = 0.0
    probabilities = []  # List to store probabilities for each unvisited aircraft

    for aircraft in unvisited:
        st = separation_time[current_aircraft][aircraft]  # Separation time between current and unvisited aircraft
        # 0.0000001 is added to avoid division by zero error in case the st is equal to zero
        visibility = 1 / (st + 0.0000001)  # Calculate visibility (inverse of separation time)
        pheromone_value = pheromone[current_aircraft][aircraft] ** alpha  # Pheromone value between current and unvisited aircraft
        heuristic = visibility ** beta  # Heuristic value (visibility) raised to the power of beta
        probabilities.append(pheromone_value * heuristic)  # Calculate probability as product of pheromone and heuristic values
        pheromone_sum += pheromone_value * heuristic  # Calculate sum of pheromone values for normalization

    probabilities = [p / pheromone_sum for p in probabilities]  # Normalize probabilities
    return probabilities


def choose_next_aircraft(probabilities, unvisited):
    if random.random() < sigma:  # Exploitation: Choose aircraft with maximum probability
        max_probability = max(probabilities)
        max_index = probabilities.index(max_probability)
        next_aircraft = unvisited[max_index]
    else:  # Exploration: Randomly choose aircraft based on probabilities
        next_aircraft = random.choices(unvisited, probabilities)[0]

    return next_aircraft


def calculate_cost(solution, aircrafts):
    total_cost = 0.0

    for i in range(len(solution)):
        current_aircraft = solution[i]  # Get current aircraft
        current_time = sum([aircrafts[solution[j]]['S'][i] for j in range(i)])  # Calculate current time based on separation times
        appearance_time = aircrafts[current_aircraft]['A']  # Get appearance time of current aircraft
        earliest_time = aircrafts[current_aircraft]['E']  # Get earliest landing time of current aircraft
        target_time = aircrafts[current_aircraft]['T']  # Get target landing time of current aircraft
        latest_time = aircrafts[current_aircraft]['L']  # Get latest landing time of current aircraft
        penalty_before = aircrafts[current_aircraft]['G']  # Get penalty per unit time if landing before target time
        penalty_after = aircrafts[current_aircraft]['H']  # Get penalty per unit time if landing after target time
            
        if current_time < target_time:
            total_cost += (target_time - current_time) * penalty_before  # Add penalty if landing before target time
        elif current_time > target_time:
            total_cost += (current_time - target_time) * penalty_after  # Add penalty if landing after the target time

        current_time += aircrafts[current_aircraft]['S'][i]  # Update current time based on separation time

    return total_cost


def update_pheromone(pheromone, solutions, costs, rho):
    num_aircrafts = len(pheromone)
    updated_pheromone = [[0] * num_aircrafts for _ in range(num_aircrafts)]  # Initialize updated pheromone matrix

    for solution, cost in zip(solutions, costs):
        for i in range(len(solution) - 1):
            current_aircraft = solution[i]  # Get current aircraft
            next_aircraft = solution[i + 1]  # Get next aircraft
            updated_pheromone[current_aircraft][next_aircraft] += 1 / (cost+0.0000001)  # Update pheromone based on the solution cost

    for i in range(num_aircrafts):
        for j in range(num_aircrafts):
            pheromone[i][j] = (1 - rho) * pheromone[i][j] + rho * updated_pheromone[i][j]  # Update pheromone with evaporation and deposit

    return pheromone


def calculate_separation_time(aircraft):
    num_aircrafts = len(aircraft)
    separation_time = [[0] * num_aircrafts for _ in range(num_aircrafts)]  # Initialize separation time matrix

    for i in range(num_aircrafts):
        for j in range(num_aircrafts):
            separation_time[i][j] = aircraft[i]['S'][j]  # Set separation time between different aircrafts

    return separation_time

#========================================================================================================================
aircraft = [
    {'A': 0, 'E': 75, 'T': 82, 'L': 486, 'G': 30.00, 'H': 30.00, 'S': [99999, 15, 15, 8, 15, 8, 15, 8, 8, 8, 8, 8, 15, 15, 15, 15, 15, 15, 8, 8]},
    {'A': 82, 'E': 157, 'T': 197, 'L': 628, 'G': 10.00, 'H': 10.00, 'S': [15, 99999, 3, 15, 3, 15, 3, 15, 15, 15, 15, 15, 3, 3, 3, 3, 3, 3, 15, 15]},
    {'A': 59, 'E': 134, 'T': 160, 'L': 561, 'G': 10.00, 'H': 10.00, 'S': [15, 3, 99999, 15, 3, 15, 3, 15, 15, 15, 15, 15, 3, 3, 3, 3, 3, 3, 15, 15]},
    {'A': 28, 'E': 103, 'T': 117, 'L': 565, 'G': 30.00, 'H': 30.00, 'S': [8, 15, 15, 99999, 15, 8, 15, 8, 8, 8, 8, 8, 15, 15, 15, 15, 15, 15, 8, 8]},
    {'A': 126, 'E': 201, 'T': 261, 'L': 735, 'G': 10.00, 'H': 10.00, 'S': [15, 3, 3, 15, 99999, 15, 3, 15, 15, 15, 15, 15, 3, 3, 3, 3, 3, 3, 15, 15]},
    {'A': 20, 'E': 95, 'T': 106, 'L': 524, 'G': 30.00, 'H': 30.00, 'S': [8, 15, 15, 8, 15, 99999, 15, 8, 8, 8, 8, 8, 15, 15, 15, 15, 15, 15, 8, 8]},
    {'A': 110, 'E': 185, 'T': 229, 'L': 664, 'G': 10.00, 'H': 10.00, 'S': [15, 3, 3, 15, 3, 15, 99999, 15, 15, 15, 15, 15, 3, 3, 3, 3, 3, 3, 15, 15]},
    {'A': 23, 'E': 98, 'T': 108, 'L': 523, 'G': 30.00, 'H': 30.00, 'S': [8, 15, 15, 8, 15, 8, 15, 99999, 8, 8, 8, 8, 15, 15, 15, 15, 15, 15, 8, 8]},
    {'A': 42, 'E': 117, 'T': 132, 'L': 578, 'G': 30.00, 'H': 30.00, 'S': [8, 15, 15, 8, 15, 8, 15, 8, 99999, 8, 8, 8, 15, 15, 15, 15, 15, 15, 8, 8]},
    {'A': 42, 'E': 117, 'T': 130, 'L': 569, 'G': 30.00, 'H': 30.00, 'S': [8, 15, 15, 8, 15, 8, 15, 8, 8, 99999, 8, 8, 15, 15, 15, 15, 15, 15, 8, 8]},
    {'A': 57, 'E': 132, 'T': 149, 'L': 615, 'G': 30.00, 'H': 30.00, 'S': [8, 15, 15, 8, 15, 8, 15, 8, 8, 8, 8, 99999, 15, 15, 15, 15, 15, 15, 8, 8]},
    {'A': 39, 'E': 114, 'T': 126, 'L': 551, 'G': 30.00, 'H': 30.00, 'S': [8, 15, 15, 8, 15, 8, 15, 8, 8, 8, 8, 99999, 15, 15, 15, 15, 15, 15, 8, 8]},
    {'A': 186, 'E': 261, 'T': 336, 'L': 834, 'G': 10.00, 'H': 10.00, 'S': [15, 3, 3, 15, 3, 15, 3, 15, 15, 15, 15, 15, 99999, 3, 3, 3, 3, 3, 15, 15]},
    {'A': 175, 'E': 250, 'T': 316, 'L': 790, 'G': 10.00, 'H': 10.00, 'S': [15, 3, 3, 15, 3, 15, 3, 15, 15, 15, 15, 15, 3, 99999, 3, 3, 3, 3, 15, 15]},
    {'A': 139, 'E': 214, 'T': 258, 'L': 688, 'G': 10.00, 'H': 10.00, 'S': [15, 3, 3, 15, 3, 15, 3, 15, 15, 15, 15, 15, 3, 3, 99999, 3, 3, 3, 15, 15]},
    {'A': 235, 'E': 310, 'T': 409, 'L': 967, 'G': 10.00, 'H': 10.00, 'S': [15, 3, 3, 15, 3, 15, 3, 15, 15, 15, 15, 15, 3, 3, 3, 99999, 3, 3, 15, 15]},
    {'A': 194, 'E': 269, 'T': 338, 'L': 818, 'G': 10.00, 'H': 10.00, 'S': [15, 3, 3, 15, 3, 15, 3, 15, 15, 15, 15, 15, 3, 3, 3, 3, 99999, 3, 15, 15]},
    {'A': 162, 'E': 237, 'T': 287, 'L': 726, 'G': 10.00, 'H': 10.00, 'S': [15, 3, 3, 15, 3, 15, 3, 15, 15, 15, 15, 15, 3, 3, 3, 3, 3, 99999, 15, 15]},
    {'A': 69, 'E': 144, 'T': 160, 'L': 607, 'G': 30.00, 'H': 30.00, 'S': [8, 15, 15, 8, 15, 8, 15, 8, 8, 8, 8, 8, 15, 15, 15, 15, 15, 15, 99999, 8]},
    {'A': 76, 'E': 151, 'T': 169, 'L': 624, 'G': 30.00, 'H': 30.00, 'S': [8, 15, 15, 8, 15, 8, 15, 8, 8, 8, 8, 8, 15, 15, 15, 15, 15, 15, 8, 99999]}
]
#========================================================================================================================


num_ants = 10
num_iterations = 100
alpha = 1
beta = 2
rho = 0.1
sigma = 0.9

start_time = time.time()

best_solution, best_cost = solve_aircraft_landing(aircraft, num_ants, num_iterations, alpha, beta, rho, sigma)

end_time = time.time()

for i in range(len(best_solution)):
    best_solution[i] = best_solution[i] + 1

print('\n\nBest solution:', best_solution)
print('Best cost:', best_cost)

execution_time = end_time - start_time

print(f"Execution time: {execution_time} seconds\n\n")