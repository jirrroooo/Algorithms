'''
    CMSC 170 - ST1L | Introduction to Artificial Intelligence
    Simple Genetic Algorithm that Solves for the Travelling Sales Person Problem
    Programmer: John Rommel B. Octavo
'''

import random

# This will compute for the sum of the distances between all the cities in the given path.
def fitness_function(path):
    path = list(path)
    sum = 0
    for i in range(path_length):
        if(i != path_length-1):
            sum += adjacency_matrix[path[i]-1][path[i+1]-1]
        else:
            sum += adjacency_matrix[path[i]-1][path[0]-1]
            
    return sum


# This will randomly generate paths based on the given number of nodes/cities
def create_initial_population(num_nodes):
    pop = []
    for i in range(POPULATION_SIZE):
        path = []
        while len(path) != num_nodes:
            x = random.randint(1, num_nodes)
            if x not in path:
                path.append(x)
        pop.append(path)
    return pop


# This will evaluate sum of distances in each path in the population
def evaluate_population(list_pop):
    fitness_scores = []
    for path in list_pop:
        fitness_scores.append(fitness_function(path))
    return fitness_scores


# This will randomly select two paths in the population
def select_parents(list_pop):
    parents = []
    while(len(parents) != 2):
        x = random.randint(1, POPULATION_SIZE)
        if(x not in parents):
            parents.append(list_pop[x-1])
    return parents


# This will perform the cycle crossover on the given parents
def cycle_crossover(parents):
    child1 = parents[0]
    child2 = parents[1]

    index = random.randint(1, len(parents[0])) - 1
    preserve_index = index
    n_index = -1

    while preserve_index != n_index:
        temp = child1[index]

        try:
            n_index = child2.index(temp)
        except:
            child1[index] = child2[index]
            child2[index] = temp
            break

        child1[index] = child2[index]
        child2[index] = temp

        index = n_index
    
    return child1, child2


# This will alter two randomly selected nodes from a randomly selected path from the population
def mutate(list_pop):
    rand_index = random.randint(0, len(list_pop)-1)

    mutated_path = list_pop[rand_index]

    index_node1, index_node2 = -1, -1

    while(index_node1 == index_node2):
        index_node1 = random.randint(1, path_length-1)
        index_node2 = random.randint(1, path_length-1)

    if random.random() < MUTATION_RATE:
        temp = mutated_path[index_node1]
        mutated_path[index_node1] = mutated_path[index_node2]
        mutated_path[index_node2] = temp

    return(mutated_path)


# This will serve as the primary function for the computation of the best route and travel cost
def calculate_tsp():
    # Create initial population
    population = create_initial_population(path_length)

    # This is done for each generations
    for i in range(GENERATIONS):
        # Evaluate the paths in the population
        fitness_scores = evaluate_population(population)

        # Select the path with the least travel cost
        best_answer_set = min(population, key=lambda answer_set: fitness_function(x for x in answer_set))
        best_x = [x for x in best_answer_set]
        best_fitness = fitness_function(best_x)
        print("Generation {}: Best solution found: nodes = {}, Travel Cost (From city A back to A)  = {}".format(i+1, best_x, best_fitness))

        # Give higher possibility of survival for the best paths with lesser travel cost
        elite_population = []
        elite_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k])[:ELITE_SIZE]

        for index in elite_indices:
            elite_population.append(population[index])

        # Generate new population for the next generation
        next_population = elite_population.copy()
        while len(next_population) < POPULATION_SIZE:
            parents = select_parents(population)

            child1, child2 = cycle_crossover(parents)

            next_population.append(child1)
            if len(next_population) < POPULATION_SIZE:
                next_population.append(child2)
            
        mutate(next_population)

        # Replace the current population with the next generation
        population = next_population


# This is the Main Function of the Program

# Constant Variable Declaration
POPULATION_SIZE = 100
MUTATION_RATE = 0.01
ELITE_SIZE = 2
GENERATIONS = 500

#====================================================================
adjacency_matrix = [
    [0, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230],
    [220, 0, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240],
    [221, 231, 0, 241, 242, 243, 244, 245, 246, 247, 248, 249],
    [222, 232, 241, 0, 250, 251, 252, 253, 254, 255, 256, 257],
    [223, 233, 242, 250, 0, 258, 259, 260, 261, 262, 263, 264],
    [224, 234, 243, 251, 258, 0, 265, 266, 267, 268, 269, 270],
    [225, 235, 244, 252, 259, 265, 0, 271, 272, 273, 274, 275],
    [226, 236, 245, 253, 260, 266, 271, 0, 276, 277, 278, 279],
    [227, 237, 246, 254, 261, 267, 272, 276, 0, 280, 281, 282],
    [228, 238, 247, 255, 262, 268, 273, 277, 280, 0, 283, 284],
    [229, 239, 248, 256, 263, 269, 274, 278, 281, 283, 0, 285],
    [230, 240, 249, 257, 264, 270, 275, 279, 282, 284, 285, 0]
]
#===================================================================

path_length = len(adjacency_matrix)

calculate_tsp()