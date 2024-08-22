'''
    CMSC 170 - ST1L | Introduction to Artificial Intelligence
    Simple Genetic Algorithm that Solves for the Five De Jong's Functions
    Programmer: John Rommel B. Octavo
'''

import math
import random

# This is the five De Jong's functions. They serve as the fitness function of the problem
def fitness_function(x_values):
    match func_num:
        
        case 1: # De Jong's Function number 1
            x = list(x_values)
            return sum(x**2 for x in x_values)
        
        case 2:  # De Jong's Function number 2
            x = list(x_values)
            return 100*((x[0]**2)-(x[1]))**2 + (1-x[0])**2
        
        case 3: # De Jong's Function number 3
            return sum(math.floor(x) for x in x_values)
        
        case 4:  # De Jong's Function number 4
            x = list(x_values)
            fitness_value = 0
    
            for i in range(1, len(x)+1):
                fitness_value += i*x[i-1] + random.gauss(0, math.sqrt(1))
    
            return fitness_value
        
        case 5: # De Jong's Function number 5
            x = list(x_values)
    
            a = [[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
                [-32, -32, -32, -32, -32, -16, -16, -16, 32, 32, 32, -32, -32, -32, -32, -16, -16, -16, 32, 32, 32, 32, 32, 32, 32]]

            fitness_value = 0
    
            for j in range(1, 26):
                fitness_value += 1/(j+(((x[0]-a[0][j-1])**6)+((x[1]-a[1][j-1])**6)))
    
            return (0.002 + fitness_value)


# This function is for creating the initial population
def create_initial_population():
    population = []
    for i in range(POPULATION_SIZE):
        answer_set = []
        for j in range(num_variables):
            person = ""
            for k in range(PERSON_INFO_LENGTH):
                person += str(random.randint(0, 1))
            answer_set.append(person)
        population.append(answer_set)
        
    # print("initial population: " + str(population))
    return population


# This function uses ratio and proportion to decode the value of an individual
def decode_person(person):
    decoded_number = int(person, 2)
    number_of_data_sets = 2**PERSON_INFO_LENGTH
    ratio = (max_range-min_range)/number_of_data_sets

    adjusted_number = min_range + (ratio*decoded_number)

    return adjusted_number


# This function is for evaluating the population based on the fitness function
def evaluate_population(population):
    fitness_scores = []
    for set_answer in population:
        arr_answers = []
        for person in set_answer:
            x = decode_person(person)
            arr_answers.append(x)
        fitness_scores.append(fitness_function(arr_answers))
    return fitness_scores


# This function randomly selects two individual (parents) from the population
def select_parents(population, fitness_scores):
    parents = []
    total_fitness = sum(fitness_scores)
    for i in range(2):
        pick = random.uniform(0, total_fitness)
        current = 0
        for j in range(POPULATION_SIZE):
            current += fitness_scores[j]
            if current > pick:
                parents.append(population[j])
                break
    return parents


# This function performs crossover of data from both parents
def crossover(parents):
    child1 = []
    child2 = []
    crossover_point = random.randint(0, PERSON_INFO_LENGTH - 1)
    
    for i in range(0, num_variables):
        child1.append((parents[0][i])[:crossover_point] + (parents[1][i])[crossover_point:])
        child2.append((parents[1][i])[:crossover_point] + (parents[0][i])[crossover_point:])

    return child1, child2


# This function performs mutation or alteration of data if certain condition is met
def mutate(answer_set):
    mutated_answers = []
    for person in answer_set:
        mutated_person = ""
        for bit in person:
            if random.random() < MUTATION_RATE:
                mutated_person += '0' if bit == '1' else '1'
            else:
                mutated_person += bit
        mutated_answers.append(mutated_person)
    # print(mutated_answers)
    return mutated_answers


# This function is responsible for calculating the best fit answer.
# All the other functions are called here.
def calculate():
    # Create the initial population
    population = create_initial_population()

    # Run the genetic algorithm for a defined number of generations
    for i in range(generations):
        # Evaluate the fitness of the population
        fitness_scores = evaluate_population(population)

        print("---------------------------------------------")
        # Print the best solution found so far
        if find_max:
            best_answer_set = max(population, key=lambda answer_set: fitness_function(decode_person(x) for x in answer_set))
        else:
            best_answer_set = min(population, key=lambda answer_set: fitness_function(decode_person(x) for x in answer_set))
        best_x = [decode_person(x) for x in best_answer_set]
        best_fitness = fitness_function(best_x)
        print("Generation {}: Best solution found: x = {}, f(x) = {}".format(i+1, best_x, best_fitness))

        # Select the elite individuals. Crucial in making each generation better.
        elite_population = []
        if find_max:
            elite_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k], reverse=True)[:ELITE_SIZE]
        else:
            elite_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k])[:ELITE_SIZE]
        for index in elite_indices:
            elite_population.append(population[index])

        # Generate the next generation of individuals
        next_population = elite_population.copy()
        while len(next_population) < POPULATION_SIZE:
            parents = select_parents(population, fitness_scores)
            while len(parents) < 2:
                parents = select_parents(population, fitness_scores)
            child1, child2 = crossover(parents)
            child1 = mutate(child1)
            child2 = mutate(child2)
            next_population.append(child1)
            if len(next_population) < POPULATION_SIZE:
                next_population.append(child2)

        # Replace the current population with the next generation
        population = next_population


# This is the main function of the program

# Constant Variable Declaration
POPULATION_SIZE = 100
PERSON_INFO_LENGTH = 10
MUTATION_RATE = 0.1
ELITE_SIZE = 2

# Global Variables Initialization
generations = 0
find_max = True
num_variables = 0
min_range = 0
max_range = 0
func_num = 0

# Screen Prompts
print("\n===================================================")
print(" De Jong's Function Solver Using Genetic Algorithm")
print("===================================================")


print("\n========== Select Function To Solve ==========")
print("\t1.) De Jong's Function Number 1")
print("\t2.) De Jong's Function Number 2")
print("\t3.) De Jong's Function Number 3")
print("\t4.) De Jong's Function Number 4")
print("\t5.) De Jong's Function Number 5")

menu_choice = int(input("\nEnter choice: "))

# Evaluate Menu Choice
match menu_choice:
    case 1:
        func_num = 1
    case 2:
        func_num = 2
    case 3:
        func_num = 3
    case 4:
        func_num = 4
    case 5:
        func_num = 5

# For the inputtung the number of generations
generations = int(input("Enter number of Generations: "))
if(generations <= 0):
    func_num = 0

# This is for the properly assigning of values to the variables and to start the computation
match func_num:
    case 1:
        find_max = False
        num_variables = 3
        min_range = -5.12
        max_range = 5.12
        calculate()
    case 2:
        find_max = False
        num_variables = 2
        min_range = -2.048
        max_range = 2.048
        calculate()
    case 3:
        find_max = False
        num_variables = 5
        min_range = -5.12
        max_range = 5.12
        calculate()
    case 4:
        find_max = False
        num_variables = 50
        min_range = -1.28
        max_range = 1.28
        calculate()
    case 5:
        find_max = True
        num_variables = 2
        min_range = -65.536
        max_range = 65.536
        calculate()
    case default:
        print("\nInvalid Input!!!\n")

# End of the code
#--------------------------------------------------------------------------------------------------