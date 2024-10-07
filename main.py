import random
import numpy as np
import pandas as pd

class DE_Individual:
    def __init__(self, dimensions, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], dimensions)
        self.fitness = None

class PSO_Individual:
    def __init__(self, dimensions, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], dimensions)
        self.velocity = np.random.uniform(-1, 1, dimensions)
        self.best_position = self.position.copy()
        self.best_score = float('inf')

def objective_fun1(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def objective_fun2(x):
    # Implement Griewanks function
    sum = 0
    prod = 1
    for i in range(len(x)):
        sum += x[i]**2/4000
        prod *= np.cos(x[i]/np.sqrt(i+1))
    return sum - prod + 1

def PSO(parameters):
    # Unpack parameters
    generations, dim, bounds, seed, obj_no, population_size = parameters

    # Set random seed
    random.seed(seed)

    # INITIALIZATION: Initialize population
    PSO_population = [PSO_Individual(dim, bounds) for _ in range(population_size)]
    best_individual = None
    best_fitness = None
    
    # PSO parameters
    w = 0.7298  # Inertia weight
    c1 = 1.49618  # Cognitive weight
    c2 = 1.49618  # Social weightt

    # Evolution loop
    for generation in range(generations):

        for particle in range(population_size):
            # Evaluate the fitness of each individual in the population
            PSO_population[particle].fitness = objective_fun1(PSO_population[particle].position) if obj_no == 1 else objective_fun2(PSO_population[particle].position)

            # Update the Perspnal best position for each individual
            if PSO_population[particle].fitness < PSO_population[particle].best_score:
                PSO_population[particle].best_position = PSO_population[particle].position
                PSO_population[particle].best_score = PSO_population[particle].fitness

            # Update the Global best position for the population
            if best_individual is None or PSO_population[particle].fitness < best_individual.fitness:
                best_individual = PSO_population[particle]
                best_fitness = PSO_population[particle].fitness

        # Update the position and velocity of each individual in the population
        for particle in range(population_size):
            r1 = random.random()
            r2 = random.random()
            PSO_population[particle].velocity = w * PSO_population[particle].velocity + c1 * r1 * (PSO_population[particle].best_position - PSO_population[particle].position) + c2 * r2 * (best_individual.position - PSO_population[particle].position)
            PSO_population[particle].position = PSO_population[particle].position + PSO_population[particle].velocity
    
    return best_individual, best_fitness

def DE(parameters):
    # Unpack parameters
    generations, dim, bounds, seed, obj_no, population_size = parameters

    # Set random seed
    random.seed(seed)

    # INITIALIZATION: Initialize population
    DE_population = [DE_Individual(dim, bounds) for _ in range(population_size)]
    #variance = np.random.uniform(low=0, high=1, size=(mu, dim))
    best_individual = None
    best_fitness = None
    CR = 0.7  # Crossover rate (CR)
    scale_factor = 3.0 # Mutation factor (F)

    # Evolution loop
    for generation in range(generations):
        for i in range(population_size):
            # Evaluate the fitness of each individual in the population
            DE_population[i].fitness = objective_fun1(DE_population[i].position) if obj_no == 1 else objective_fun2(DE_population[i].position)

            # MUTATION: Mutation
            """ a, b, c = np.random.choice(DE_population, 3, replace=False)
            mutant = DE_Individual(dim, bounds)
            mutant.position = a.position + scale_factor * (b.position - c.position) """
            a, b, c, d, e = np.random.choice(DE_population, 5, replace=False)
            mutant = DE_Individual(dim, bounds)
            mutant.position = a.position + scale_factor * (b.position - c.position + d.position - e.position)

            # CROSSOVER: Crossover
            trial = DE_Individual(dim, bounds)
            for j in range(dim):
                if np.random.rand() < CR:
                    trial.position[j] = mutant.position[j]
                else:
                    trial.position[j] = DE_population[i].position[j]

            # SELECTION: Replace the current individual with the trial if it has a better fitness 
            trial.fitness = objective_fun1(DE_population[i].position) if obj_no == 1 else objective_fun2(DE_population[i].position)
            if trial.fitness < DE_population[i].fitness:
                DE_population[i] = trial

        # Update the best individual
        for individual in DE_population:
            if best_individual is None or individual.fitness < best_individual.fitness:
                best_individual = individual
                best_fitness = individual.fitness
    
    return best_individual, best_fitness

def print_summary(runs, fitness_obj1, fitness_obj2):
    # First column
    #runs_column = []
    #for i in range(runs):
    #    runs_column.append(f'Run {i+1}')
    
    # Mean and standard deviation
    mean_obj1 = np.mean(fitness_obj1)
    std_obj1 = np.std(fitness_obj1)
    mean_obj2 = np.mean(fitness_obj2)
    std_obj2 = np.std(fitness_obj2)

    # Create a dictionary with the two lists as values
    #data = {'Fun': '', 'Fitness Obj 1': '', 'Fitness Obj 2': ''}
    data = {'': ['Mean'], 'Fitness Obj 1': [mean_obj1], 'Fitness Obj 2': [mean_obj2]}

    # Create a pandas DataFrame from the dictionary
    data_table = pd.DataFrame(data)

    # Create a new DataFrame with the mean and concatenate it with (data_table)
    #mean_row = pd.DataFrame({'': ['Mean'], 'Fitness Obj 1': [mean_obj1], 'Fitness Obj 2': [mean_obj2]})
    #data_table = pd.concat([data_table, mean_row], ignore_index=True)

    # Create a new DataFrame with the stander deviation and concatenate it with (data_table)
    std_row = pd.DataFrame({'': ['STD'], 'Fitness Obj 1': [std_obj1], 'Fitness Obj 2': [std_obj2]})
    data_table = pd.concat([data_table, std_row], ignore_index=True)

    return data_table

def main():
    dimention_li = [20, 50]  # Number of dimensions
    generations = 50  # Number of generations
    times = 5  # Number of runs
    bounds = [-30, 30]  # Search space
    population_size = 100  # Population size

    # generate 30 random seeds with determine incremental value
    seeds = [i+2 for i in range(times)]

    # Iterate over objective functions, once for each objective function
    fitness_obj1_d20 = np.zeros((times,2)) # Store the fitness values for objective function 1
    fitness_obj1_d50 = np.zeros((times,2))
    fitness_obj2_d20 = np.zeros((times,2)) # Store the fitness values for objective function 2
    fitness_obj2_d50 = np.zeros((times,2))

    for i in range(2):
        print(f"Objective function {i+1}...", end="\n")
        # for EP optmization algorithm     
        for run in range(times):
            print('--------------------------------------')
            print(f"Run {run+1}/{times}...", end="\n")

            for dim in dimention_li:
                PSO_parameters = [generations, dim, bounds, seeds[run], i+1, population_size]  # (i) objective function number, 1 = objective_fun1, 2 = objective_fun2
                PSO_best_individual, PSO_best_fitness = PSO(PSO_parameters)

                if i == 0:
                    if(dim == 20):
                        fitness_obj1_d20[run][0] = PSO_best_fitness
                    else:
                        fitness_obj1_d50[run][0] = PSO_best_fitness
                else:
                    if(dim == 20):
                        fitness_obj2_d20[run][0] = PSO_best_fitness
                    else:
                        fitness_obj2_d50[run][0] = PSO_best_fitness

                print(f"PSO:  Dimension {dim}, Run {run+1}/{times}: Best fitness: {PSO_best_fitness}")
                print()

            for dim in dimention_li:
                DE_parameters = [generations, dim, bounds, seeds[run], i+1, population_size]  # (i) objective function number, 1 = objective_fun1, 2 = objective_fun2
                DE_best_individual, DE_best_fitness = DE(DE_parameters)

                if i == 0:
                    if(dim == 20):
                        fitness_obj1_d20[run][1] = DE_best_fitness
                    else:
                        fitness_obj1_d50[run][1] = DE_best_fitness
                else:
                    if(dim == 20):
                        fitness_obj2_d20[run][1] = DE_best_fitness
                    else:
                        fitness_obj2_d50[run][1] = DE_best_fitness

                print(f"EP: Dimension {dim}, Run {run+1}/{times}: Best fitness: {DE_best_fitness}")
                print()

    # Print the summary
    PSO_d20_table = print_summary(times, [item[0] for item in fitness_obj1_d20], [item[0] for item in fitness_obj2_d20])
    PSO_d20_table = print_summary(times, [item[0] for item in fitness_obj1_d50], [item[0] for item in fitness_obj2_d50])
    DE_d20_table = print_summary(times, [item[1] for item in fitness_obj1_d20], [item[1] for item in fitness_obj2_d20])
    DE_d50_table = print_summary(times, [item[1] for item in fitness_obj1_d50], [item[1] for item in fitness_obj2_d50])

    print("PSO: Objective function 1, Dimension 20")
    print(PSO_d20_table)
    print()
    print("PSO: Objective function 1, Dimension 50")
    print(PSO_d20_table)
    print()
    print("DE: Objective function 2, Dimension 20")
    print(DE_d20_table)
    print()
    print("DE: Objective function 2, Dimension 50")
    print(DE_d50_table)
    
    
                

if __name__ == "__main__":
    main()