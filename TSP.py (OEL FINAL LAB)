import random
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Define parameters
population_size = 50
mutation_rate = 0.01
generations = 200
city_count = 20

# Generate random cities (each with x, y coordinates)
cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(city_count)]

# Calculate the distance between two cities
def distance(city1, city2):
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

# Calculate the total distance of the route
def route_distance(route):
    return sum(distance(cities[route[i]], cities[route[i + 1]]) for i in range(len(route) - 1)) + distance(cities[route[-1]], cities[route[0]])

# Initialize population with random routes
def initialize_population():
    return [random.sample(range(city_count), city_count) for _ in range(population_size)]

# Fitness function (we aim to minimize the route distance, so take the reciprocal)
def fitness(route):
    return 1 / route_distance(route)

# Selection based on fitness proportionate selection (roulette wheel)
def select(population):
    fitness_scores = [fitness(route) for route in population]
    total_fitness = sum(fitness_scores)

    # Avoid division by zero if all fitness scores are zero
    if total_fitness == 0:
        return random.sample(population, min(len(population), population_size // 2))

    # Calculate probabilities
    probabilities = [score / total_fitness for score in fitness_scores]

    # Ensure probabilities sum up to 1
    if len(probabilities) != len(population):
        raise ValueError("The length of probabilities and population must be the same.")
    
    selected_indices = np.random.choice(range(len(population)), size=population_size // 2, p=probabilities)
    return [population[i] for i in selected_indices]

# Crossover to combine two parents and create offspring
def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(city_count), 2))
    child = [None] * city_count
    child[start:end] = parent1[start:end]
    position = end
    for city in parent2:
        if city not in child:
            if position >= city_count:
                position = 0
            child[position] = city
            position += 1
    return child

# Mutation to introduce diversity
def mutate(route):
    for _ in range(city_count):
        if random.random() < mutation_rate:
            i, j = random.sample(range(city_count), 2)
            route[i], route[j] = route[j], route[i]  # Swap two cities
    return route

# Main genetic algorithm loop
def genetic_algorithm():
    population = initialize_population()
    best_route = min(population, key=route_distance)
    best_distance = route_distance(best_route)
    distances = []

    for gen in range(generations):
        selected_parents = select(population)
        offspring = []

        for i in range(0, len(selected_parents), 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[(i + 1) % len(selected_parents)]
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            offspring.extend([mutate(child1), mutate(child2)])

        population = selected_parents + offspring
        current_best = min(population, key=route_distance)
        current_distance = route_distance(current_best)

        # Update best solution
        if current_distance < best_distance:
            best_route, best_distance = current_best, current_distance

        distances.append(best_distance)
        print(f"Generation {gen + 1}: Best Distance = {best_distance:.2f}")

    return best_route, distances

# Visualization of the TSP route
def plot_route(route, distances):
    # Plot the best route
    x, y = zip(*[cities[i] for i in route] + [cities[route[0]]])
    plt.figure(figsize=(10, 5))

    # Plot the cities and route
    plt.subplot(1, 2, 1)
    plt.plot(x, y, marker="o")
    plt.title("Optimal TSP Route")

    # Plot the distance progress
    plt.subplot(1, 2, 2)
    plt.plot(distances)
    plt.title("Distance Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Distance")
    plt.show()

# Run the genetic algorithm and visualize results
best_route, distances = genetic_algorithm()
plot_route(best_route, distances)
