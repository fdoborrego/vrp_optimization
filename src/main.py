# Script principal: resolución del problema CVRP/VRPTW.

import os
import utils
import solver
import logging

# Logger
logger_main = logging.getLogger('evaluation_logger')
logger_main.disabled = False                                       # Se habilita/deshabilita el logger
logger_main.setLevel(logging.INFO)                                 # DEBUG, INFO, WARNING, ERROR, EXCEPTION, CRITICAL

consoleh = logging.StreamHandler()
formatterConsole = logging.Formatter('%(levelname)s - %(message)s')
consoleh.setFormatter(formatterConsole)                             # Formato de salida
logger_main.addHandler(consoleh)

path = '../log'
if not os.path.exists(path):
    os.mkdir(path)
fileh = logging.FileHandler(path + '/main_log.log')
fileh.setLevel(logging.INFO)
formatterFile = logging.Formatter('%(levelname)s - %(message)s')
fileh.setFormatter(formatterFile)
logger_main.addHandler(fileh)


def main():

    # Lectura de datos
    data, max_capacity, max_time, max_vehicles, depots = utils.read_file("../data/Datos.txt", 30)
    max_timer = 5

    # Definición del problema
    tsp = solver.TSPSolver(data, depots=depots)
    tsp.set_timer(max_timer)

    cvrp = solver.CVRPSolver(data, max_capacity=max_capacity, max_time=max_time, max_vehicles=max_vehicles,
                             depots=depots, cost_vehicles=100, cost_km=10, velocity=60)
    cvrp.set_timer(max_timer)

    vrptw = solver.VRPTWSolver(data, max_time=max_time, max_vehicles=max_vehicles, depots=depots,
                               cost_vehicles=100, cost_km=10, velocity=60)
    vrptw.set_timer(max_timer)

    # Resolución del problema
    initial_solution = list(range(1, len(data['customer'])))

    # methods = ['Simulated Annealing', 'Tabu Search']
    methods = ['Bruteforce', 'Nearest Neighbour', 'Cost Saving', 'Local Search (SWAP)', 'Local Search (INSERTION)',
               'VND', 'VNS', 'Simulated Annealing', 'Tabu Search']

    initial_solution, _, _, _ = cvrp.solve(initial_solution, 'Nearest Neighbour')
    for method in methods:
        solution, solution_value, solution_values, dt = cvrp.solve(initial_solution, method)
        cvrp.graph_solution()

    initial_solution, _, _, _ = vrptw.solve(initial_solution, 'Nearest Neighbour')
    for method in methods:
        solution, solution_value, solution_values, dt = vrptw.solve(initial_solution, method)
        vrptw.graph_solution()


if __name__ == "__main__":
    main()
