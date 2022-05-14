# Solver del problema de optimización.

import timer
import logging
import heuristics
import graphbuild


class OptimizationSolver(object):
    """
    <class 'OptimizationSolver'>
    Solver de problemas de optimización. Esta clase desarrolla la estructura general que sigue cualquiera de los
    siguientes problemas de optimización: CVRP, VRPTW.
    """

    # Logger
    logger = logging.getLogger('logger')
    logger.disabled = False  # Se habilita/deshabilita el logger
    logger.setLevel(logging.INFO)  # DEBUG, INFO, WARNING, ERROR, EXCEPTION, CRITICAL

    consoleh = logging.StreamHandler()
    formatterConsole = logging.Formatter('%(levelname)s - %(message)s')
    consoleh.setFormatter(formatterConsole)  # Formato de salida
    logger.addHandler(consoleh)

    fileh = logging.FileHandler('../log/solver_log.log')
    fileh.setLevel(logging.INFO)
    formatterFile = logging.Formatter('%(levelname)s - %(message)s')
    fileh.setFormatter(formatterFile)
    logger.addHandler(fileh)

    # Parámetros
    problem = 'TBD'     # = To Be Defined

    def __init__(self, data, depots=tuple([0]),
                 max_capacity=float('inf'), max_time=float('inf'), max_vehicles=float('inf'),
                 cost_vehicles=0, cost_km=0, velocity=60):

        # Características del problema
        self.data = data
        self.depots = depots
        self.max_capacity = max_capacity
        self.max_time = max_time
        self.max_vehicles = max_vehicles
        self.cost_km = cost_km
        self.cost_vehicles = cost_vehicles
        self.velocity = velocity

        # Inicialización
        self.solution = []
        self.solution_route = []
        self.solution_value = float('inf')
        self.solution_value_history = [float('inf')]
        self.solution_vehicles = []

        self.method = 'None'

        # Constructor de gráficos
        self.graphbuilder = graphbuild.GraphicsBuilder(self.problem, data)

        # Timer
        self.timer = timer.Timer(max_time=5)

    def __repr__(self):
        return "<[%s] C%.0f T%.0f V%.0f>" % (self.problem, self.max_capacity, self.max_time, self.max_vehicles)

    def __str__(self):
        return "<[%s problem] Max Capacity: %.0f; Max Time: %.0f; Max Vehicles: %.0f>" % \
               (self.problem, self.max_capacity, self.max_time, self.max_vehicles)

    def evalfun(self, solution, data):
        """
        Función de evaluación del problema de optmización.

        :param solution: Solución a analizar (formada por los índices de cada cliente a visitar).
        :param data: Estructura de datos del problema.

        :return: solution_value: Valor de la solución (inf si la solución es inadmisible).
        :return: solution: Solución de entrada.
        :return: routes: Solución construida (lista de clientes a satisfacer por cada vehículo).
        :return: vehicles: Estructura de datos con información sobre vehículos: id, capacity, time y distance.
        """
        return float('inf'), [], [], []

    def solve(self, initial_solution, method):
        """
        Aplicación de los distintos algoritmos al problema de optimización

        :param initial_solution: Solución de partida para comenzar el algoritmo.
        :param method: Método a utilizar.

        :return: solution: Mejor solución encontrada por el algoritmo.
        :return: solution_value: Valor de la mejor solución encontrada por el algoritmo.
        :return: history: Histórico de los de valores de las distintas soluciones que atraviesa el algoritmo.
        :return: dt: Tiempo tomado por el algoritmo en el proceso de optimización.
        """

        # Inicialización
        solution = []
        solution_value = 'inf'
        history = []

        # Inicio de temporizador
        self.timer.reset()
        self.timer.start()

        # Algoritmos de optimización
        if method == "Bruteforce":
            solution, solution_value, history = heuristics.bruteforce(self, initial_solution, start=self.depots)

        elif method == "Nearest Neighbour":
            solution, solution_value, history = heuristics.nearest_neighbor(self, initial_solution, start=self.depots)

        elif method == 'Cost Saving':
            solution, solution_value, history = heuristics.cost_saving(self, initial_solution, start=self.depots)

        elif method == 'Local Search (SWAP)':
            solution, solution_value, history = heuristics.local_search(self, initial_solution, operator='swap')

        elif method == 'Local Search (INSERTION)':
            solution, solution_value, history = heuristics.local_search(self, initial_solution, operator='insertion')

        elif method == 'VND':
            solution, solution_value, history = heuristics.vnd(self, initial_solution)

        elif method == 'VNS':
            solution, solution_value, history = heuristics.vns(self, initial_solution)

        elif method == 'Simulated Annealing':
            solution, solution_value, history = heuristics.simulated_annealing(self, initial_solution)

        elif method == 'Tabu Search':
            solution, solution_value, history = heuristics.tabu_search(self, initial_solution)

        else:
            self.logger.error('[' + self.problem + '][SOLVE] El método introducido (' + method + ') no es válido.')
            exit()

        # Fin de temporizador y resultado
        dt = self.timer.stop()
        if self.timer.check():
            self.logger.warning(
                '[' + self.problem + '][SOLVE - ' + method + '] Optimización abortada (' + str(dt) +
                ' s) -> Valor: ' + str(solution_value) + '; Solución: ' + str(solution) + '.')
        else:
            self.logger.info(
                '   [' + self.problem + '][SOLVE - ' + method + '] Optimización finalizada (' + str(dt) +
                ' s) -> Valor: ' + str(solution_value) + '; Solución: ' + str(solution) + '.')

        self.set_solution(solution, history, method)

        return solution, solution_value, history, dt

    def set_solution(self, solution, solution_history, method):
        """
        Método para almacenar la solución obtenida.
        :param solution: Solución actual al problema de optimización.
        :param solution_history: Histórico de soluciones encontradas en el problema de optimización.
        :param method: Método utilizado para la optimización.
        """

        # Se almacena solución
        self.method = method
        self.solution_value_history = solution_history
        self.solution_value, self.solution, self.solution_route, self.solution_vehicles =\
            self.evalfun(solution, self.data)

    def graph_solution(self):
        """
        Representación gráfica de la solución del problema de optimización.
        """

        if self.solution_value == float('inf'):
            self.logger.error('  [' + self.problem + '][GRAPH - ' + self.method +
                              '] Solución no válida. No es posible representar.')
            return

        # Representación
        self.graphbuilder.graph_history(self.solution_value_history, self.method)

        if self.problem == 'TSP':
            self.graphbuilder.graph_routes(self.solution_route, self.depots, self.method)
        elif self.problem == 'CVRP':
            self.graphbuilder.graph_routes(self.solution_route, self.depots, self.method)
            self.graphbuilder.graph_veh_distance(self.solution_vehicles, self.method)
            self.graphbuilder.graph_veh_time(self.solution_vehicles, self.method)
            self.graphbuilder.graph_veh_capacity(self.solution_vehicles, self.method)
        elif self.problem == 'VRPTW':
            self.graphbuilder.graph_routes(self.solution_route, self.depots, self.method)
            self.graphbuilder.graph_veh_distance(self.solution_vehicles, self.method)
            self.graphbuilder.graph_veh_time(self.solution_vehicles, self.method)

    def set_timer(self, max_time):
        """
        Establecimiento de tiempo de ejecución máximo admisible de un algoritmo de optimización.
        :param max_time: Tiempo máximo de ejecución.
        """
        self.timer.set(max_time)

    @staticmethod
    def distance(point1, point2):
        """
        Devuelve la distancia euclídea de dos puntos en el plano cartesiano.
        :param point1: Coordenadas [x, y] del punto 1 (origen).
        :param point2: Coordenadas [x, y] del punto 2 (destino).

        :return: Distancia euclídea entre point1 y point2.
        """
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


class TSPSolver(OptimizationSolver):

    problem = 'TSP'    # Travelling Salesman Problem

    def evalfun(self, solution, data):
        """
        Devuelve la distancia euclídea entre una lista de puntos (considerando vuelta al inicio).
        :param solution: Solución a analizar (formada por los índices de cada cliente a visitar).
        :param data: Estructura de datos del problema.

        :return: solution_value: Valor de la solución (inf si la solución es inadmisible).
        :return: solution: Solución de entrada.
        :return: routes: Solución construida (lista de clientes a satisfacer por cada vehículo).
        :return: vehicles: Estructura de datos con información sobre vehículos: id, capacity, time y distance (VACÍO).
        """

        # Inicialización
        routes = self.depots + solution

        # Recorrido de principio a fin
        dist = 0
        for i in range(len(routes) - 1):
            dist += self.distance([data['x_coord'][routes[i]], data['y_coord'][routes[i]]],
                                  [data['x_coord'][routes[i + 1]], data['y_coord'][routes[i + 1]]])

        # Vuelta al inicio
        solution_value = dist + self.distance([data['x_coord'][routes[-1]], data['y_coord'][routes[-1]]],
                                              [data['x_coord'][routes[0]], data['y_coord'][routes[0]]])

        veh = {'id': 1, 'capacity': 0, 'time': 0, 'distance': solution_value}
        vehicles = [veh]

        self.logger.debug(
            '[TSP] Valor: ' + str(solution_value) + '; Solución: ' + str(solution) + '; Rutas: ' + str(routes) +
            '; Vehículos: ' + str(vehicles))

        return solution_value, solution, routes, vehicles


class CVRPSolver(OptimizationSolver):

    problem = 'CVRP'    # Capacity Vehicle Routing Problem

    def evalfun(self, solution, data):
        """
            Función de evaluación para el problema CVRP (Capacity Vehicle Routing Problem).

            :param solution: Solución a analizar (formada por los índices de cada cliente a visitar).
            :param data: Estructura de datos del problema.

            :return: solution_value: Valor de la solución (inf si la solución es inadmisible).
            :return: solution: Solución de entrada.
            :return: routes: Solución construida (lista de clientes a satisfacer por cada vehículo).
            :return: vehicles: Estructura de datos con información sobre los vehículos: id, capacity, time y distance.
            """
        # Datos de entrada
        cost_vehicles = self.cost_vehicles
        cost_km = self.cost_km
        max_capacity = self.max_capacity
        max_time = self.max_time
        max_vehicles = self.max_vehicles
        velocity = self.velocity

        depot = self.depots[0]  # Ya que únicamente se considera la existencia de un depósito
        depot_pos = (data['x_coord'][depot], data['y_coord'][depot])
        dim_sol = len(solution)

        # Inicialización de solución
        routes = []
        vehicles = []

        # Inicialización de datos
        total_customers = 0
        total_dist = 0
        total_vehicles = 0

        # Evaluación de la solución
        while total_vehicles < max_vehicles:

            # Inicialización del vehículo
            current_time = 0
            current_demand = 0
            current_dist = 0
            current_dist2deposit = 0
            current_pos = depot_pos
            current_route = [depot]

            while total_customers < dim_sol:

                # Datos del siguiente cliente
                next_customer = solution[total_customers]

                next_pos = [data['x_coord'][next_customer], data['y_coord'][next_customer]]
                next_demand = data['demand'][next_customer]
                next_service_time = data['service_time'][next_customer]

                dist2customer = self.distance(current_pos, next_pos)
                dist2deposit = self.distance(next_pos, depot_pos)

                # Comprobación de que es alcanzable (demanda y tiempo)
                if (current_demand + next_demand <= max_capacity) and \
                        (current_time + (dist2customer + dist2deposit) * 60 / velocity + next_service_time <= max_time):
                    total_customers += 1

                    # Actualización de datos
                    current_dist += dist2customer
                    current_demand += next_demand
                    current_time += dist2customer * 60 / velocity + next_service_time
                    current_dist2deposit = dist2deposit

                    current_pos = next_pos
                    current_route.append(next_customer)

                # Si no es alcanzable se utiliza otro vehículo
                else:
                    break

            # Si ha recorrido distancia -> se almacena información
            if current_route != [depot]:
                total_vehicles += 1

                # Vuelta al depósito
                current_time += current_dist2deposit * 60 / velocity
                current_dist += current_dist2deposit

                # Se almacena la información del anterior vehículo
                total_dist += current_dist

                veh = {'id': total_vehicles, 'capacity': current_demand, 'time': current_time, 'distance': current_dist}
                vehicles.append(veh)
                routes.append(current_route)

            # Si el actual vehículo no se ha movido -> fin
            else:
                break

        # Comprobación de solución válida
        if (dim_sol > 0) and (total_customers == dim_sol):
            solution_value = total_vehicles * cost_vehicles + total_dist * cost_km
        elif dim_sol == 0:
            solution_value = float('inf')
        else:  # total_customers != dim_sol
            routes = []
            solution_value = float('inf')

        self.logger.debug(
            '[CVRP] Valor: ' + str(solution_value) + '; Solución: ' + str(solution) + '; Rutas: ' + str(routes) +
            '; Vehículos: ' + str(vehicles))

        return solution_value, solution, routes, vehicles


class VRPTWSolver(OptimizationSolver):

    problem = 'VRPTW'   # Vehicle Routing Problem with Time Windows
    max_capacity = float('inf')

    def evalfun(self, solution, data):
        """
        Función de evaluación para el problema VRPTW (Vehicle Routing Problem with Time Windows).

        :param solution: Solución a analizar (formada por los índices de cada cliente a visitar).
        :param data: Estructura de datos del problema.

        :return: solution_value: Valor de la solución (inf si la solución es inadmisible).
        :return: solution: Solución de entrada.
        :return: routes: Solución construida (lista de clientes a satisfacer por cada vehículo).
        :return: vehicles: Estructura de datos con información sobre los vehículos: id, capacity, time y distance.
        """
        # Datos de entrada
        cost_vehicles = self.cost_vehicles
        cost_km = self.cost_km
        max_time = self.max_time
        max_vehicles = self.max_vehicles
        velocity = self.velocity

        depot = self.depots[0]  # Ya que únicamente se considera la existencia de un depósito
        depot_pos = (data['x_coord'][depot], data['y_coord'][depot])
        dim_sol = len(solution)

        # Inicialización de solución
        routes = []
        vehicles = []

        # Inicialización de datos
        total_customers = 0
        total_dist = 0
        total_vehicles = 0

        # Evaluación de la solución
        while total_vehicles < max_vehicles:

            # Inicialización del vehículo
            current_time = 0
            current_dist = 0
            current_dist2deposit = 0
            current_pos = depot_pos
            current_route = [depot]

            while total_customers < dim_sol:

                # Datos del siguiente cliente
                next_customer = solution[total_customers]

                next_pos = [data['x_coord'][next_customer], data['y_coord'][next_customer]]
                next_service_time = data['service_time'][next_customer]
                next_ready_time = data['ready_time'][next_customer]
                next_due_date = data['due_date'][next_customer]

                dist2customer = self.distance(current_pos, next_pos)
                dist2deposit = self.distance(next_pos, depot_pos)

                # Comprobación de que es alcanzable (demanda y tiempo)
                next_time = max(current_time + dist2customer * 60 / velocity, next_ready_time)
                if (next_time <= next_due_date) and \
                        (next_time + next_service_time + dist2deposit * 60 / velocity <= max_time):
                    total_customers += 1

                    # Actualización de datos
                    current_dist += dist2customer

                    current_time = next_time + next_service_time
                    current_dist2deposit = dist2deposit

                    current_pos = next_pos
                    current_route.append(next_customer)

                # Si no es alcanzable se utiliza otro vehículo
                else:
                    break

            # Si ha recorrido distancia -> se almacena información
            if current_route != [depot]:
                total_vehicles += 1

                # Vuelta al depósito
                current_time += current_dist2deposit * 60 / velocity
                current_dist += current_dist2deposit

                # Se almacena la información del anterior vehículo
                total_dist += current_dist

                veh = {'id': total_vehicles, 'time': current_time, 'distance': current_dist}
                vehicles.append(veh)
                routes.append(current_route)

            # Si el actual vehículo no ha recorrido distancia -> fin
            else:
                break

        # Comprobación de solución válida
        if (dim_sol > 0) and (total_customers == dim_sol):
            solution_value = total_vehicles * cost_vehicles + total_dist * cost_km
        elif dim_sol == 0:
            solution_value = float('inf')
        else:  # total_customers != dim_sol
            routes = []
            solution_value = float('inf')

        self.logger.debug(
            '[VRPTW] Valor: ' + str(solution_value) + '; Solución: ' + str(solution) + '; Rutas: ' + str(routes) +
            '; Vehículos: ' + str(vehicles))

        return solution_value, solution, routes, vehicles
