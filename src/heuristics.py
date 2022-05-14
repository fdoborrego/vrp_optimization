# Algoritmos de optimización

import math
import random
from itertools import permutations


def distance(point1, point2):
    """
    Devuelve la distancia euclídea de dos puntos en el plano cartesiano.
    :param point1: Coordenadas [x, y] del punto 1 (origen).
    :param point2: Coordenadas [x, y] del punto 2 (destino).

    :return: Distancia euclídea entre point1 y point2.
    """
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def bruteforce(optimizer, solution, start=None):
    """
    Método de fuerza bruta.
    Este método realiza una exploración de todas las posibles soluciones del problema para buscar la óptima entre
    todas ellas.
    """

    # Datos de entrada
    data = optimizer.data
    eval_function = optimizer.evalfun
    timer = optimizer.timer

    # Inicialización
    best_solution_value, best_solution, _, _ = eval_function(solution, data)
    solution_values = []

    if start is not None:
        solution = start + solution
    fixed = solution[0]

    # Exploración de todas las posibles soluciones
    for perm in permutations(solution):

        # Comprobación de tiempo máximo
        if timer.update():
            return best_solution, best_solution_value, solution_values

        # Solo se consideran válidas las que comienzan por "start" (en este caso el depósito)
        if perm[0] == fixed:
            new_solution_value, new_solution, _, _ = eval_function(list(perm[1:]), data)

            # Mejor solución
            if new_solution_value < best_solution_value:
                best_solution = new_solution
                best_solution_value = new_solution_value

            # Histórico de soluciones
            solution_values.append(new_solution_value)

    return best_solution, best_solution_value, solution_values


def nearest_neighbor(optimizer, solution, start=None):
    """
    Método del vecino más cercano.
    Este método realiza una construcción de una solución subóptima. Para ello, une consecutivamente aquellos puntos más
    cercanos entre sí.
    """

    # Datos de entrada
    data = optimizer.data
    eval_function = optimizer.evalfun
    timer = optimizer.timer

    # Inicialización
    best_solution_value, best_solution, _, _ = eval_function(solution, data)
    solution_values = [best_solution_value]

    if start is not None:
        solution = start + solution
    fixed = solution[0]

    # Construcción de la solución
    must_visit = solution[:]
    path = [fixed]
    must_visit.remove(fixed)
    while must_visit:
        current_pos = [data['x_coord'][path[-1]], data['y_coord'][path[-1]]]
        nearest = min(must_visit, key=lambda x: distance(current_pos, [data['x_coord'][x], data['y_coord'][x]]))
        path.append(nearest)
        must_visit.remove(nearest)

    # Resultados
    best_solution_value, best_solution, _, _ = eval_function(path[1:], data)
    solution_values.append(best_solution_value)
    timer.update()

    return best_solution, best_solution_value, solution_values


def cost_saving(optimizer, solution, start=None):
    """
    Método del ahorro de costes.
    Este método realiza una construcción de una solución subóptima. Para ello, se incorporan a la solución aquellos
    arcos de menor coste y que sean compatibles con las restricciones (= no formen ciclos).
    """

    # Datos de entrada
    data = optimizer.data
    eval_function = optimizer.evalfun
    timer = optimizer.timer

    # Inicialización
    best_solution_value, _, best_solution, _ = eval_function(solution, data)
    solution_values = [best_solution_value]

    if start is not None:
        solution = start + solution
    fixed = solution[0]

    # Coste de todos los posibles arcos
    def coords(x):
        return [data['x_coord'][solution[x]], data['y_coord'][solution[x]]]

    cost_list = [(i, j, distance(coords(i), coords(j)))
                 for i in range(len(solution)) for j in range(i + 1, len(solution))]
    cost_list.sort(key=lambda cost: cost[2])

    # Inicialización de variables
    inode_list = list(range(len(solution)))
    jnode_list = list(range(len(solution)))
    must_visit = cost_list[:]

    # Cálculo de la solución
    arcs_path = [[must_visit[0][0], must_visit[0][1]]]
    inode_list.remove(must_visit[0][0])
    jnode_list.remove(must_visit[0][1])
    must_visit.remove(must_visit[0])

    while must_visit:
        node1, node2 = must_visit[0][0], must_visit[0][1]

        # Se considera arco (i,j) y arco (j,i)
        for arc in ([node1, node2], [node2, node1]):

            # Si no forma ciclo, y aún no existe un arco que salga desde i y llegue a j -> forma parte de solución
            if not _check_cycle(arcs_path, arc):
                if (arc[0] in inode_list) and (arc[1] in jnode_list):
                    arcs_path.append(arc)
                    inode_list.remove(arc[0])
                    jnode_list.remove(arc[1])

        must_visit.remove(must_visit[0])

    # Conversión arcs -> points
    nodes_path = _arcs2nodes(arcs_path, start=fixed)

    # Construcción de solución
    path = []
    for node in nodes_path:
        path.append(solution[node])

    # Resultados
    best_solution_value, best_solution, _, _ = eval_function(path[1:], data)
    solution_values.append(best_solution_value)
    timer.update()

    return best_solution, best_solution_value, solution_values


def _check_cycle(arcs_path, new_arc):
    """ Comprueba si se forman ciclos al incorporar new_arc al arc_path """

    # Número de arcos en ciclo
    n_arcs = 0

    # Análisis de ciclo
    start_node = new_arc[1]             # Comienzo del ciclo: nodo 'j' del nuevo arco ([_, start_node])
    while n_arcs < len(arcs_path):      # Si nº de arcos recorridos sin encontrar ciclo = arcos totales -> No hay ciclo

        # Búsqueda de siguiente arco
        next_node = None
        for arc in arcs_path:

            # Nuevo arco: [start_node, next_node]
            if start_node == arc[0]:
                next_node = arc[1]
                n_arcs += 1
                break

        # Si no se ha encontrado arco -> no es ciclo
        if next_node is None:
            return False

        # Si se ha encontrado arco...
        else:

            # Si se cierra el ciclo → Ciclo encontrado
            if next_node == new_arc[0]:
                return True

            # Si no -> Se continúa la cadena
            else:
                start_node = next_node

    # Si se ha recorrido toda la cadena y no hay ciclo
    return False


def _arcs2nodes(arcs, start=None):
    """ Convierte una trayectoria modelada en forma de arcos a nodos """

    # Datos de entrada
    disordered_route = arcs[:]

    if start is None:
        start = 0

    # Se ordena la ruta por arcos consecutivos
    ordered_route = [disordered_route[0]]
    disordered_route.remove(disordered_route[0])

    while disordered_route:
        for arc in disordered_route:

            # Se añade arco al principio
            if arc[1] == ordered_route[0][0]:
                ordered_route.insert(0, arc)
                disordered_route.remove(arc)

            # Se añade arco al final
            elif arc[0] == ordered_route[-1][1]:
                ordered_route.append(arc)
                disordered_route.remove(arc)

    # Se crea ruta en nodos
    node = ordered_route[0][0]
    nodes = [node]

    while ordered_route:
        for arc in ordered_route:
            if node == arc[0]:
                node = arc[1]
                nodes.append(node)
                ordered_route.remove(arc)

    # Se ordena la ruta, comenzando en el nodo deseado
    start_idx = nodes.index(start)
    nodes.extend(nodes[:start_idx])  # "Cola circular"; se añaden elementos antes del "start" al final
    nodes = nodes[start_idx:]  # Corto hasta start

    return nodes


def local_search(optimizer, solution, operator='swap'):
    """
    Método de la búsqueda local (o método del gradiente).
    Este método realiza una exploración de aquel espacio de soluciones alcanzable mediante un operador de vecindad. Una
    vez encontrado un óptimo local, se detiene la exploración y se devuelve la solución encontrada.
    """

    # Datos de entrada
    data = optimizer.data
    eval_function = optimizer.evalfun
    timer = optimizer.timer

    # Inicialización
    current_solution_value, current_solution, _, _ = eval_function(solution, data)
    neighbourhood = []

    best_solution_value = current_solution_value
    best_solution = current_solution
    solutions = [current_solution_value]

    # Búsqueda del óptimo local
    while True:

        # Comprobación de tiempo máximo
        if timer.update():
            return best_solution, best_solution_value, solutions

        # Cálculo de nueva vecindad
        if operator == 'swap':
            neighbourhood = _neighbourhood_swap(current_solution)
        elif operator == 'insertion':
            neighbourhood = _neighbourhood_insertion(current_solution)

        # Cálculo de vecino más favorable
        new_solution = min(neighbourhood, key=lambda x: eval_function(x, data)[0])
        new_solution_value, new_solution, _, _ = eval_function(new_solution, data)

        # Comprobación de mejor solución
        if new_solution_value < best_solution_value:
            best_solution_value = new_solution_value
            best_solution = new_solution
            neighbourhood = []
            solutions.append(best_solution_value)

        else:
            break

        # Actualización del bucle
        current_solution = new_solution

    return best_solution, best_solution_value, solutions


def _neighbourhood_swap(solution):
    """ Operador de vecindad SWAP (intercambia dos elementos de la solución) """

    # Inicialización
    neighbourhood = []

    # Construcción de vecinos
    for i in range(len(solution) - 1):
        for j in range(i + 1, len(solution)):
            neighbour = solution[:]
            neighbour[i], neighbour[j] = neighbour[j], neighbour[i]

            neighbourhood.append(neighbour)

    return neighbourhood


def _neighbourhood_swap_p(solution, p=0.2):
    """ Operador de vecindad SWAP (intercambia un porcentaje 'p' de los elementos de la solución) """

    # Inicialización
    neighbourhood = []

    # Construcción de vecinos
    N = math.floor(p * len(solution))
    for i in range(len(solution) - 1 + N):
        for j in range(i + 1 + N, len(solution)):
            neighbour = solution[:]
            neighbour[i:i + N], neighbour[j:j + N] = neighbour[j:j + N], neighbour[i:i + N]

            neighbourhood.append(neighbour)

    return neighbourhood


def _neighbourhood_insertion(solution):
    """ Operador de vecindad INSERTION (introduce el elemento 'i' en la posición 'j') """

    # Inicialización
    neighbourhood = []

    # Construcción de vecinos
    for i in range(len(solution)):
        for j in range(len(solution)):
            if (i != j) and (i != j + 1):
                neighbour = solution[:]
                neighbour[j:j] = [neighbour.pop(i)]

                neighbourhood.append(neighbour)

    return neighbourhood


def vnd(optimizer, solution):
    """
    Método de búsqueda de entorno variable (VNS) descendiente.
    Este método consiste en una metaheurística que extiende el método de búsqueda local utilizando varias vecindades. De
    esta forma, cuando se alcanza un óptimo local dado un operador de vecindad, se utiliza otro operador de forma que
    pueda hallarse un nuevo óptimo.
    """

    # Datos de entrada
    data = optimizer.data
    eval_function = optimizer.evalfun
    timer = optimizer.timer

    # Inicialización
    current_solution_value, current_solution, current_route, _ = eval_function(solution, data)

    best_solution = current_solution
    best_solution_value = current_solution_value
    solutions = [current_solution_value]

    # Operadores
    operators = ['swap', 'insertion']
    k = 1

    # Algoritmo VND
    while k <= len(operators):

        # Comprobación de tiempo máximo
        if timer.update():
            return best_solution, best_solution_value, solutions

        # Búsqueda local
        new_solution, new_solution_value, solutions_ls = \
            local_search(optimizer, current_solution, operator=operators[k - 1])

        solutions.extend(solutions_ls)

        # Actualización de mejor solución
        if new_solution_value < current_solution_value:
            current_solution_value = new_solution_value
            current_solution = new_solution

            best_solution = new_solution
            best_solution_value = current_solution_value

            k = 1  # Vuelta a operador inicial

        else:
            k += 1  # Se prueba nuevo operador

    return best_solution, best_solution_value, solutions


def vns(optimizer, solution):
    """
    Método de búsqueda de entorno variable (VNS).
    Este método consiste en una metaheurística que extiende el método de búsqueda local utilizando varias vecindades. De
    esta forma, cuando se alcanza un óptimo local dado un operador de vecindad, se utiliza otro operador de forma que
    pueda hallarse un nuevo óptimo.
    Con respecto al algoritmo VND, el VNS introduce una función de shaking que permite al algoritmo realizar un cambio
    brusco en la solución al atascarse en un óptimo local. Esto le permitirá una mayor exploración al mismo.
    """

    # Datos de entrada
    data = optimizer.data
    eval_function = optimizer.evalfun
    timer = optimizer.timer

    # Inicialización
    current_solution_value, current_solution, _, _ = eval_function(solution, data)

    best_solution = current_solution
    best_solution_value = current_solution_value
    solutions = [current_solution_value]

    # Operadores
    operators = ['swap', 'insertion']
    k = 1

    # Algoritmo VND
    while k <= len(operators):

        # Comprobación de tiempo máximo
        if timer.update():
            return best_solution, best_solution_value, solutions

        # Shaking
        current_solution = _shaking(current_solution)

        # Búsqueda local
        new_solution, new_solution_value, solutions_ls = \
            local_search(optimizer, current_solution, operator=operators[k - 1])

        solutions.extend(solutions_ls)

        # Actualización de mejor solución
        if new_solution_value < current_solution_value:
            current_solution_value = new_solution_value
            current_solution = new_solution

            best_solution = new_solution
            best_solution_value = current_solution_value

            k = 1  # Vuelta a operador inicial

        else:
            k += 1  # Se prueba nuevo operador

    return best_solution, best_solution_value, solutions


def _shaking(solution):
    """ Función de shaking: aplica un operador de vecindad más agresivo a la solución """

    # Vecindario de la solución
    current_solution = solution[:]
    neighbourhood = _neighbourhood_swap_p(current_solution)

    # Vecino aleatorio
    new_solution = random.sample(neighbourhood, k=1)[0]

    return new_solution


def simulated_annealing(optimizer, solution):
    """
    Método de recocido simulado.
    Este método consiste en una metaheurística que permite al algoritmo alejarse de la solución óptima encontrada
    siguiendo el comportamiento del recocido de los metales. De esta forma, se le dota de una mayor exploración al
    algoritmo.
    El algoritmo podrá alejarse de la solución óptima encontrada en función de una temperatura que irá decreciendo con
    el tiempo. EL funcionamiento del mismo será tal que, si la solución encontrada es peor a la óptima histórica, el
    algoritmo tendrá cierta probabilidad de saltar hacia ella (en función de la temperatura antes mencionada).
    """

    # Datos de entrada
    data = optimizer.data
    eval_function = optimizer.evalfun
    timer = optimizer.timer

    # Parámetros del algoritmo
    max_temp = 100
    min_temp = 1
    L = 10
    alpha = 0.001

    # Inicialización
    temp = max_temp
    current_solution_value, current_solution, _, _ = eval_function(solution, data)

    best_solution = current_solution
    best_solution_value = current_solution_value
    solutions = [current_solution_value]

    # Búsqueda de la solución óptima
    while temp > min_temp:
        for _ in range(L):

            # Creación de vecindad
            neighbourhood = _neighbourhood_swap(current_solution)

            # Orden de evaluación de la vecindad
            order = list(range(len(neighbourhood)))
            random.shuffle(order)

            # Evaluación de la vecindad
            for i in range(len(neighbourhood)):

                # Comprobación de tiempo máximo
                if timer.update():
                    return best_solution, best_solution_value, solutions

                new_solution_value, new_solution, _, _ = eval_function(neighbourhood[order[i]], data)

                delta = new_solution_value - current_solution_value

                # Si mejora la solución → se acepta
                if delta < 0:

                    # Mejor solución histórica
                    if current_solution_value < best_solution_value:
                        best_solution_value = new_solution_value
                        best_solution = neighbourhood[order[i]][:]

                    # Actualización
                    current_solution_value = new_solution_value
                    current_solution = neighbourhood[order[i]][:]
                    break

                # Si no mejora la solución → probabilidad de aceptación
                else:
                    a = random.random()
                    if a < math.exp(-delta / temp):
                        current_solution_value = new_solution_value
                        current_solution = neighbourhood[order[i]][:]
                        break

            solutions.append(current_solution_value)
        temp = temp / (1 + alpha * temp)

    return best_solution, best_solution_value, solutions


def tabu_search(optimizer, solution):
    """
    Método de búsqueda tabú.
    Este método consiste en una metaheurística que, partiendo de una solución inicial, construye un entorno de
    soluciones adyacentes que pueden ser alcanzadas. El algoritmo explora el espacio de búsqueda atendiendo a
    restricciones basadas en memorias de lo reciente y lo frecuente y a niveles de aspiración (excepciones a estas
    restricciones).
    """

    # Datos de entrada
    data = optimizer.data
    eval_function = optimizer.evalfun
    timer = optimizer.timer

    # Parámetros del algoritmo
    tabu_size = len(solution) / 2
    n_iteration = 1000

    # Solución inicial
    current_solution_value, current_solution, _, _ = eval_function(solution, data)

    best_solution = current_solution[:]
    best_solution_value = current_solution_value
    solutions = [current_solution_value]

    # Listas tabús
    recent_tabu_list = []
    frequent_tabu_list = [0 for i in range(len(current_solution) - 1) for _ in range(i + 1, len(current_solution))]

    # Búsqueda de la mejor solución
    for _ in range(n_iteration):

        # Comprobación de tiempo máximo
        if timer.update():
            return best_solution, best_solution_value, solutions

        # Creación de vecindad: swap
        swap = []                               # Swaps realizados
        neighbourhood = []                      # Vecindad
        neighbourhood_value = []                # Valor de la función objetivo de la vecindad
        neighbourhood_value_penalty = []        # Valor tras penalización de la vecindad

        for i in range(len(current_solution) - 1):
            for j in range(i + 1, len(current_solution)):

                # Atributos almacenados (elementos que participan en el swap)
                ti = min(current_solution[i], current_solution[j])
                tj = max(current_solution[i], current_solution[j])

                # Nuevo vecino
                neighbour = current_solution[:]
                neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
                neighbour_val, _, _, _ = eval_function(neighbour, data)
                neighbour_val_penalty = neighbour_val + frequent_tabu_list[int(ti + ((tj - 2) * (tj-1)) / 2) - 1]

                # Comprobación: no tabú o mejor histórico
                if ([ti, tj] not in recent_tabu_list) or (
                        [ti, tj] in recent_tabu_list and neighbour_val < best_solution_value):
                    neighbourhood.append(neighbour)
                    neighbourhood_value.append(neighbour_val)
                    neighbourhood_value_penalty.append(neighbour_val_penalty)
                    swap.append([current_solution[i], current_solution[j]])

        _, current_solution_value, swap, current_solution = \
            min(zip(neighbourhood_value_penalty, neighbourhood_value, swap, neighbourhood))

        solutions.append(current_solution_value)

        # Mejor solución
        if current_solution_value < best_solution_value:
            best_solution = current_solution[:]
            best_solution_value = current_solution_value

        # Actualización de listas tabú
        ti, tj = min(swap), max(swap)
        recent_tabu_list.append([min(swap), max(swap)])
        if len(recent_tabu_list) > tabu_size:
            recent_tabu_list.pop(0)

        frequent_tabu_list[int(ti + ((tj - 2) * (tj-1)) / 2) - 1] += 1

    return best_solution, best_solution_value, solutions
