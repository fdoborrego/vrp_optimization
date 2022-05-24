# Clase de construcción de figuras

import os
from itertools import cycle
import matplotlib.pyplot as plt


class GraphicsBuilder(object):
    """
    <class 'GraphicsBuilder'>
    Clase que permite construir los gráficos de una solución al problema del viajero, ya sea en su modalidad CVRP
    (Capacity Vehicle Routing Problem) o VRPTW (Vehicle Routing Problem with Time Windows).
    """

    def __init__(self, problem_type, data, path='../figures'):

        # Definición del problema
        self.problem_type = problem_type
        self.data = data

        # Construcción de ruta
        self.path = path
        if not os.path.exists(path):
            os.mkdir(path)

    def __repr__(self):
        return "<GB-%s:" % self.problem_type

    def __str__(self):
        return "Graphics Builder: %s problem. " % self.problem_type

    def graph_veh_distance(self, vehicles, method):
        """
        Gráfica que representa la distancia de trayecto de cada vehículo.
        :param vehicles: Estructura de datos que contiene la información de los vehículos.
        :param method: Método usado en el proceso de optimización.
        """

        # Captura de datos
        data = {}
        for veh in vehicles:
            name = 'Veh. ' + str(veh['id'])
            distance = veh['distance']

            data[name] = distance

        # Representación
        fig, ax = plt.subplots()

        for i, key in enumerate(data):
            p = plt.bar(i, data[key])
            ax.bar_label(p, padding=3)

        ax.set_title('[' + self.problem_type + '] Distancia recorrida por cada vehículo - Total: ' + str(
            round(sum(data.values()), 2)) + ' km')
        ax.set_xlabel('Vehículo')
        ax.set_ylabel('Distancia [km]')

        # ax.set_yticks(list(data.values()))
        ax.set_xticks(range(len(data)), data.keys())

        plt.show()

        # Guardado
        fig.savefig(self.path + '/' + self.problem_type + '_' + method + '_VEH_distancia.png')

        # Cierre
        plt.close()

    def graph_veh_time(self, vehicles, method):
        """
        Gráfica que representa el tiempo de trayecto de cada vehículo.
        :param vehicles: Estructura de datos que contiene la información de los vehículos.
        :param method: Método usado en el proceso de optimización.
        """

        # Captura de datos
        data = {}
        for veh in vehicles:
            name = 'Veh. ' + str(veh['id'])
            time = veh['time']

            data[name] = time

        # Representación
        fig, ax = plt.subplots()

        for i, key in enumerate(data):
            p = plt.bar(i, data[key])
            ax.bar_label(p, padding=3)

        ax.set_title('[' + self.problem_type + '] Tiempo empleado por cada vehículo - Total: ' + str(
            round(sum(data.values()), 2)) + ' min')
        ax.set_xlabel('Vehículo')
        ax.set_ylabel('Tiempo [min]')

        # ax.set_yticks(list(data.values()))
        ax.set_xticks(range(len(data)), data.keys())

        plt.show()

        # Guardado
        fig.savefig(self.path + '/' + self.problem_type + '_' + method + '_VEH_tiempo.png')

        # Cierre
        plt.close()

    def graph_veh_capacity(self, vehicles, method):
        """
        Gráfica que representa la capacidad de cada vehículo de la solución.
        :param vehicles: Estructura de datos que contiene la información de los vehículos.
        :param method: Método usado en el proceso de optimización.
        """

        # Captura de datos
        data = {}
        for veh in vehicles:
            name = 'Veh. ' + str(veh['id'])
            capacity = veh['capacity']

            data[name] = capacity

        # Representación
        fig, ax = plt.subplots()

        for i, key in enumerate(data):
            p = ax.bar(i, data[key])
            ax.bar_label(p, padding=3)

        ax.set_title(
            '[' + self.problem_type + '] Demanda cubierta por vehículo - Total: ' + str(sum(data.values())) + ' kg')
        ax.set_xlabel('Vehículo')
        ax.set_ylabel('Demanda cubierta [kg]')

        # ax.set_yticks(list(data.values()))
        ax.set_xticks(range(len(data)), data.keys())

        plt.show()

        # Guardado
        fig.savefig(self.path + '/' + self.problem_type + '_' + method + '_VEH_capacidad.png')

        # Cierre
        plt.close()

    def graph_routes(self, routes, depots, method):
        """
        Salida gráfica de la representación de las rutas que conforman una solución.
        :param routes: Solución construida (contiene lista con los índices de las paradas según vehículos).
        :param depots: Índice de cliente de los depósitos.
        :param method: Método usado en el proceso de optimización.
        """
        # Parámetros de entrada
        data = self.data
        depositos = depots[0]

        # Configuración de gráficos
        fig, ax = plt.subplots()
        linestyle_tuple = cycle([
            'solid',
            'dashed',
            'dashdot',
            'dotted',
            (0, (3, 1, 1, 1)),
            (0, (1, 10)),
            (0, (1, 1)),
            (0, (1, 1)),
            (0, (5, 10)),
            (0, (5, 5)),
            (0, (5, 1)),
            (0, (3, 10, 1, 10)),
            (0, (3, 5, 1, 5)),
            (0, (3, 5, 1, 5, 1, 5)),
            (0, (3, 10, 1, 10, 1, 10)),
            (0, (3, 1, 1, 1, 1, 1))])
        x_max, y_max = 0, 0

        # Representación del depósito
        ax.scatter(data['x_coord'][depositos], data['y_coord'][depositos], color='k', marker='D', linewidths=5,
                   label='Depósito')
        ax.text(data['x_coord'][depositos], data['y_coord'][depositos],
                " Depósito: (" + str(data['x_coord'][depositos]) + ', ' + str(data['y_coord'][depositos]) + ')')

        # Representación de rutas de vehículos
        for i, vehicle_route in enumerate(routes):

            # Captura de datos
            x_coord, y_coord = [], []
            for customer in vehicle_route:
                x_coord.append(data['x_coord'][customer])
                y_coord.append(data['y_coord'][customer])

                # Para representación de ejes
                if x_coord[-1] > x_max:
                    x_max = x_coord[-1]
                if y_coord[-1] > y_max:
                    y_max = y_coord[-1]

            # Representación de la ruta seguida
            line, = ax.plot([*x_coord, data['x_coord'][depositos]], [*y_coord, data['y_coord'][depositos]],
                            linestyle=next(linestyle_tuple))

            # Representación de las coordenadas
            label = 'Vehículo ' + str(i + 1)
            ax.scatter(x_coord[1:], y_coord[1:], marker='o', color=line.get_color(), label=label)

            # Texto
            for customer in vehicle_route:
                if customer != depositos:
                    x, y = data['x_coord'][customer], data['y_coord'][customer]
                    ax.text(x, y, ' ' + str(customer) + ': (' + str(x) + ', ' + str(y) + ')', color=line.get_color())

        # Título
        ax.set_title('[' + self.problem_type + '] Rutas descritas en la solución')
        ax.set_ylabel('km', horizontalalignment='right', y=1.0)
        ax.set_xlabel('km', horizontalalignment='right', x=1.0)
        ax.set_xlim([0, x_max + 10])
        ax.set_ylim([0, y_max + 10])
        ax.legend(loc='upper left')

        plt.show()

        # Guardado
        fig.savefig(self.path + '/' + self.problem_type + '_' + method + '_SOL_rutas.png')

        # Cierre
        plt.close()

    def graph_history(self, solution_value_history, method):
        """
        Salida gráfica de la evolución de una metaheurística (iteraciones).

        :param solution_value_history: Estructura de datos que almacena los valores de las soluciones alcanzadas.
        :param method: Método usado en el proceso de optimización.
        """

        # Captura de datos
        best_solution_value = solution_value_history[0]
        best_solution_value_history = [best_solution_value] * len(solution_value_history)
        for i in range(len(solution_value_history)):
            if solution_value_history[i] < best_solution_value:
                best_solution_value = solution_value_history[i]
            best_solution_value_history[i] = best_solution_value

        # Representación
        fig, ax = plt.subplots()
        ax.plot(solution_value_history, linewidth=2, marker='o', markersize=3, label='Solución')
        ax.plot(best_solution_value_history, color='r', label='Mejor solución')
        ax.plot(best_solution_value_history.index(min(best_solution_value_history)), min(best_solution_value_history),
                linestyle=' ', marker='x', markersize=5, markeredgewidth=2, color='r')

        ax.set_title('[' + self.problem_type + '] ' + method + ' - Evolución de la solución (por Iteraciones)')
        ax.set_ylabel('Valor de la función objetivo [€]')
        ax.set_xlabel('Nº de Iteraciones')
        ax.set_xlim([0, len(solution_value_history) + 1])
        # ax.set_ylim([0, max(sol_evol)+10])
        ax.legend(loc='upper right')

        plt.show()

        # Guardado
        fig.savefig(self.path + '/' + self.problem_type + '_' + method + '_EVOLxITER.png')

        # Cierre
        plt.close()
