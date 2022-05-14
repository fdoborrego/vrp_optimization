# Timer para cortar ejecución

import time


class Timer(object):
    """
    <class 'Timer'>
    Timer para comprobar tiempos de ejecución de los algoritmos de ejecución.
    """

    def __init__(self, max_time=1):

        # Configuración del timer
        self.max_time = max_time                        # [min]

        # Inicialización del timer
        self.tic = 0
        self.toc = 0

        self.dt = 0

    def set(self, max_time):
        """ Tiempo máximo del timer """
        self.max_time = max_time

    def start(self):
        """ Lectura de instante inicial """
        self.tic = time.process_time()

    def reset(self):
        """ Reset del timer """
        self.tic = 0
        self.toc = 0

        self.dt = 0

    def update(self):
        """ Lectura de tiempo transcurrido """
        self.toc = time.process_time()
        return self.check()

    def check(self):
        """ Comprobación de toc - tic = max_time """
        return True if self.toc - self.tic >= self.max_time * 60 else False

    def stop(self):
        """ Fin de temporización """
        self.dt = self.toc - self.tic
        return self.dt
