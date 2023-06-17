"""
Aqui se encuentran los parametros de la población
"""

from glob import glob
import os

import numpy as np

from motion import get_motion_parameters
from utils import check_folder


def initialize_population(
    Config, mean_age=45, max_age=105, xbounds=[0, 1], ybounds=[0, 1]
):
    """Inicializa la poblacion para la simulación

    La matriz de poblacion para esta simulación contiene las siguientes columnas:

    0 : ID Unico
    1 : Coordenada X actual
    2 : Coordenada Y actual
    3 : Dirección en X actual
    4 : Dirección en Y actual
    5 : Velocidad actual
    6 : Estado actual (0=Sano, 1=Enfermo, 2=Inmune, 3=Muerto, 4=Inmune pero infectado)
    7 : Edad
    8 : Infectado desde (Instante de tiempo en el que la persona fue infectada)
    9 : Vector de recuperación
    10 : En tratamiento
    11 : Destino activo (0 = destino aleatorio, 1, .. = Indice matriz de destino)
    12 : Ya esta en el destino? (0=Viajando , 1=En el destino)
    13 : wander_range_x : Rango en X donde se encuentra confinado
    14 : wander_range_y : Rango en Y donde se encuentra confinado

    Keyword arguments
    -----------------
    pop_size : int
        Tamaño de la población

    mean_age : int
        Edad minima de la población, la edad afecta la mortalidad

    max_age : int
        Edad maxima de la población

    xbounds : 2d array
        Limites del eje X

    ybounds : 2d array
        Limites del eje Y
    """

    # Inicializa la matriz de población
    population = np.zeros((Config.pop_size, 15))

    # Inicializa los Ids unicos
    population[:, 0] = [x for x in range(Config.pop_size)]

    # Inicializa coordenadas aleatorias
    population[:, 1] = np.random.uniform(
        low=xbounds[0] + 0.05, high=xbounds[1] - 0.05, size=(Config.pop_size,)
    )
    population[:, 2] = np.random.uniform(
        low=ybounds[0] + 0.05, high=ybounds[1] - 0.05, size=(Config.pop_size,)
    )

    # Inicializa la direccion de -1 a 1
    population[:, 3] = np.random.normal(loc=0, scale=1 / 3, size=(Config.pop_size,))
    population[:, 4] = np.random.normal(loc=0, scale=1 / 3, size=(Config.pop_size,))

    # Inicializa la velocidad de la población
    population[:, 5] = np.random.normal(Config.speed, Config.speed / 3)

    # Inicializa las edades de la población
    std_age = (max_age - mean_age) / 3
    population[:, 7] = np.int32(
        np.random.normal(loc=mean_age, scale=std_age, size=(Config.pop_size,))
    )

    population[:, 7] = np.clip(
        population[:, 7], a_min=0, a_max=max_age
    )  # La edad minima no puede estar por debajo de 0

    population[:, 9] = np.random.normal(loc=0.5, scale=0.5 / 3, size=(Config.pop_size,))

    return population


def initialize_destination_matrix(pop_size, total_destinations):
    """Inicializa la matriz de destinos

    Función que inicializa la matriz de destino utilizada para definir la ubicación individual y las zonas de recorrido de los miembros de la población

    Keyword arguments
    -----------------
    pop_size : int
        El tamaño de la población

    total_destinations : int
        El numero de destinos, puede haber mas de un destino en caso de que la persona quiera ir al trabajo, a la casa, etc...
    """

    destinations = np.zeros((pop_size, total_destinations * 2))

    return destinations


def set_destination_bounds(
    population, destinations, xmin, ymin, xmax, ymax, dest_no=1, teleport=True
):
    """Todas las personas deben estar dentro de los limites

    Función que toma la población y las coordenadas teletransporta a todos allí, establece el destino activo y destino como alcanzado

    Keyword arguments
    -----------------
    population : ndarray
        El arreglo que contiene toda la informacion de la población

    destinations : ndarray
        El arreglo que contiene toda la información de destinos

    xmin, ymin, xmax, ymax : int or float
        Limites

    dest_no : int
        El destino que está activo actualmente

    teleport : bool
        Si se debe teletransportar inmediatamente
    """

    # Teletransporte
    if teleport:
        population[:, 1] = np.random.uniform(low=xmin, high=xmax, size=len(population))
        population[:, 2] = np.random.uniform(low=ymin, high=ymax, size=len(population))

    # Obtener paramrtros
    x_center, y_center, x_wander, y_wander = get_motion_parameters(
        xmin, ymin, xmax, ymax
    )

    # Configurar centros del destino
    destinations[:, (dest_no - 1) * 2] = x_center
    destinations[:, ((dest_no - 1) * 2) + 1] = y_center

    # Configurar limites de recorrido
    population[:, 13] = x_wander
    population[:, 14] = y_wander

    population[:, 11] = dest_no  # Configurar destino activo
    population[:, 12] = 1  # Configurar destino cercano

    return population, destinations


def save_data(population, pop_tracker):
    """Guarda los datos de la población

    Funcion que almacena el estado de la simulacion

    Keyword arguments
    -----------------
    population : ndarray
        El areglo que contiene toda la informacion de la población 

    infected : list or ndarray
        El arreglo que contiene los infectados a travez del tiempo

    fatalities : list or ndarray
        El arreglo que contiene la informacion de muertes a travez del tiempo
    """
    num_files = len(glob("data/*"))
    check_folder("data/%i" % num_files)
    np.save("data/%i/population.npy" % num_files, population)
    np.save("data/%i/infected.npy" % num_files, pop_tracker.infectious)
    np.save("data/%i/recovered.npy" % num_files, pop_tracker.recovered)
    np.save("data/%i/fatalities.npy" % num_files, pop_tracker.fatalities)


def save_population(population, tstep=0, folder="data_tstep"):
    """Guarda los datos obtenidos en un instante de tiempo de la población

   Función que vuelca los datos de la simulación a archivos específicos del disco. Guarda el estado final de la matriz de población, la matriz de infectados a lo largo del tiempo y la matriz de víctimas mortales a lo largo del tiempo

    Keyword arguments
    -----------------
    population : ndarray
        El areglo que contiene toda la informacion de la población 

    tstep : int
        El instante de tiempo que se esta guardando
    """
    check_folder("%s/" % (folder))
    np.save("%s/population_%i.npy" % (folder, tstep), population)


class Population_trackers:
    """ Clase para rastrar los parametros de la población

    Puede realizar un seguimiento de los parámetros de la población a lo largo del tiempo que luego puede utilizarse para calcular estadísticas o visualizar.

    """

    def __init__(self):
        self.susceptible = []
        self.infectious = []
        self.recovered = []
        self.fatalities = []

        # PLACEHOLDER - Si un recuperado se puede volver a infectar
        self.reinfect = True

    def update_counts(self, population):
        pop_size = population.shape[0]
        self.infectious.append(len(population[population[:, 6] == 1]))
        self.recovered.append(len(population[population[:, 6] == 2]))
        self.fatalities.append(len(population[population[:, 6] == 3]))

        if self.reinfect:
            self.susceptible.append(
                pop_size - (self.infectious[-1] + self.fatalities[-1])
            )
        else:
            self.susceptible.append(
                pop_size
                - (self.infectious[-1] + self.recovered[-1] + self.fatalities[-1])
            )
