"""
Contiene todos los metodos relacionados con la mobilidad
"""

import numpy as np


def update_positions(population):
    """Actualiza la pocision de todas las personas

    Use la velocidad y la direccion para actualizar la pocision para el siguiente instante de tiempo

    Keyword arguments
    -----------------
    population : ndarray
        Contiene toda la informacion de la población
    """

    # Actualiza pocisiones en
    # x
    population[:, 1] = population[:, 1] + (population[:, 3] * population[:, 5])
    # y
    population[:, 2] = population[:, 2] + (population[:, 4] * population[:, 5])

    return population


def out_of_bounds(population, xbounds, ybounds):
    """comprueba qué personas están a punto de salirse de los límites y corrige

    Función que actualiza las cabeceras de los individuos que están a punto de salir de los límites del mundo.

     Keyword arguments
     -----------------
     population : ndarray
         Contiene toda la informacion de la población

     xbounds, ybounds : list or tuple
         Contiene los limites superior e inferior [min, max]
    """
    # Actualiza la dirección cuando se encuentra en los limites
    # Actualiza la dirección en X
    # Determina la cantidad de elementos que necesitan ser actualizados

    shp = population[:, 3][
        (population[:, 1] <= xbounds[:, 0]) & (population[:, 3] < 0)
    ].shape
    population[:, 3][
        (population[:, 1] <= xbounds[:, 0]) & (population[:, 3] < 0)
    ] = np.clip(np.random.normal(loc=0.5, scale=0.5 / 3, size=shp), a_min=0.05, a_max=1)

    shp = population[:, 3][
        (population[:, 1] >= xbounds[:, 1]) & (population[:, 3] > 0)
    ].shape
    population[:, 3][
        (population[:, 1] >= xbounds[:, 1]) & (population[:, 3] > 0)
    ] = np.clip(
        -np.random.normal(loc=0.5, scale=0.5 / 3, size=shp), a_min=-1, a_max=-0.05
    )

    # Actualiza la dirección en Y
    shp = population[:, 4][
        (population[:, 2] <= ybounds[:, 0]) & (population[:, 4] < 0)
    ].shape
    population[:, 4][
        (population[:, 2] <= ybounds[:, 0]) & (population[:, 4] < 0)
    ] = np.clip(np.random.normal(loc=0.5, scale=0.5 / 3, size=shp), a_min=0.05, a_max=1)

    shp = population[:, 4][
        (population[:, 2] >= ybounds[:, 1]) & (population[:, 4] > 0)
    ].shape
    population[:, 4][
        (population[:, 2] >= ybounds[:, 1]) & (population[:, 4] > 0)
    ] = np.clip(
        -np.random.normal(loc=0.5, scale=0.5 / 3, size=shp), a_min=-1, a_max=-0.05
    )

    return population


def update_randoms(
    population,
    pop_size,
    speed=0.01,
    heading_update_chance=0.02,
    speed_update_chance=0.02,
    heading_multiplication=1,
    speed_multiplication=1,
):
    """actualiza estados aleatorios como el rumbo y la velocidad

    Función que aleatoriza los encabezamientos y las velocidades de los miembros de la población con probabilidades ajustables.

    Keyword arguments
    -----------------
    population : ndarray
        El arreglo que contiene toda la información de la población

    pop_size : int
        El tamaño de la poblacición

    heading_update_chance : float
        las probabilidades de actualizar la rúbrica de cada miembro, cada paso de tiempo

    speed_update_chance : float
        las probabilidades de actualizar la velocidad de cada miembro, cada paso de tiempo

    heading_multiplication : int or float
        factor por el que multiplicar la dirección (las direeciones por defecto están entre -1 y 1)

    speed_multiplication : int or float
        factor por el que multiplicar la velocidad (las velocidades por defecto están entre 0.0001 y 0.05)

    speed : int or float
        velocidad media de los miembros de la población, las velocidades se tomarán de una distribución gaussiana con media 'velocidad' y sd 'velocidad / 3'
    """

    # Actualiza la dirección aleatoriamente
    # x
    update = np.random.random(size=(pop_size,))
    shp = update[update <= heading_update_chance].shape
    population[:, 3][update <= heading_update_chance] = (
        np.random.normal(loc=0, scale=1 / 3, size=shp) * heading_multiplication
    )
    # y
    update = np.random.random(size=(pop_size,))
    shp = update[update <= heading_update_chance].shape
    population[:, 4][update <= heading_update_chance] = (
        np.random.normal(loc=0, scale=1 / 3, size=shp) * heading_multiplication
    )
    # Aleatoriza la velocidad
    update = np.random.random(size=(pop_size,))
    shp = update[update <= heading_update_chance].shape
    population[:, 5][update <= heading_update_chance] = (
        np.random.normal(loc=speed, scale=speed / 3, size=shp) * speed_multiplication
    )

    population[:, 5] = np.clip(population[:, 5], a_min=0.0001, a_max=0.05)
    return population


def get_motion_parameters(xmin, ymin, xmax, ymax):
    """obtiene centro de destino y rangos de recorrido

    Función que devuelve los parámetros geométricos del destino que han establecido los miembros de la población.

    Keyword arguments:
    ------------------
        xmin, ymin, xmax, ymax : int or float
        Limites superiores e inferiores del area

    """

    x_center = xmin + ((xmax - xmin) / 2)
    y_center = ymin + ((ymax - ymin) / 2)

    x_wander = (xmax - xmin) / 2
    y_wander = (ymax - ymin) / 2

    return x_center, y_center, x_wander, y_wander
