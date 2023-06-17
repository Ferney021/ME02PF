"""
Contine metodos relacionados a la planeacion de rutasa hacia el destino
"""

import numpy as np

from motion import get_motion_parameters, update_randoms


def go_to_location(patient, destination, location_bounds, dest_no=1):
    """Envía un paciente a la ubicacion deseada

    Función que toma un paciente y un destino, y establece la ubicación como activa para ese paciente.

    Keyword arguments
    -----------------
    patient : 1d array
        Es una fila de la matriz de población

    destination : 1d array
        Es una fila de la matriz de destinos

    location_bounds : list or tuple
        Define los limites en los que el paciente se puede mover

    dest_no : int
        Define el numero del destino al cual se debe llegar

    """

    x_center, y_center, x_wander, y_wander = get_motion_parameters(
        location_bounds[0], location_bounds[1], location_bounds[2], location_bounds[3]
    )
    patient[13] = x_wander
    patient[14] = y_wander

    destination[(dest_no - 1) * 2] = x_center
    destination[((dest_no - 1) * 2) + 1] = y_center

    patient[11] = dest_no

    return patient, destination


def set_destination(population, destinations):
    """Configurar el destino de la pobalción

    Establece el destino de la población si el marcador de destino no es 0. Actualiza también las cabeceras y las velocidades.

    Keyword arguments
    -----------------
    population : ndarray
        El arreglo que contiene la información de la población

    destinations : ndarray
        El arreglo que contiene la información de los destinos
    """

    # Cuantos destinos estan activos
    active_dests = np.unique(population[:, 11][population[:, 11] != 0])

    # Configurar Destino
    for d in active_dests:
        dest_x = destinations[:, int((d - 1) * 2)]
        dest_y = destinations[:, int(((d - 1) * 2) + 1)]

        # Calcular nuevas direcciones
        head_x = dest_x - population[:, 1]
        head_y = dest_y - population[:, 2]

        population[:, 3][(population[:, 11] == d) & (population[:, 12] == 0)] = head_x[
            (population[:, 11] == d) & (population[:, 12] == 0)
        ]
        population[:, 4][(population[:, 11] == d) & (population[:, 12] == 0)] = head_y[
            (population[:, 11] == d) & (population[:, 12] == 0)
        ]
        population[:, 5][(population[:, 11] == d) & (population[:, 12] == 0)] = 0.02

    return population


def check_at_destination(population, destinations, wander_factor=1.5, speed=0.01):
    """comprobar quién se encuentra ya en su destino

    Toma subconjunto de población con destino activo y comprueba quién se encuentra en las coordenadas requeridas. Actualiza en destino para las personas en destino.

    Keyword arguments
    -----------------
    population : ndarray
        El arreglo que contiene la información de la población

    destinations : ndarray
        El arreglo que contiene la información de los destinos

    wander_factor : int or float
        define a qué distancia fuera del "rango de recorrido" se alcanza el destino se activa
    """

    # Cuantos destinos estan activos
    active_dests = np.unique(population[:, 11][(population[:, 11] != 0)])

    # Ver quien esta en el destino
    for d in active_dests:
        dest_x = destinations[:, int((d - 1) * 2)]
        dest_y = destinations[:, int(((d - 1) * 2) + 1)]

        # ver quién ha llegado al destino y filtrar quién ya estaba allí
        at_dest = population[
            (np.abs(population[:, 1] - dest_x) < (population[:, 13] * wander_factor))
            & (np.abs(population[:, 2] - dest_y) < (population[:, 14] * wander_factor))
            & (population[:, 12] == 0)
        ]

        if len(at_dest) > 0:
            # Marcar como llegado
            at_dest[:, 12] = 1
            # insertar cabeceras y velocidades aleatorias para los de destino
            at_dest = update_randoms(
                at_dest,
                pop_size=len(at_dest),
                speed=speed,
                heading_update_chance=1,
                speed_update_chance=1,
            )

            # Re insertar a la población
            population[
                (
                    np.abs(population[:, 1] - dest_x)
                    < (population[:, 13] * wander_factor)
                )
                & (
                    np.abs(population[:, 2] - dest_y)
                    < (population[:, 14] * wander_factor)
                )
                & (population[:, 12] == 0)
            ] = at_dest

    return population


def keep_at_destination(population, destinations, wander_factor=1):
    """mantiene a los que han llegado, al alcance de los vagabundos

    Función que mantiene a los que han sido marcados como llegados a su destino dentro de sus respectivos rangos de vagabundeo

    Keyword arguments
    -----------------
    population : ndarray
        El arreglo que contiene la información de la población

    destinations : ndarray
        El arreglo que contiene la información de los destinos

    wander_factor : int or float
        define a qué distancia fuera del "rango de recorrido" se alcanza el destino se activa
    """

    active_dests = np.unique(
        population[:, 11][(population[:, 11] != 0) & (population[:, 12] == 1)]
    )

    for d in active_dests:
        dest_x = destinations[:, int((d - 1) * 2)][
            (population[:, 12] == 1) & (population[:, 11] == d)
        ]
        dest_y = destinations[:, int(((d - 1) * 2) + 1)][
            (population[:, 12] == 1) & (population[:, 11] == d)
        ]

        arrived = population[(population[:, 12] == 1) & (population[:, 11] == d)]

        ids = np.int32(arrived[:, 0])  # find unique IDs of arrived persons

        shp = arrived[:, 3][
            arrived[:, 1] > (dest_x + (arrived[:, 13] * wander_factor))
        ].shape

        arrived[:, 3][
            arrived[:, 1] > (dest_x + (arrived[:, 13] * wander_factor))
        ] = -np.random.normal(loc=0.5, scale=0.5 / 3, size=shp)

        shp = arrived[:, 3][
            arrived[:, 1] < (dest_x - (arrived[:, 13] * wander_factor))
        ].shape
        arrived[:, 3][
            arrived[:, 1] < (dest_x - (arrived[:, 13] * wander_factor))
        ] = np.random.normal(loc=0.5, scale=0.5 / 3, size=shp)
        shp = arrived[:, 4][
            arrived[:, 2] > (dest_y + (arrived[:, 14] * wander_factor))
        ].shape
        arrived[:, 4][
            arrived[:, 2] > (dest_y + (arrived[:, 14] * wander_factor))
        ] = -np.random.normal(loc=0.5, scale=0.5 / 3, size=shp)
        shp = arrived[:, 4][
            arrived[:, 2] < (dest_y - (arrived[:, 14] * wander_factor))
        ].shape
        arrived[:, 4][
            arrived[:, 2] < (dest_y - (arrived[:, 14] * wander_factor))
        ] = np.random.normal(loc=0.5, scale=0.5 / 3, size=shp)

        # Reducir velocidad
        arrived[:, 5] = np.random.normal(
            loc=0.005, scale=0.005 / 3, size=arrived[:, 5].shape
        )

        # Reinsertar en la población
        population[(population[:, 12] == 1) & (population[:, 11] == d)] = arrived

    return population


def reset_destinations(population, ids=[]):
    """Limpia los marcadores de destino

    Función que borra todos los marcadores de destino activos de la población

    Keyword arguments
    -----------------
    population : ndarray
        El arreglo que contiene toda la información de la población

    ids : ndarray or list
        Arreglo de los ids de los miembros de la población que necesitan reiniciar sus destinos
    """

    if len(ids) == 0:
        # Si Ids esta vacio, reiniciar todos los destinos.
        population[:, 11] = 0
    else:
        pass

    pass
