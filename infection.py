"""
contiene todas las funciones necesarias para calcular
nuevas infecciones, recuperaciones y muertes
"""


import numpy as np
from path_planning import go_to_location


def find_nearby(
    population,
    infection_zone,
    traveling_infects=False,
    kind="healthy",
    infected_previous_step=[],
):
    """
    Argumentos clave
    -----------------

    kind : str (puede ser 'healthy' o 'infected')
        determina si se devuelven individuos infectados o sanos 
        dentro de la zona de infección


    Retorna
    -------
    if kind='healthy', se devuelven los índices de agentes sanos dentro de la zona de 
    infección. Esto se debe a que para cada agente sano, 
    se debe evaluar la posibilidad de infectarse.

    if kind='infected', solo se devuelve el número de infectados dentro de la zona de 
    infección. Esto se debe a que, en esta situación, las probabilidades de que el 
    agente sano esté en el centro de la zona de infección dependen de cuántos agentes 
    infecciosos haya a su alrededor.
    """

    if kind.lower() == "healthy":
        indices = np.int32(
            population[:, 0][
                (infection_zone[0] < population[:, 1])
                & (population[:, 1] < infection_zone[2])
                & (infection_zone[1] < population[:, 2])
                & (population[:, 2] < infection_zone[3])
                & (population[:, 6] == 0)
            ]
        )
        return indices

    elif kind.lower() == "infected":
        if traveling_infects:
            infected_number = len(
                infected_previous_step[:, 6][
                    (infection_zone[0] < infected_previous_step[:, 1])
                    & (infected_previous_step[:, 1] < infection_zone[2])
                    & (infection_zone[1] < infected_previous_step[:, 2])
                    & (infected_previous_step[:, 2] < infection_zone[3])
                    & (infected_previous_step[:, 6] == 1)
                ]
            )
        else:
            infected_number = len(
                infected_previous_step[:, 6][
                    (infection_zone[0] < infected_previous_step[:, 1])
                    & (infected_previous_step[:, 1] < infection_zone[2])
                    & (infection_zone[1] < infected_previous_step[:, 2])
                    & (infected_previous_step[:, 2] < infection_zone[3])
                    & (infected_previous_step[:, 6] == 1)
                    & (infected_previous_step[:, 11] == 0)
                ]
            )
        return infected_number

    else:
        raise ValueError(
            "type to find %s not understood! Must be either 'healthy' or 'ill'"
        )


def infect(
    population,
    Config,
    frame,
    send_to_location=False,
    location_bounds=[],
    destinations=[],
    location_no=1,
    location_odds=1.0,
):
    """encuentra nuevas infecciones

    Función que encuentra nuevas infecciones en un área alrededor de personas infectadas 
    definidas por infection_range, y la oportunidad de infertar a otros infection_chance

    Argumentos clave
    -----------------
    population : ndarray
        matriz que contiene los datos sobre la población

    pop_size : int
        número de individuos en la población

    infection_range : float
        radio alrededor de cada persona infectada donde puede tener lugar 
        la transmisión del virus

    infection_chance : float
        probabilidades de que el virus infecte a alguien dentro del rango (rango 0 a 1)

    frame : int
        paso de tiempo actual en la simulación

    healthcare_capacity : int
        número de camas disponibles en el sistema sanitario

    verbose : bool
        si reportar eventos de enfermedad

    send_to_location : bool
        si dar a las personas infectadas un destino

    location_bounds : list
        límites de ubicación a donde se envía a la persona infectada y puede deambular

    destinations : list or ndarray
        vector de destinos que contiene destinos para cada individuo de la población.

    location_no : int
        índice para la matriz de destinos si se definen múltiples destinos posibles

    location_odds: float
        probabilidades de que alguien vaya a un lugar o no.

    traveling_infects : bool
        si las personas infectadas que se dirigen a un destino aún pueden infectar a otros en el camino hacia allí
    """

    # marcar primero a los que ya están infectados
    infected_previous_step = population[population[:, 6] == 1]
    healthy_previous_step = population[population[:, 6] == 0]

    new_infections = []

    # si menos de la mitad están infectados, se divide en función de los infectados para acelerar el cálculo
    if len(infected_previous_step) < (Config.pop_size // 2):
        for patient in infected_previous_step:
            # zona de infección para el paciente
            infection_zone = [
                patient[1] - Config.infection_range,
                patient[2] - Config.infection_range,
                patient[1] + Config.infection_range,
                patient[2] + Config.infection_range,
            ]

            # personas sanas que rodean al paciente infectado
            if Config.traveling_infects or patient[11] == 0:
                indices = find_nearby(population, infection_zone, kind="healthy")
            else:
                indices = []

            for idx in indices:
                # tirar el dado para ver si una persona sana se infecta
                if np.random.random() < Config.infection_chance:
                    population[idx][6] = 1
                    population[idx][8] = frame
                    if (
                        len(population[population[:, 10] == 1])
                        <= Config.healthcare_capacity
                    ):
                        population[idx][10] = 1
                        if send_to_location:
                            # enviar a la ubicación si la tirada es positiva
                            if np.random.uniform() <= location_odds:
                                population[idx], destinations[idx] = go_to_location(
                                    population[idx],
                                    destinations[idx],
                                    location_bounds,
                                    dest_no=location_no,
                                )
                        else:
                            pass
                    new_infections.append(idx)

    else:
        # si más de la mitad están infectados, basado en personas sanas para acelerar el cálculo

        for person in healthy_previous_step:
            # Definir el rango de infección en torno a una persona sana.
            infection_zone = [
                person[1] - Config.infection_range,
                person[2] - Config.infection_range,
                person[1] + Config.infection_range,
                person[2] + Config.infection_range,
            ]

            if (
                person[6] == 0
            ):  # si la persona aún no está infectada, averiguar si hay infectados cerca
                # encontrar una persona sana cercana infectada
                if Config.traveling_infects:
                    poplen = find_nearby(
                        population,
                        infection_zone,
                        traveling_infects=True,
                        kind="infected",
                    )
                else:
                    poplen = find_nearby(
                        population,
                        infection_zone,
                        traveling_infects=True,
                        kind="infected",
                        infected_previous_step=infected_previous_step,
                    )

                if poplen > 0:
                    if np.random.random() < (Config.infection_chance * poplen):
                        # Tira el dado para ver si la persona sana se infectará
                        population[np.int32(person[0])][6] = 1
                        population[np.int32(person[0])][8] = frame
                        if (
                            len(population[population[:, 10] == 1])
                            <= Config.healthcare_capacity
                        ):
                            population[np.int32(person[0])][10] = 1
                            if send_to_location:
                                # enviar al lugar y añadir al tratamiento si la tirada es positiva
                                if np.random.uniform() < location_odds:
                                    (
                                        population[np.int32(person[0])],
                                        destinations[np.int32(person[0])],
                                    ) = go_to_location(
                                        population[np.int32(person[0])],
                                        destinations[np.int32(person[0])],
                                        location_bounds,
                                        dest_no=location_no,
                                    )

                        new_infections.append(np.int32(person[0]))

    if len(new_infections) > 0 and Config.verbose:
        print("\nat timestep %i these people got sick: %s" % (frame, new_infections))

    if len(destinations) == 0:
        return population
    else:
        return population, destinations


def recover_or_die(population, frame, Config):
    """ver si recuperarse o morir


    Argumentos Clave
    -----------------
    population : ndarray
        matriz que contiene todos los datos sobre la población

    frame : int
        El instatne de tiempo actual de la simulación

    recovery_duration : tuple
        límites inferior y superior de la duración de la recuperación, en pasos de simulación

    mortality_chance : float
        las probabilidades de que alguien muera en lugar de recuperarse (entre 0 y 1)

    risk_age : int or flaot
        la edad a partir de la cual empieza a aumentar el riesgo de mortalidad

    critical_age: int or float
        la edad en la que el riesgo de mortalidad es igual al critical_mortality_change

    critical_mortality_chance : float
        las mayores probabilidades de que una persona infectada tenga un desenlace fatal

    risk_increase : string
        puede ser "cuadrática" o "lineal", determina si el riesgo de mortalidad entre 
        la edad de riesgo y la edad crítica aumenta de forma lineal o exponencialmente

    no_treatment_factor : int or float
        defines a change in mortality odds if someone cannot get treatment. Can
        be larger than one to increase risk, or lower to decrease it.

    treatment_dependent_risk : bool
        si la disponibilidad de tratamiento influye en el riesgo del paciente

    treatment_factor : int or float
        define un cambio en las probabilidades de mortalidad si alguien está en tratamiento. 
        Puede ser mayor que uno para aumentar el riesgo, o menor para disminuirlo.

    verbose : bool
        si se informa a la terminal de las recuperaciones y muertes de cada paso de la simulación
    """

    # Encontrar personas infectadas
    infected_people = population[population[:, 6] == 1]

    # Define el vector de por cuanto tiempo la persona va a estar enferma
    illness_duration_vector = frame - infected_people[:, 8]

    recovery_odds_vector = (
        illness_duration_vector - Config.recovery_duration[0]
    ) / np.ptp(Config.recovery_duration)
    recovery_odds_vector = np.clip(recovery_odds_vector, a_min=0, a_max=None)

    # Actualiza el estado de los riesgos de las personas
    indices = infected_people[:, 0][recovery_odds_vector >= infected_people[:, 9]]

    recovered = []
    fatalities = []

    # decide whether to die or recover
    for idx in indices:
        # Si la edad altera el riesgo
        if Config.age_dependent_risk:
            updated_mortality_chance = compute_mortality(
                infected_people[infected_people[:, 0] == idx][:, 7][0],
                Config.mortality_chance,
                Config.risk_age,
                Config.critical_age,
                Config.critical_mortality_chance,
                Config.risk_increase,
            )
        else:
            updated_mortality_chance = Config.mortality_chance

        if (
            infected_people[infected_people[:, 0] == int(idx)][:, 10] == 0
            and Config.treatment_dependent_risk
        ):
            # Si la persona no esta en tratamiento se aumenta el riesgo 
            updated_mortality_chance = (
                updated_mortality_chance * Config.no_treatment_factor
            )
        elif (
            infected_people[infected_people[:, 0] == int(idx)][:, 10] == 1
            and Config.treatment_dependent_risk
        ):
            # Si la persona se encuentra en tratamiento, disminuye el riesgo
            updated_mortality_chance = (
                updated_mortality_chance * Config.treatment_factor
            )

        if np.random.random() <= updated_mortality_chance:
            # Muere
            infected_people[:, 6][infected_people[:, 0] == idx] = 3
            infected_people[:, 10][infected_people[:, 0] == idx] = 0
            fatalities.append(
                np.int32(infected_people[infected_people[:, 0] == idx][:, 0][0])
            )
        else:
            # Recuperado: se vuelve inmune
            infected_people[:, 6][infected_people[:, 0] == idx] = 2
            infected_people[:, 10][infected_people[:, 0] == idx] = 0
            recovered.append(
                np.int32(infected_people[infected_people[:, 0] == idx][:, 0][0])
            )

    if len(fatalities) > 0 and Config.verbose:
        print("\nat timestep %i these people died: %s" % (frame, fatalities))
    if len(recovered) > 0 and Config.verbose:
        print("\nat timestep %i these people recovered: %s" % (frame, recovered))

    # Reinsertar a la población
    population[population[:, 6] == 1] = infected_people

    return population


def compute_mortality(
    age,
    mortality_chance,
    risk_age=50,
    critical_age=80,
    critical_mortality_chance=0.5,
    risk_increase="linear",
):
    """Calcular la mortalidad basado en la edad

   El riesgo se calcula en función de la edad. la edad en la que el riesgo empieza
     a aumentar, y la edad crticial marca el momento en el que las 'critical_mortality_odds' 
     se convierten en la nueva probabilidad de mortalidad.

    Se puede establecer si el riesgo aumenta de forma lineal o cuadrática.

    Argumentos clave
    -----------------
    age : int
        La edad de la persona

    mortality_chance : float
        La probabilidad base de mortalidad

    risk_age : int
        La edad a la que el riesgo empieza a aumentar

    critical_age : int
        la edad en la que el riesgo de mortalidad es igual a la
        critical_mortality_odds

    critical_mortality_chance : float
        las probabilidades de morir a la edad crítica

    risk_increase : str
       define si el riesgo de mortalidad entre la edad de riesgo y la edad 
       crítica aumenta de forma lineal o exponencial
    """

    if risk_age < age < critical_age:  # Si esta en el rango de edades
        if risk_increase == "linear":
            # Encontrar riesgo lineal
            step_increase = (critical_mortality_chance) / (
                (critical_age - risk_age) + 1
            )
            risk = critical_mortality_chance - ((critical_age - age) * step_increase)
            return risk
        elif risk_increase == "quadratic":
            # define funcion exponencial entre risk_age y critical_age
            pw = 15
            A = np.exp(np.log(mortality_chance / critical_mortality_chance) / pw)
            a = ((risk_age - 1) - critical_age * A) / (A - 1)
            b = mortality_chance / ((risk_age - 1) + a) ** pw

            # definir espacio lineal
            x = np.linspace(0, critical_age, critical_age)
            # Encuetra valores
            risk_values = ((x + a) ** pw) * b
            return risk_values[np.int32(age - 1)]
    elif age <= risk_age:
        # Retorna la probabilidad de muerte
        return mortality_chance
    elif age >= critical_age:
        # Returna la probabilidad critica de muerte 
        return critical_mortality_chance


def healthcare_infection_correction(worker_population, healthcare_risk_factor=0.2):
    """corrige la infección a la población sanitaria.

    Toma el factor de riesgo sanitario y ajusta el personal sanitario enfermo reduciendo (si < 0) o 
    aumentando (si > 0) el personal sanitario enfermo

    Keyword arguments
    -----------------
    worker_population : ndarray
        la matriz que contiene todas las variables relacionadas con la población sanitaria. 
        Es un subconjunto de la matriz "población".

    healthcare_risk_factor : int or float
        si es distinto de uno, define el cambio en las probabilidades de contraer una infección. 
        Puede utilizarse para simular que el personal sanitario dispone de protecciones adicionales 
        (< 1) o que corren más riesgo debido a la exposición, la fatiga u otros factores (> 1)
    """

    if healthcare_risk_factor < 0:
        # Configurar 1 - healthcare_risk_factor no enfermos
        sick_workers = worker_population[:, 6][worker_population[:, 6] == 1]
        cure_vector = np.random.uniform((len(sick_workers)))
        sick_workers[:, 6][cure_vector >= healthcare_risk_factor] = 0
    elif healthcare_risk_factor > 0:
        # TODO: make proportion of extra workers sick
        pass
    else:
        pass  # Si no cambia el riesgo, no hacer nada

    return worker_population
