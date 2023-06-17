import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from config import Configuration, config_error
from infection import (
    infect,
    recover_or_die,
)
from motion import (
    update_positions,
    out_of_bounds,
    update_randoms,
)
from path_planning import (
    set_destination,
    check_at_destination,
    keep_at_destination,
)
from population import (
    initialize_population,
    initialize_destination_matrix,
    save_data,
    save_population,
    Population_trackers,
)
from visualiser import build_fig, draw_tstep, plot_sir


class Simulation:
    def __init__(self, *args, **kwargs):
        # Cargar la configuracion por defecto
        self.Config = Configuration(*args, **kwargs)
        self.frame = 0

        # Inicializarla poblacion por defecto
        self.population_init()

        self.pop_tracker = Population_trackers()

        # Inicializar los vectores de destino
        self.destinations = initialize_destination_matrix(self.Config.pop_size, 1)

    def population_init(self):
        """Re-Inicializa la poblacion"""
        self.population = initialize_population(
            self.Config,
            self.Config.mean_age,
            self.Config.max_age,
            self.Config.xbounds,
            self.Config.ybounds,
        )

    def tstep(self):
        """
        Toma un instante de tiempo en la simulación
        """

        if self.frame == 0 and self.Config.visualise:
            # Mostrar ventana
            self.fig, self.spec, self.ax1, self.ax2 = build_fig(self.Config)

        # Verificar que el destino este activo
        # Definir vectores de movimiento
        active_dests = len(self.population[self.population[:, 11] != 0])

        if active_dests > 0 and len(self.population[self.population[:, 12] == 0]) > 0:
            self.population = set_destination(self.population, self.destinations)
            self.population = check_at_destination(
                self.population,
                self.destinations,
                wander_factor=self.Config.wander_factor_dest,
                speed=self.Config.speed,
            )

        if active_dests > 0 and len(self.population[self.population[:, 12] == 1]) > 0:
            self.population = keep_at_destination(
                self.population, self.destinations, self.Config.wander_factor
            )

        # Fuera de limites
        # Se definen arreglos de limites
        if len(self.population[:, 11] == 0) > 0:
            _xbounds = np.array(
                [[self.Config.xbounds[0] + 0.02, self.Config.xbounds[1] - 0.02]]
                * len(self.population[self.population[:, 11] == 0])
            )
            _ybounds = np.array(
                [[self.Config.ybounds[0] + 0.02, self.Config.ybounds[1] - 0.02]]
                * len(self.population[self.population[:, 11] == 0])
            )
            self.population[self.population[:, 11] == 0] = out_of_bounds(
                self.population[self.population[:, 11] == 0], _xbounds, _ybounds
            )

        # Variables aleatorias
        if self.Config.lockdown:
            if len(self.pop_tracker.infectious) == 0:
                mx = 0
            else:
                mx = np.max(self.pop_tracker.infectious)

            if len(self.population[self.population[:, 6] == 1]) >= len(
                self.population
            ) * self.Config.lockdown_percentage or mx >= (
                len(self.population) * self.Config.lockdown_percentage
            ):
                # Reduce la velocidad de todos los miembros de la sociedad
                self.population[:, 5] = np.clip(
                    self.population[:, 5], a_min=None, a_max=0.001
                )
                # Ajustar la velocidad a 0 para las personas que cumplen la condición
                self.population[:, 5][self.Config.lockdown_vector == 0] = 0
            else:
                # Actualizar valores aleatorios
                self.population = update_randoms(
                    self.population, self.Config.pop_size, self.Config.speed
                )
        else:
            # Actualizar valores aleatorios
            self.population = update_randoms(
                self.population, self.Config.pop_size, self.Config.speed
            )

        # Para estados (dead) pone la velocidad en 0
        self.population[:, 3:5][self.population[:, 6] == 3] = 0

        # Actualizar pocisiones
        self.population = update_positions(self.population)

        # Infectar
        self.population, self.destinations = infect(
            self.population,
            self.Config,
            self.frame,
            send_to_location=self.Config.self_isolate,
            location_bounds=self.Config.isolation_bounds,
            destinations=self.destinations,
            location_no=1,
            location_odds=self.Config.self_isolate_proportion,
        )

        # Se decide el futuro de la persona
        self.population = recover_or_die(self.population, self.frame, self.Config)

        # Envia los curados de vuelta a la población
        self.population[:, 11][self.population[:, 6] == 2] = 0

        # Actualiza las estadisticas de la población
        self.pop_tracker.update_counts(self.population)

        # Mostrar gráfico
        if self.Config.visualise:
            draw_tstep(
                self.Config,
                self.population,
                self.pop_tracker,
                self.frame,
                self.fig,
                self.spec,
                self.ax1,
                self.ax2,
            )

        # Reportes por consola
        sys.stdout.write("\r")
        sys.stdout.write(
            "%i: Sanos: %i, Infectados: %i, Inmune: %i, En tratamiento: %i, \
Fallecidos: %i, of Total: %i"
            % (
                self.frame,
                self.pop_tracker.susceptible[-1],
                self.pop_tracker.infectious[-1],
                self.pop_tracker.recovered[-1],
                len(self.population[self.population[:, 10] == 1]),
                self.pop_tracker.fatalities[-1],
                self.Config.pop_size,
            )
        )

        # Guardar informacion si se requiere
        if self.Config.save_pop and (self.frame % self.Config.save_pop_freq) == 0:
            save_population(self.population, self.frame, self.Config.save_pop_folder)
        self.callback()

        # Actualizar frame
        self.frame += 1

    def callback(self):
        if self.frame == 50:
            print("\ninfecting patient zero")
            self.population[0][6] = 1
            self.population[0][8] = 50
            self.population[0][10] = 1

    def run(self):
        """Correr Simulación"""

        i = 0

        while i < self.Config.simulation_steps:
            try:
                self.tstep()
            except KeyboardInterrupt:
                print("\nCTRL-C caught, exiting")
                sys.exit(1)

            # Si no quedan personas infectadas
            # Inicialmente sin infectados
            if self.Config.endif_no_infections and self.frame >= 500:
                if (
                    len(
                        self.population[
                            (self.population[:, 6] == 1) | (self.population[:, 6] == 4)
                        ]
                    )
                    == 0
                ):
                    i = self.Config.simulation_steps

        if self.Config.save_data:
            save_data(self.population, self.pop_tracker)

        # Al finalizar la simulación, resumen.
        print("\n-----stopping-----\n")
        print("Instantes de tiempo simulados: %i" % self.frame)
        print("Total fallecidos: %i" % len(self.population[self.population[:, 6] == 3]))
        print(
            "Total recuperados: %i" % len(self.population[self.population[:, 6] == 2])
        )
        print("Total infectados: %i" % len(self.population[self.population[:, 6] == 1]))
        print(
            "Total infecciones: %i"
            % len(
                self.population[
                    (self.population[:, 6] == 1) | (self.population[:, 6] == 4)
                ]
            )
        )
        print(
            "Total NO infectados: %i" % len(self.population[self.population[:, 6] == 0])
        )

    def plot_sir(
        self, size=(6, 3), include_fatalities=False, title="S-I-R plot of simulation"
    ):
        plot_sir(self.Config, self.pop_tracker, size, include_fatalities, title)


if __name__ == "__main__":
    # initialize
    sim = Simulation()

    # ! Pasos en la simulación
    sim.Config.simulation_steps = 20000

    # set color mode
    sim.Config.plot_style = "default"  # puede ser 'dark' para un tema oscuro

    # ! Escenario: Interaccion reducida
    # sim.Config.set_reduced_interaction()
    # sim.population_init()

    # ! Escenario: Encierro
    sim.Config.set_lockdown(lockdown_percentage = 0.1, lockdown_compliance = 0.95)

    # ! Escenario: Auto aislamiento
    # sim.Config.set_self_isolation(
    #     self_isolate_proportion=0.9,
    #     isolation_bounds=[0.02, 0.02, 0.09, 0.98],
    #     traveling_infects=False,
    # )
    # sim.population_init()
    sim.run()
