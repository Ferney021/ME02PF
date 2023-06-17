"""
file that contains all configuration related methods and classes
"""

import numpy as np


class config_error(Exception):
    pass


class Configuration:
    def __init__(self, *args, **kwargs):
        # variables de simulación
        self.verbose = kwargs.get(
            "verbose", True
        )  # condicional para imprimir reporte de cambios de la simulación en la terminal
        self.simulation_steps = kwargs.get(
            "simulation_steps", 10000
        )  # total de pasos de simulación 
        self.tstep = kwargs.get("tstep", 0)  # tiempo de pasos actual en la simulación
        self.save_data = kwargs.get(
            "save_data", False
        )  # si el reporte final se imprime en la terminal
        self.save_pop = kwargs.get(
            "save_pop", False
        )  # si se guarda la información cada número de pasos
        self.save_pop_freq = kwargs.get(
            "save_pop_freq", 10
        )  # se guardará la información cada cierto número de pasos
        self.save_pop_folder = kwargs.get(
            "save_pop_folder", "pop_data/"
        )  # escribe datos de pasos temporales
        self.endif_no_infections = kwargs.get(
            "endif_no_infections", True
        )  # si se detiene la simulación al no quedar infecciones
        self.world_size = kwargs.get("world_size", [2, 2])  # x, y tamaños del mundo

        # banderas del escenario
        self.traveling_infects = kwargs.get("traveling_infects", False)
        self.self_isolate = kwargs.get("self_isolate", False)
        self.lockdown = kwargs.get("lockdown", False)
        self.lockdown_percentage = kwargs.get(
            "lockdown_percentage", 0.1
        )  # despues de esta proporción de infectados, el encierro empieza
        self.lockdown_compliance = kwargs.get(
            "lockdown_compliance", 0.95
        )  # fracción de la población que obedece el encierro

        # Variables de simulación
        self.visualise = kwargs.get(
            "visualise", True
        )  # Si se visualiza la simulación
        self.plot_mode = kwargs.get("plot_mode", "sir")
        # tamaño del mundo simulado en coordenadas
        self.x_plot = kwargs.get("x_plot", [0, self.world_size[0]])
        self.y_plot = kwargs.get("y_plot", [0, self.world_size[1]])
        self.save_plot = kwargs.get("save_plot", False)
        self.plot_path = kwargs.get(
            "plot_path", "render/"
        )  # carpeta donde están guardados los plots
        self.plot_style = kwargs.get(
            "plot_style", "default"
        )  # estilo del plot, oscuro, por defecto
        
        # variables del mundo, define donde puede andar la población
        self.xbounds = kwargs.get(
            "xbounds", [self.x_plot[0] + 0.02, self.x_plot[1] - 0.02]
        )
        self.ybounds = kwargs.get(
            "ybounds", [self.y_plot[0] + 0.02, self.y_plot[1] - 0.02]
        )

        # variables de población
        self.pop_size = kwargs.get("pop_size", 2000)
        self.mean_age = kwargs.get("mean_age", 10)
        self.max_age = kwargs.get("max_age", 105)
        self.age_dependent_risk = kwargs.get(
            "age_dependent_risk", True
        )  # si el riesgo incrementa con la edad
        self.risk_age = kwargs.get(
            "risk_age", 55
        )  # edad en la que el riesgo de mortalidad incrementa
        self.critical_age = kwargs.get(
            "critical_age", 75
        )  # edad en la que el riesgo de mortalidad alcanza su máximo
        self.critical_mortality_chance = kwargs.get(
            "critical_mortality_chance", 0.1
        )  # riesgo máximo de mortalidad 
        self.risk_increase = kwargs.get(
            "risk_increase", "quadratic"
        )  # si el riesgo entre el riesgo y la edad crítica aumenta lineal o cuadráticamente

        # variables de movimiento
        self.proportion_distancing = kwargs.get("proportion_distancing", 0)
        self.speed = kwargs.get("speed", 0.01)  # velocidad media de la población
        # cuando hay un destino activo, se define la zona que rodea el destino por la que andarán hasta llegar a él
        self.wander_range = kwargs.get("wander_range", 0.05)
        self.wander_factor = kwargs.get("wander_factor", 1)
        self.wander_factor_dest = kwargs.get(
            "wander_factor_dest", 1.5
        )  # área que rodea el destino

        # variables de infección
        self.infection_range = kwargs.get(
            "infection_range", 0.01
        )  # rango que rodea al paciente enfermo
        self.infection_chance = kwargs.get(
            "infection_chance", 0.03
        )  # posibilidad de que una infección se propague a personas sanas cercanas cada tick
        self.recovery_duration = kwargs.get(
            "recovery_duration", (200, 500)
        )  # cuántos ticks se requieren para mejorar
        self.mortality_chance = kwargs.get(
            "mortality_chance", 0.02
        )  # posibilidad global de morir a causa de la enfermedad

        # variables sanitarias
        self.healthcare_capacity = kwargs.get(
            "healthcare_capacity", 300
        )  # capacidad de camas
        self.treatment_factor = kwargs.get(
            "treatment_factor", 0.5
        )  # afecta el riesgo por este valor cuando está en tratamiento
        self.no_treatment_factor = kwargs.get(
            "no_treatment_factor", 3
        )  # factor de aumento del riesgo a utilizar si el sistema sanitario está lleno
        # parámetros de riego
        self.treatment_dependent_risk = kwargs.get(
            "treatment_dependent_risk", True
        )  # si el riesgo es afectado por el tratamiento

        # variables de auto aislamiento
        self.self_isolate_proportion = kwargs.get("self_isolate_proportion", 0.6)
        self.isolation_bounds = kwargs.get("isolation_bounds", [0.02, 0.02, 0.1, 0.98])

        # variables de encierro
        self.lockdown_percentage = kwargs.get("lockdown_percentage", 0.1)
        self.lockdown_vector = kwargs.get("lockdown_vector", [])

    def get_palette(self):

        # los colores son: [sano, infectado, inmune, fallecido]
        palettes = {
            "regular": {
                "default": ["gray", "red", "green", "black"],
                "dark": ["#404040", "#ff0000", "#00ff00", "#000000"],
            },
        }

        
        return palettes["regular"][self.plot_style]

    def set_lockdown(self, lockdown_percentage=0.1, lockdown_compliance=0.9):
        """activa el encierro"""

        self.lockdown = True

        # fracción de la población que obedecerá el encierro
        self.lockdown_percentage = lockdown_percentage
        self.lockdown_vector = np.zeros((self.pop_size,))
        # el vector de encierro es 1 para los que no cumplen
        self.lockdown_vector[
            np.random.uniform(size=(self.pop_size,)) >= lockdown_compliance
        ] = 1

    def set_self_isolation(
        self,
        self_isolate_proportion=0.9,
        isolation_bounds=[0.02, 0.02, 0.09, 0.98],
        traveling_infects=True,
    ):
        """activa el escenario de autoaislamiento"""

        self.self_isolate = True
        self.isolation_bounds = isolation_bounds
        self.self_isolate_proportion = self_isolate_proportion
        # límites de itinerancia fuera del área aislada
        self.xbounds = [0.1, 1.1]
        self.ybounds = [0.02, 0.98]
        # límite del plot
        self.x_plot = [0, 1.1]
        self.y_plot = [0, 1]
        # actualiza si los agentes viajeros también infectan
        self.traveling_infects = traveling_infects

    def set_reduced_interaction(self, speed=0.001):
        """activa el escenario de interacción reducida"""

        self.speed = speed
