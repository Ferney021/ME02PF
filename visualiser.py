"""
Contiene todos los metodos para las tareas de visualización
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from environment import build_hospital
from utils import check_folder


def set_style(Config):
    """Estilo de la gráfica"""
    if Config.plot_style.lower() == "dark":
        mpl.style.use("plot_styles/dark.mplstyle")


def build_fig(Config, figsize=(5, 7)):
    set_style(Config)
    fig = plt.figure(figsize=(5, 7))
    spec = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[5, 2])

    ax1 = fig.add_subplot(spec[0, 0])
    plt.title("Simulación de infecciones")
    plt.xlim(Config.xbounds[0], Config.xbounds[1])
    plt.ylim(Config.ybounds[0], Config.ybounds[1])

    ax2 = fig.add_subplot(spec[1, 0])
    ax2.set_title("Numero de infectados")
    ax2.set_ylim(0, Config.pop_size + 100)
    return fig, spec, ax1, ax2


def draw_tstep(Config, population, pop_tracker, frame, fig, spec, ax1, ax2):
    # Construir gráfica y visualizar 

    # Estilo del la gráfica
    set_style(Config)

    # Paleta de colores
    palette = Config.get_palette()

    spec = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[5, 2])
    ax1.clear()
    ax2.clear()

    ax1.set_xlim(Config.x_plot[0], Config.x_plot[1])
    ax1.set_ylim(Config.y_plot[0], Config.y_plot[1])

    if Config.self_isolate and Config.isolation_bounds != None:
        build_hospital(
            Config.isolation_bounds[0],
            Config.isolation_bounds[2],
            Config.isolation_bounds[1],
            Config.isolation_bounds[3],
            ax1
        )

    # Segmentos de población
    healthy = population[population[:, 6] == 0][:, 1:3]
    ax1.scatter(healthy[:, 0], healthy[:, 1], color=palette[0], s=2, label="healthy")

    infected = population[population[:, 6] == 1][:, 1:3]
    ax1.scatter(infected[:, 0], infected[:, 1], color=palette[1], s=2, label="infected")

    immune = population[population[:, 6] == 2][:, 1:3]
    ax1.scatter(immune[:, 0], immune[:, 1], color=palette[2], s=2, label="immune")

    fatalities = population[population[:, 6] == 3][:, 1:3]
    ax1.scatter(fatalities[:, 0], fatalities[:, 1], color=palette[3], s=2, label="dead")

    ax1.text(
        Config.x_plot[0],
        Config.y_plot[1] + ((Config.y_plot[1] - Config.y_plot[0]) / 100),
        "Instante de tiempo: %i, Total: %i, Sanos: %i Infectados: %i Inmune: %i Fallecidos: %i"
        % (
            frame,
            len(population),
            len(healthy),
            len(infected),
            len(immune),
            len(fatalities),
        ),
        fontsize=6,
    )

    ax2.set_title("Numero de infectados")
    ax2.set_ylim(0, Config.pop_size + 200)

    if Config.treatment_dependent_risk:
        ax2.plot(
            [Config.healthcare_capacity for x in range(len(pop_tracker.infectious))],
            "r:",
            label="Capacidad sanitaria",
        )

    if Config.plot_mode.lower() == "sir":
        ax2.plot(pop_tracker.susceptible, color=palette[0], label="Susceptible")
        ax2.plot(pop_tracker.infectious, color=palette[1], label="Infecciones")
        ax2.plot(pop_tracker.recovered, color=palette[2], label="Recuperados")
        ax2.plot(pop_tracker.fatalities, color=palette[3], label="Fallecidos")
    else:
        raise ValueError("Valor incorrecto use 'sir' para ver los datos completos")

    ax2.legend(loc="best", fontsize=6)

    plt.draw()
    plt.pause(0.0001)

    if Config.save_plot:
        try:
            plt.savefig("%s/%i.png" % (Config.plot_path, frame))
        except:
            check_folder(Config.plot_path)
            plt.savefig("%s/%i.png" % (Config.plot_path, frame))


def plot_sir(
    Config,
    pop_tracker,
    size=(6, 3),
    include_fatalities=False,
    title="S-I-R plot of simulation",
):
  
    set_style(Config)

    palette = Config.get_palette()

    plt.figure(figsize=size)
    plt.title(title)
    plt.plot(pop_tracker.susceptible, color=palette[0], label="Susceptible")
    plt.plot(pop_tracker.infectious, color=palette[1], label="Infecciones")
    plt.plot(pop_tracker.recovered, color=palette[2], label="Recuperados")
    if include_fatalities:
        plt.plot(pop_tracker.fatalities, color=palette[3], label="Fallecidos")

    # add axis labels
    plt.xlabel("Tiempo en horas")
    plt.ylabel("Población")

    # add legend
    plt.legend()

    # beautify
    plt.tight_layout()

    # initialise
    plt.show()
