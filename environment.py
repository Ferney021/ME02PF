"""
contiene todas las funciones para definir destinos en el 
entorno del mundo simulado.
"""

import numpy as np

# Retorna las coordenadas de las paredes para el hospital
def build_hospital(xmin, xmax, ymin, ymax, plt):

    # paredes del plot
    plt.plot([xmin, xmin], [ymin, ymax], color="black")
    plt.plot([xmax, xmax], [ymin, ymax], color="black")
    plt.plot([xmin, xmax], [ymin, ymin], color="black")
    plt.plot([xmin, xmax], [ymax, ymax], color="black")
