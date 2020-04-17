#!/usr/bin/env python3
from entrada import leer_entrada, checkear_entrada
from CN2_SD import CN2_SD

# Entrada del usuario:
# fichero : fichero con los datos
# salida : nombre de la columna que es la salida
# clase_positiva : nombre de la clase positiva

# Valores para el ejemplo
fichero = "entrada.csv"
columna_salida = "Survived"
clase_positiva = "Yes"


def ejecutar_algoritmo(fichero, columna_salida, clase_positiva):
    """
    Función que ejecuta los algoritmos PRIM / C50-SD de búsqueda de subgrupos
    sobre un dataset de entrada.

    :param fichero: Nombre del fichero con los datos
    :param columna_salida: Nombre de la columna que es la salida
    :param clase_positiva: Nombre de la clase positiva
    :return:
    """
    entrada = leer_entrada(fichero)
    checkear_entrada(df=entrada, col=columna_salida, positiva=clase_positiva)
    cn2 = CN2_SD(entrada, columna_salida, clase_positiva, max_exp=2, weight_method=1, gamma=0.1, min_wracc=0.001)
    cn2.execute()


ejecutar_algoritmo(fichero, columna_salida, clase_positiva)
