import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar el archivo CSV
datos = pd.read_csv('Problema_2\datos_seno.csv')

# Crear rangos desde -6 hasta 6
rangos = list(range(-6, 7))
conteo = {}

# Inicializar el diccionario para almacenar conteo
for i in range(len(rangos)-1):
    etiqueta = f"{rangos[i]} a {rangos[i+1]}"
    conteo[etiqueta] = 0

# Contar cuántos datos caen en cada rango
for valor in datos['x']:
    for i in range(len(rangos)-1):
        if rangos[i] <= valor < rangos[i+1]:
            etiqueta = f"{rangos[i]} a {rangos[i+1]}"
            conteo[etiqueta] += 1

# Mostrar los resultados
print("Distribución de datos en el archivo datos_seno.csv:")
print("="*50)
print(f"Total de datos: {len(datos)}")
print("="*50)
for rango, cantidad in conteo.items():
    print(f"Rango {rango}: {cantidad} datos")

# Crear una visualización
plt.figure(figsize=(12, 6))
plt.bar(conteo.keys(), conteo.values(), color='skyblue')
plt.title('Distribución de datos por rango en datos_seno.csv')
plt.xlabel('Rango de valores')
plt.ylabel('Cantidad de datos')
plt.xticks(rotation=45)
plt.tight_layout()

# Guardar la gráfica
plt.savefig('./Problema_2/distribucion_datos_seno.png')
print("="*50)
print("Gráfico guardado como 'distribucion_datos_seno.png'")

# Si el entorno lo permite, mostrar la gráfica
try:
    plt.show()
except:
    print("No se pudo mostrar el gráfico (posible entorno sin interfaz gráfica)")