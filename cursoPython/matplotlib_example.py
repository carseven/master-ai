#%%
import matplotlib.pyplot as plt
import numpy as np

#%% Plot
values_x = np.array([0, 1, 2, 3, 4])
values_y = np.array([4, 3, 2, 1, 0])

# Primero el color y luego el tipo de línea
plt.plot(values_x, values_y, 'b--', label='Línea 1', linewidth=5)
plt.plot(values_x * 2, values_y * 2, 'r^', label='Línea 2', linewidth=5)

plt.legend()
plt.xlabel('Eje x')
plt.ylabel('Eje y')

plt.show()

#%% Histogramas
values = np.random.randn(1000)
values2 = np.random.randn(1000)
plt.hist(values, orientation='horizontal', color='yellow', range=range(-1, 1))
plt.show()

plt.hist2d(values, values2)
plt.show()

#%% Barras
fig = plt.figure()
ax = fig.add_axes([0, 1, 1, 1])

values_x =  ['a', 'b', 'c', 'd', 'f']
values_y = np.random.randint(low=1, high=20, size=5)

ax.bar(values_x, values_y)
plt.show()

#%% Barras multiple
n_data = 4
values_1 = (90, 55, 40, 65)
values_2 = (85, 62, 54, 20)

# Configurar el plot
fig, aux = plt.subplots()
index = np.arange(n_data)
bar_wight = 0.35

# Configuración de las dos barras
barras_1 = plt.bar(index, 
                   values_1,
                   bar_wight, 
                   color='r', 
                   label='Valores 1')
# Añadimos los índices desplazados bar_wight para que salga la otra barra al 
# lado. Si ponemos index sale todo en una misma barra.
barras_2 = plt.bar(index + bar_wight, 
                   values_2,
                   bar_wight, 
                   color='b', 
                   label='Valores 2')

# Títulos y leyendas
plt.xlabel('Etiquetas')
plt.ylabel('Valores')
plt.title('Título del gráfico')

# Leyenda para cada barra
plt.xticks(index + bar_wight / 2, ('A', 'B', 'C', 'D'))
plt.legend()

# Color de fondo blanco
fig.patch.set_facecolor('#ffffff')

plt.tight_layout() # Para que salgan dos gráficas y todas en uno
fig.savefig('grafica.png', facecolor=fig.get_facecolor(), edgecolor='none')

plt.show()



#%% Pie
valores = np.array([0.1, 0.3, 0.2, 0.4])
labels = ['Valor 1', 'Valor 2', 'Valor 3', 'Valor 4']

plt.pie(valores, labels=labels, radius=2, autopct="%.2f%%")
plt.show()