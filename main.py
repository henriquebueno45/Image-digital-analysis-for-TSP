import cv2
import csv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

img = cv2.imread('imagem.png', 0)

# Aplicando thresholding na imagem
ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Detectando os pontos pretos na imagem thresholded
points = []
for y in range(thresh.shape[0]):
    for x in range(thresh.shape[1]):
        if thresh[y, x] == 0:
            points.append((x, y))

# Salvando a tabela de pontos em um arquivo CSV
with open('arquivo_modificado.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x', 'y'])
    writer.writerows(points)

# Criando um gráfico de dispersão com os pontos
df = pd.read_csv('points.csv')
plt.scatter(df['x'], df['y'], s=5)
plt.gca().invert_yaxis()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# leia o arquivo csv
df = pd.read_csv('points.csv')

# distância máxima permitida entre os pontos
max_distance = 0.5

# lista para armazenar os pontos médios
new_x = []
new_y = []

# itere sobre as linhas do DataFrame
for i, row in df.iterrows():
    x = row['x'] # nome da coluna que contém as coordenadas x
    y = row['y'] # nome da coluna que contém as coordenadas y
    
    # encontre pontos próximos
    near_points = []
    for j, other_row in df.iloc[i+1:].iterrows():
        other_x = other_row['x']
        other_y = other_row['y']
        distance = np.sqrt((x-other_x)**2 + (y-other_y)**2)
        if distance < max_distance:
            near_points.append([other_x, other_y])
    
    # calcule o ponto médio se houver pontos próximos
    if len(near_points) > 0:
        near_points.append([x, y])
        avg_x = sum([p[0] for p in near_points]) / len(near_points)
        avg_y = sum([p[1] for p in near_points]) / len(near_points)
        new_x.append(avg_x)
        new_y.append(avg_y)
    else:
        new_x.append(x)
        new_y.append(y)

# plote o novo conjunto de dados com os pontos médios adicionados
plt.scatter(new_x, new_y)
plt.show()

