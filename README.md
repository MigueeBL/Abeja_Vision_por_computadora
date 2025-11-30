🐝 Proyecto: Abeja Exploradora con Búsqueda DFS/BFS y Clasificación de Imágenes
Este proyecto implementa una simulación interactiva donde una abejita se mueve dentro de una cuadrícula y debe encontrar un camino desde un punto inicial hasta una meta utilizando los algoritmos de búsqueda en profundidad (DFS) y búsqueda en amplitud (BFS).
Durante su recorrido, la abeja detecta obstáculos, cada uno asociado a una imagen real. Al encontrarse con ellos, el sistema:
🖼️ Procesamiento de imágenes
Para cada imagen asociada al obstáculo, se aplican técnicas de visión por computadora:
•	Subexposición y sobreexposición (en las imágenes del dataset).
•	Conversión a escala de grises.
•	Ecualización de histograma global.
•	Ecualización adaptativa CLAHE.
Las tres versiones se muestran en una ventana emergente cuando la abeja detecta el obstáculo.
🤖 Clasificación con Inteligencia Artificial
El sistema usa un modelo Zero-Shot Image Classification basado en CLIP (openai/clip-vit-base-patch32) para reconocer el contenido de la imagen y clasificarla entre las siguientes categorías:
•	ave, perro, carro, helado, lluvia, flor.
Si el modelo identifica la imagen como flor, se incrementa un contador de "flores detectadas" para el algoritmo que se esté ejecutando (DFS o BFS).
🧭 Características principales
•	Mapa generado aleatoriamente con obstáculos.
•	Interfaz gráfica hecha en Pygame.
•	Selección manual de puntos de inicio y meta para cada algoritmo.
•	Visualización independiente para DFS y BFS.
•	Ventanas emergentes con imagen original, ecualizada y CLAHE.
•	Detección automática de flores mediante IA.
•	Contadores de:
o	Obstáculos encontrados.
o	Flores detectadas por algoritmo.
o	Tiempo total de ejecución.
o	Progreso del recorrido paso a paso.
🎯 Objetivo del proyecto
Este trabajo combina:
•	Algoritmos de búsqueda en inteligencia artificial.
•	Visión computacional mediante preprocesamiento de imágenes.
•	Clasificación Zero-Shot usando modelos modernos de machine learning.
•	Interacción visual a través de Pygame y Tkinter.
