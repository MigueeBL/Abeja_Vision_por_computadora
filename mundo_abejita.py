import pygame
import random
import time
from collections import deque
import os
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import matplotlib.pyplot as plt

from transformers import pipeline
from PIL import Image

# Mapa global de posiciones de obstáculos a imágenes
OBSTACLE_IMAGE_MAP = {}

# Contador global de flores detectadas
SCORE_FLORES = {"DFS": 0, "BFS": 0}

# Dimensiones de la cuadrícula
N = 10
CELL_SIZE = 25
WINDOW_SIZE = N * CELL_SIZE
WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 500

# Colores RGB
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (105, 147, 255)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
YELLOW = (255, 255, 0)

# Cargar clasificador zero-shot para imágenes
classifier = pipeline("zero-shot-image-classification", 
                     model="openai/clip-vit-base-patch32",
                     use_fast=True)

# Definir las categorías temáticas
CANDIDATE_LABELS = ["ave", "perro", "carro", "helado", "lluvia", "flor"]


# Construcción del mundo con diccionario
def crear_mundo(n, num_obstaculos=20):
    mundo = {}
    for i in range(n):
        for j in range(n):
            mundo[(i, j)] = " "  # espacio vacío

    # Obstáculos aleatorios
    for _ in range(num_obstaculos):
        x, y = random.randint(0, n-1), random.randint(0, n-1)
        mundo[(x, y)] = "X"  # obstáculo

    return mundo

def cargar_imagenes_de_carpeta(folder_name):
    base_dir = os.path.join(os.path.dirname(__file__), folder_name)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Carpeta no encontrada: {base_dir}")
    archivos = [os.path.join(base_dir, f) for f in os.listdir(base_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    if not archivos:
        raise FileNotFoundError(f"No se encontraron imágenes en {base_dir}")
    return archivos

def asignar_imagenes_a_obstaculos(mundo, image_list):
    mapping = {}
    obstaculos = [pos for pos, val in mundo.items() if val == "X"]
    
    # Mezclar aleatoriamente la lista de imágenes
    random.shuffle(image_list)
    
    # Si hay menos imágenes que obstáculos, se repetirán algunas
    for i, pos in enumerate(obstaculos):
        mapping[pos] = image_list[i % len(image_list)]
    return mapping


def mostrar_imagen_ventana(image_path):
    # Inicialización perezosa del root de Tk (oculto)
    if not hasattr(mostrar_imagen_ventana, "_root"):
        mostrar_imagen_ventana._root = tk.Tk()
        mostrar_imagen_ventana._root.withdraw()

    win = tk.Toplevel(mostrar_imagen_ventana._root)
    win.title("¡Obstáculo detectado!")
    # Asegura que salga al frente (puedes quitar si molesta)
    try:
        win.attributes("-topmost", True)
    except Exception:
        pass
    
    # Cargar imagen en escala de grises
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img_gray is None:
        print(f"No se pudo cargar la imagen {image_path}")
        return
    
    # Ecualización de histograma normal
    eq_hist = cv2.equalizeHist(img_gray)
    
    # Ecualización adaptativa CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(img_gray)
    
    # Convertir de nuevo a formato RGB para mostrar con PIL
    # img_rgb = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)

    # Convertir a objeto PIL
    # img = Image.fromarray(img_rgb)
    
    # Clasificación con el modelo
    try:
        img_for_model = Image.fromarray(cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB))
        results = classifier(img_for_model, candidate_labels=CANDIDATE_LABELS)
        best_label = results[0]['label']
        best_score = results[0]['score']
        classification_text = f"El modelo detecta: {best_label}" #({best_score:.2f}
        
        # 🟢 Si detecta una flor, sumar al score del algoritmo activo
        if best_label.lower() == "flor":
            # Detectar algoritmo actual (DFS o BFS)
            import inspect
            current_frame = inspect.currentframe()
            while current_frame:
                if "algoritmo" in current_frame.f_locals:
                    algoritmo_actual = current_frame.f_locals["algoritmo"]
                    break
                current_frame = current_frame.f_back
            else:
                algoritmo_actual = "DFS"  # Por defecto
                
            if algoritmo_actual in SCORE_FLORES:
                SCORE_FLORES[algoritmo_actual] += 1
                print(f"🌸 Flor detectada ({algoritmo_actual}): +1 punto (total = {SCORE_FLORES[algoritmo_actual]})")
        
    except Exception as e:
        classification_text = f"Error clasificando: {e}"
        best_label = None
    
    img_original_pil = Image.fromarray(cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB))
    img_eq_pil = Image.fromarray(cv2.cvtColor(eq_hist, cv2.COLOR_GRAY2RGB))
    img_clahe_pil = Image.fromarray(cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB))
    
    # Redimensionar imágenes
    max_w, max_h = 400, 300
    for im in [img_original_pil, img_eq_pil, img_clahe_pil]:
        im.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

    photo_orig = ImageTk.PhotoImage(img_original_pil)
    photo_eq = ImageTk.PhotoImage(img_eq_pil)
    photo_clahe = ImageTk.PhotoImage(img_clahe_pil)
    
    # Mostrar las dos imágenes lado a lado
    frame = tk.Frame(win)
    frame.pack(padx=10, pady=10)

    lbl1 = tk.Label(frame, image=photo_orig, text="Original", compound="top")
    lbl1.image = photo_orig
    lbl1.pack(side="left", padx=10)

    lbl2 = tk.Label(frame, image=photo_eq, text="Ecualización Global", compound="top")
    lbl2.image = photo_eq
    lbl2.pack(side="left", padx=10)

    lbl3 = tk.Label(frame, image=photo_clahe, text="CLAHE (Adaptativa)", compound="top")
    lbl3.image = photo_clahe
    lbl3.pack(side="left", padx=10)
    
    # Mostrar el resultado del modelo debajo
    lbl_result = tk.Label(win, text=classification_text, font=("Arial", 12, "bold"), fg="blue")
    lbl_result.pack(pady=5)

    btn = tk.Button(win, text="Cerrar", command=win.destroy)
    btn.pack(pady=8)

    # Modal: bloquea hasta cerrar la ventana
    win.grab_set()
    win.focus_force()
    win.wait_window()


# DFS para encontrar un camino
def dfs(mundo, inicio, meta, n):
    stack = [(inicio, [inicio])]
    visitados = set()
    obstaculos_detectados = 0  # Contador de obstáculos

    while stack:
        (actual, path) = stack.pop()
        if actual in visitados:
            continue
        visitados.add(actual)

        if actual == meta:
            return path, obstaculos_detectados  # Retornar camino y contador

        x, y = actual
        vecinos = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        for nx, ny in vecinos:
            if 0 <= nx < n and 0 <= ny < n:
                if mundo[(nx, ny)] != "X":  # no es obstáculo
                    stack.append(((nx, ny), path + [(nx, ny)]))
                else:
                    print(f"🚧Obstáculo encontrado en {nx, ny}")
                    obstaculos_detectados += 1  # Incrementar contador
    return None, obstaculos_detectados

# BFS para encontrar un camino
def bfs(mundo, inicio, meta, n):
    cola = deque([(inicio, [inicio])])
    visitados = set()
    obstaculos_detectados = 0  # Contador de obstáculos

    while cola:
        (actual, path) = cola.popleft()
        if actual in visitados:
            continue
        visitados.add(actual)

        if actual == meta:
            return path, obstaculos_detectados  # Retornar camino y contador

        x, y = actual
        vecinos = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        for nx, ny in vecinos:
            if 0 <= nx < n and 0 <= ny < n:
                if mundo[(nx, ny)] != "X":  # si no es obstáculo
                    cola.append(((nx, ny), path + [(nx, ny)]))
                else:
                    print(f"🚧Objeto detectado en {nx, ny}")
                    obstaculos_detectados += 1  # Incrementar contador
    return None, obstaculos_detectados

# Elegir inicio y meta manualmente 
def elegir_puntos_separados(mundo, n):
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Selecciona inicio, meta y algoritmo")
    font = pygame.font.Font(None, 25)
    clock = pygame.time.Clock()

    inicio_dfs = None
    meta_dfs = None
    inicio_bfs = None
    meta_bfs = None
    
    
    # Copias del mundo para cada algoritmo
    mundo_dfs = mundo.copy()
    mundo_bfs = mundo.copy()
    
    # Botones para seleccionar algoritmo
    dfs_button = pygame.Rect(WINDOW_SIZE + 20, 150, 150, 30)
    bfs_button = pygame.Rect(WINDOW_SIZE + 680,150, 150, 30)
    
    # Algoritmo seleccionado
    algoritmo = None
    datos_elegidos = None
    
    seleccionando = True
    while seleccionando:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, None
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                
                #Vereficar clic en la cuadrícula DFS
                if x < WINDOW_SIZE and y < WINDOW_SIZE:
                    fila = y // CELL_SIZE
                    columna = x // CELL_SIZE
                    # Evitar seleccionar un obstáculo
                    if mundo_dfs[(fila, columna)] != "X":
                        if event.button == 1:  # Click izquierdo
                            if inicio_dfs:
                                mundo_dfs[inicio_dfs] = " "
                            inicio_dfs = (fila, columna)
                            mundo_dfs[inicio_dfs] = "S"
                        elif event.button == 3:  # Click derecho
                            if meta_dfs:
                                mundo_dfs[meta_dfs] = " "
                            meta_dfs = (fila, columna)
                            mundo_dfs[meta_dfs] = "G"

                
                # Verificar clic en cuadrícula BFS 
                elif WINDOW_SIZE + 400 < x < WINDOW_SIZE * 2 + 400 and y < WINDOW_SIZE:
                    fila = y // CELL_SIZE
                    columna = (x - WINDOW_SIZE - 400) // CELL_SIZE
                    # Evitar seleccionar un obstáculo
                    if mundo_bfs[(fila, columna)] != "X":
                        if event.button == 1:  # Click izquierdo
                            if inicio_bfs:
                                mundo_bfs[inicio_bfs] = " "
                            inicio_bfs = (fila, columna)
                            mundo_bfs[inicio_bfs] = "S"
                        elif event.button == 3:  # Click derecho
                            if meta_bfs:
                                mundo_bfs[meta_bfs] = " "
                            meta_bfs = (fila, columna)
                            mundo_bfs[meta_bfs] = "G"

                        
                # Verificar clic en botones
                if dfs_button.collidepoint(x, y) and inicio_dfs and meta_dfs:
                    algoritmo_elegido = "DFS"
                    datos_elegidos = (inicio_dfs, meta_dfs, mundo_dfs)
                    seleccionando = False
                    
                if bfs_button.collidepoint(x, y) and inicio_bfs and meta_bfs:
                    algoritmo_elegido = "BFS"
                    datos_elegidos = (inicio_bfs, meta_bfs, mundo_bfs)
                    seleccionando = False

        # Dibujar el mundo
        screen.fill(WHITE)
        
        # CUADRÍCULA DFS (IZQUIERDA) 
        for i in range(n):
            for j in range(n):
                valor = mundo_dfs[(i, j)]
                if valor == "X":
                    color = BLACK
                elif valor == "S":
                    color = GREEN
                elif valor == "G":
                    color = RED
                else:
                    color = WHITE

                rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, BLUE, rect, 1)
        
        # Panel información para DFS
        panel_dfs_x = WINDOW_SIZE + 20
        
        # Título
        titulo_dfs = font.render("Algoritmo DFS ", True, BLACK)
        screen.blit(titulo_dfs, (panel_dfs_x, 20))
        
        # Información de inicio y meta
        if inicio_dfs:
            inicio_dfs_text = font.render(f"Inicio: {inicio_dfs}", True, GREEN)
            screen.blit(inicio_dfs_text, (panel_dfs_x, 60))
        else:
            inicio_dfs_text = font.render("Inicio: No seleccionado", True, BLUE)
            screen.blit(inicio_dfs_text, (panel_dfs_x, 60))
            
        if meta_dfs:
            meta_dfs_text = font.render(f"Meta: {meta_dfs}", True, RED)
            screen.blit(meta_dfs_text, (panel_dfs_x, 100))
        else:
            meta_dfs_text = font.render("Meta: No seleccionado", True, BLUE)
            screen.blit(meta_dfs_text, (panel_dfs_x, 100))
        
        # Botón DFS
        dfs_color = (0, 200, 0) if inicio_dfs and meta_dfs else (100, 100, 100)
        pygame.draw.rect(screen, dfs_color, dfs_button)
        dfs_text = font.render("Ejecutar DFS", True, WHITE)
        
        # Obtener el rectángulo del texto y centrarlo en el botón
        dfs_text_rect = dfs_text.get_rect(center=dfs_button.center)

        # Dibujar el texto centrado
        screen.blit(dfs_text, dfs_text_rect)
        
        
        # Instrucciones DFS
        instrucciones = [
            "Instrucciones:",
            "- Click izquierdo: Inicio (S)",
            "- Click derecho: Meta (G)", 
            "- Enter para iniciar"
        ]
        
        for i, linea in enumerate(instrucciones):
            texto = font.render(linea, True, BLACK)
            screen.blit(texto, (panel_dfs_x, 200 + i * 30))
        
        # CUADRÍCULA BFS (DERECHA)
        espacio_x= WINDOW_SIZE + 400
        
        # Dibujar mundo BFS
        for i in range(n):
            for j in range(n):
                valor = mundo_bfs[(i, j)]
                if valor == "X":
                    color = BLACK
                elif valor == "S":
                    color = GREEN
                elif valor == "G":
                    color = RED
                else:
                    color = WHITE

                rect = pygame.Rect(espacio_x + j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, BLUE, rect, 1)
        
        # Panel información BFS
        panel_bfs_x = espacio_x + WINDOW_SIZE + 20
        
        titulo_bfs = font.render("ALGORITMO BFS", True, (0, 0, 200))
        screen.blit(titulo_bfs, (panel_bfs_x, 20))
        
        if inicio_bfs:
            inicio_bfs_text = font.render(f"Inicio: {inicio_bfs}", True, GREEN)
            screen.blit(inicio_bfs_text, (panel_bfs_x, 60))
        else:
            inicio_bfs_text = font.render("Inicio: No seleccionado", True, BLUE)
            screen.blit(inicio_bfs_text, (panel_bfs_x, 60))
            
        if meta_bfs:
            meta_bfs_text = font.render(f"Meta: {meta_bfs}", True, RED)
            screen.blit(meta_bfs_text, (panel_bfs_x, 100))
        else:
            meta_bfs_text = font.render("Meta: No seleccionado", True, BLUE)
            screen.blit(meta_bfs_text, (panel_bfs_x, 100))
        
        # Botón BFS
        bfs_color = (0, 0, 200) if inicio_bfs and meta_bfs else (100, 100, 100)
        pygame.draw.rect(screen, bfs_color, bfs_button)
        bfs_text = font.render("Ejecutar BFS", True, WHITE)
        
        bfs_text_rect = bfs_text.get_rect(center=bfs_button.center)
        screen.blit(bfs_text, bfs_text_rect)
        
        # Instrucciones BFS
        instrucciones_bfs = [
            "Instrucciones BFS:",
            "- Click izquierdo: INICIO (S)",
            "- Click derecho: META (G)", 
            "- Luego 'Ejecutar BFS'"
        ]
        
        for i, linea in enumerate(instrucciones_bfs):
            texto = font.render(linea, True, BLACK)
            screen.blit(texto, (panel_bfs_x, 200 + i * 30))
        
        pygame.display.flip()
        clock.tick(30)

    return algoritmo_elegido, datos_elegidos

# Visualizar el mundo y el movimiento del agente
def mostrar_mundo(mundo, path, inicio, meta, n, algoritmo, obstaculos_totales, tiempo_total):
    screen = pygame.display.get_surface()  # Usa la misma ventana existente
    pygame.display.set_caption(f"Abeja DFS y BFS 🐝")
    font = pygame.font.Font(None, 25)
    bee_font = pygame.font.Font(None, CELL_SIZE + 5)
    clock = pygame.time.Clock()

    # Iniciar contadores en tiempo real
    tiempo_inicio = time.time()
    obstaculos_en_tiempo_real = 0
    obstaculos_contados = set()

    # Determinar desplazamiento (izquierda = DFS, derecha = BFS)
    offset_x = 0 if algoritmo == "DFS" else WINDOW_SIZE + 400

    for paso_index, paso in enumerate(path):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Limpiar solo el área de la cuadrícula actual (sin borrar otras)
        pygame.draw.rect(screen, WHITE, (offset_x, 0, WINDOW_SIZE + 300, WINDOW_SIZE))


        # Comprobar obstáculos cercanos SOLO si no es la meta
        obstaculos_cercanos = []
        if paso != meta:
            x, y = paso
            vecinos = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            for nx, ny in vecinos:
                if 0 <= nx < n and 0 <= ny < n:
                    if mundo[(nx, ny)] == "X":
                        obstaculos_cercanos.append((nx, ny))
                        if (nx, ny) not in obstaculos_contados:
                            obstaculos_contados.add((nx, ny))
                            obstaculos_en_tiempo_real += 1
                        
                        # Mostrar ventana con la imagen asignada a ese obstáculo (pausa hasta cerrarla)
                        img_path = OBSTACLE_IMAGE_MAP.get((nx, ny))
                        if img_path:
                            mostrar_imagen_ventana(img_path)

        # Dibujar el mundo del algoritmo activo
        for i in range(n):
            for j in range(n):
                valor = mundo[(i, j)]
                if valor == "X":
                    color = BLACK
                elif (i, j) == inicio:
                    color = GREEN
                elif (i, j) == meta:
                    color = RED
                else:
                    color = WHITE
                rect = pygame.Rect(offset_x + j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, BLUE, rect, 1)

        # Dibujar la abeja en el camino
        x, y = paso
        bee_text = bee_font.render("🐝", True, YELLOW)
        screen.blit(bee_text, (offset_x + y * CELL_SIZE + CELL_SIZE // 4, x * CELL_SIZE + CELL_SIZE // 4))

        # Calcular tiempo transcurrido
        tiempo_transcurrido = time.time() - tiempo_inicio

        # Mostrar información del algoritmo y estadísticas
        panel_x = offset_x + WINDOW_SIZE + 20
        algo_text = font.render(f"Algoritmo: {algoritmo}", True, BLACK)
        screen.blit(algo_text, (panel_x, 20))

        obstaculos_text = font.render(f"Obstáculos: {obstaculos_en_tiempo_real}", True, BLACK)
        screen.blit(obstaculos_text, (panel_x, 60))
        
        score_text = font.render(f"Flores: {SCORE_FLORES[algoritmo]}", True, (128, 0, 128))
        screen.blit(score_text, (panel_x, 80))


        tiempo_text = font.render(f"Tiempo: {tiempo_transcurrido:.2f}s", True, BLACK)
        screen.blit(tiempo_text, (panel_x, 100))

        progreso_text = font.render(f"Progreso: {paso_index+1}/{len(path)}", True, BLACK)
        screen.blit(progreso_text, (panel_x, 140))

        pygame.display.flip()
        time.sleep(0.6)
        clock.tick(60)

    # Limpiar área del mensaje "Meta alcanzada" (parte inferior de la cuadrícula)
    pygame.draw.rect(screen, WHITE, (offset_x, WINDOW_SIZE, WINDOW_SIZE, 50))
    
    # Limpiar área del panel lateral (estadísticas)
    pygame.draw.rect(screen, WHITE, (offset_x + WINDOW_SIZE, 0, 300, WINDOW_HEIGHT))

    # Mostrar texto final (limpio)
    fin_text = font.render("¡Meta alcanzada!", True, RED)
    screen.blit(fin_text, (offset_x + 30, WINDOW_SIZE + 20))

    # Estadísticas finales (limpias)
    panel_x = offset_x + WINDOW_SIZE + 20
    tiempo_final = time.time() - tiempo_inicio

    algo_fin_text = font.render(f"Algoritmo: {algoritmo}", True, BLACK)
    screen.blit(algo_fin_text, (panel_x, 20))

    obstaculos_fin_text = font.render(f"Obstáculos: {obstaculos_en_tiempo_real}", True, BLACK)
    screen.blit(obstaculos_fin_text, (panel_x, 60))
    
    score_text = font.render(f"Flores: {SCORE_FLORES[algoritmo]}", True, (128, 0, 128))
    screen.blit(score_text, (panel_x, 80))

    tiempo_fin_text = font.render(f"Tiempo: {tiempo_final:.2f}s", True, BLACK)
    screen.blit(tiempo_fin_text, (panel_x, 100))

    pasos_fin_text = font.render(f"Pasos: {len(path)}", True, BLACK)
    screen.blit(pasos_fin_text, (panel_x, 140))

    pygame.display.flip()
    
    # Esperar para volver al menú
    esperando = True
    while esperando:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                esperando = False


def mostrar_comparativa(resultados_dfs, resultados_bfs):
    """Muestra una ventana comparativa entre DFS y BFS según las métricas recolectadas."""
    if not resultados_dfs or not resultados_bfs:
        return  # Solo mostrar si ambos existen

    screen = pygame.display.set_mode((600, 400))
    pygame.display.set_caption("Comparativa DFS vs BFS 🌸")

    font_title = pygame.font.Font(None, 36)
    font_text = pygame.font.Font(None, 28)
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                running = False

        screen.fill((255, 255, 255))

        title = font_title.render("Comparativa DFS vs BFS", True, (0, 0, 0))
        screen.blit(title, (150, 20))

        headers = ["Métrica", "DFS", "BFS"]
        datos = [
            ("Flores", SCORE_FLORES["DFS"], SCORE_FLORES["BFS"]),
            ("Tiempo", f"{resultados_dfs['tiempo']:.2f}s", f"{resultados_bfs['tiempo']:.2f}s"),
            ("Obstáculos", resultados_dfs['obstaculos'], resultados_bfs['obstaculos']),
            ("Longitud camino",
             len(resultados_dfs['camino']) if resultados_dfs['camino'] else "—",
             len(resultados_bfs['camino']) if resultados_bfs['camino'] else "—")
        ]

        # Dibujar encabezados
        x_positions = [100, 300, 450]
        for i, header in enumerate(headers):
            text = font_text.render(header, True, (0, 0, 200))
            screen.blit(text, (x_positions[i], 80))

        # Dibujar filas
        for row_index, (metrica, dfs_val, bfs_val) in enumerate(datos):
            y = 130 + row_index * 50
            metrica_text = font_text.render(str(metrica), True, (0, 0, 0))
            dfs_text = font_text.render(str(dfs_val), True, (0, 128, 0))
            bfs_text = font_text.render(str(bfs_val), True, (200, 0, 0))
            screen.blit(metrica_text, (100, y))
            screen.blit(dfs_text, (300, y))
            screen.blit(bfs_text, (450, y))

        nota = font_text.render("Presiona una tecla o haz clic para continuar", True, (100, 100, 100))
        screen.blit(nota, (100, 340))

        pygame.display.flip()
        clock.tick(30)

    pygame.display.quit()


# Programa principal CORREGIDO
if __name__ == "__main__":
    pygame.init()
    mundo = crear_mundo(N)
    
    
    # Cargar imágenes y asignarlas aleatoriamente a cada obstáculo
    try:
        imagenes_sub = cargar_imagenes_de_carpeta('imagenes_sub')
        imagenes_sobre = cargar_imagenes_de_carpeta('imagenes_sobre')
        
        # Unir ambas listas en una sola
        imagenes = imagenes_sub + imagenes_sobre
        
        OBSTACLE_IMAGE_MAP = asignar_imagenes_a_obstaculos(mundo, imagenes)
        print(f"Se cargaron {len(imagenes)} imágenes en total.")
        
    except Exception as e:
        print("Aviso: no se pudieron cargar imágenes:", e)
        OBSTACLE_IMAGE_MAP = {}
    
    # Variables para guardar resultados
    resultado_dfs = None
    resultado_bfs = None
    
    ejecutando = True
    while ejecutando:
        algoritmo, datos = elegir_puntos_separados(mundo, N)

        if algoritmo and datos:
            inicio, meta, mundo_elegido = datos

            inicio_tiempo = time.time()

            if algoritmo == "DFS":
                camino, obstaculos_detectados = dfs(mundo_elegido, inicio, meta, N)
                # GUARDAR RESULTADO DFS
                resultado_dfs = {
                    'camino': camino,
                    'obstaculos': obstaculos_detectados,
                    'tiempo': 0,  # Se actualizará después
                    'inicio': inicio,
                    'meta': meta,
                    'exitoso': camino is not None
                }

            elif algoritmo == "BFS":
                camino, obstaculos_detectados = bfs(mundo_elegido, inicio, meta, N)
                # GUARDAR RESULTADO BFS
                resultado_bfs = {
                    'camino': camino,
                    'obstaculos': obstaculos_detectados,
                    'tiempo': 0,  # Se actualizará después
                    'inicio': inicio,
                    'meta': meta,
                    'exitoso': camino is not None
                }

            fin_tiempo = time.time()
            tiempo_ejecucion = fin_tiempo - inicio_tiempo
            
            # Actualizar tiempo en los resultados
            if algoritmo == "DFS" and resultado_dfs:
                resultado_dfs['tiempo'] = tiempo_ejecucion
            elif algoritmo == "BFS" and resultado_bfs:
                resultado_bfs['tiempo'] = tiempo_ejecucion

            if camino:
                print(f"Ruta encontrada con {algoritmo}: {camino}")
                mostrar_mundo(mundo_elegido, camino, inicio, meta, N, algoritmo, obstaculos_detectados, tiempo_ejecucion)
            else:
                print("No se encontró ruta :(")
                print(f"Obstáculos detectados: {obstaculos_detectados}")
                print(f"Tiempo: {tiempo_ejecucion:.2f}s")
            
            # MOSTRAR RESULTADOS EN CONSOLA
            print("\n" + "="*50)
            print("RESULTADOS ACTUALES:")
            print("="*50)
            
            if resultado_dfs:
                if resultado_dfs['exitoso']:
                    print(f"✅ DFS: {len(resultado_dfs['camino'])} pasos, {resultado_dfs['obstaculos']} obstáculos, {resultado_dfs['tiempo']:.2f}s, 🌸 Score: {SCORE_FLORES['DFS']}")
                else:
                    print(f"❌ DFS: No encontró camino, {resultado_dfs['obstaculos']} obstáculos, {resultado_dfs['tiempo']:.2f}s")
            else:
                print("🔵 DFS: Aún no ejecutado")
                
            if resultado_bfs:
                if resultado_bfs['exitoso']:
                    print(f"✅ BFS: {len(resultado_bfs['camino'])} pasos, {resultado_bfs['obstaculos']} obstáculos, {resultado_bfs['tiempo']:.2f}s, 🌸 Score: {SCORE_FLORES['BFS']}")
                else:
                    print(f"❌ BFS: No encontró camino, {resultado_bfs['obstaculos']} obstáculos, {resultado_bfs['tiempo']:.2f}s")
            else:
                print("🔵 BFS: Aún no ejecutado")
            
            # Si ambos algoritmos se ejecutaron, mostrar comparativa
            if resultado_dfs and resultado_bfs:
                mostrar_comparativa(resultado_dfs, resultado_bfs)

            
            print("="*50)
            print("Volviendo al menú de selección...")
            print("="*50 + "\n")

        else:
            print("No se definieron los puntos de inicio y meta.")
            ejecutando = False

    pygame.quit()