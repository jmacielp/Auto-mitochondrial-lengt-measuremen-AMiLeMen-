# -*- coding: utf-8 -*-
"""
Analizador de Mitocondrias - Python puro
Autor: Juan Ignacio Maciel Paccini

Correr desde consola:
    python mito_analyzer.py

O doble click en mito_analyzer.bat

Modos:
  1. Prediccion  : analiza CZIs nuevos usando el perfil ya aprendido
  2. Entrenamiento: refina el perfil superponiendo CZIs con sus ROIs manuales
"""

import os
import re
import csv
import json
import math
import zipfile
import struct
import warnings
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import numpy as np
from scipy import ndimage
from skimage import filters, measure, morphology, exposure
from skimage.io import imsave
import tifffile

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Lectura de CZI  (aicsimageio)
# ---------------------------------------------------------------------------

def abrir_czi(ruta_czi):
    """
    Abre un CZI y devuelve (array_numpy, escala_um_por_pixel).
    array shape: (T, Y, X)
    escala: micrones por pixel (1.0 si no se puede leer del metadata).

    Intenta aicsimageio primero; si falla, usa czifile como fallback.
    """
    # --- Intento 1: aicsimageio (mas completo, lee metadata de escala) ---
    try:
        from aicsimageio import AICSImage
        img = AICSImage(ruta_czi)

        escala = 1.0
        try:
            ps = img.physical_pixel_sizes
            if ps.Y and ps.Y > 0:
                escala = float(ps.Y)
        except Exception:
            pass

        # Pedir TCZYX y colapsar canales/Z
        data = img.get_image_data("TCZYX")
        if data.ndim == 5:
            data = data[:, 0, 0, :, :]   # (T, Y, X)
        elif data.ndim == 4:
            data = data[:, 0, :, :]
        return data.astype(np.float32), escala

    except Exception as e1:
        pass  # intentar con czifile

    # --- Intento 2: czifile (mas simple, sin metadata de escala) ---
    try:
        import czifile
        with czifile.CziFile(ruta_czi) as czi:
            data = czi.asarray()   # shape variable segun el archivo
            # Intentar leer escala del XML de metadata
            escala = 1.0
            try:
                import xml.etree.ElementTree as ET
                meta_xml = czi.metadata()
                root = ET.fromstring(meta_xml)
                # Buscar ScalingX o ScalingY en nanometros
                for elem in root.iter():
                    if "ScalingY" in elem.tag and elem.text:
                        val = float(elem.text)
                        # Zeiss guarda en metros; convertir a um
                        if val < 1e-3:
                            escala = val * 1e6
                        break
            except Exception:
                pass

            # Colapsar todas las dimensiones menos Y, X y T
            # czifile devuelve shape tipo (1, T, C, Z, Y, X, 1)
            arr = np.squeeze(data).astype(np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis, :, :]   # (1, Y, X)
            elif arr.ndim == 3:
                # puede ser (T, Y, X) o (C, Y, X) o (Z, Y, X)
                pass  # asumimos que es T
            elif arr.ndim == 4:
                arr = arr[:, 0, :, :]         # tomar primer canal -> (T, Y, X)
            elif arr.ndim >= 5:
                # (T, C, Z, Y, X) o similar: colapsar todo menos T, Y, X
                arr = arr[:, 0, 0, :, :]

            return arr, escala

    except ImportError:
        raise ImportError(
            "No se pudo abrir el CZI. Correr instalar_dependencias.bat primero."
        )
    except Exception as e2:
        raise RuntimeError("No se pudo abrir {}\n  Error: {}".format(
            os.path.basename(ruta_czi), e2))


# ---------------------------------------------------------------------------
# Seleccion del frame mas nitido
# ---------------------------------------------------------------------------

def nitidez_frame(frame_2d):
    """Varianza del Laplaciano: mayor = mas nitido."""
    lap = ndimage.laplace(frame_2d.astype(np.float32))
    return float(lap.var())


def encontrar_frame_nitido(stack_tyx):
    """Devuelve el indice del frame mas nitido en el stack (T, Y, X)."""
    scores = [nitidez_frame(stack_tyx[t]) for t in range(len(stack_tyx))]
    return int(np.argmax(scores))


# ---------------------------------------------------------------------------
# Lectura de ROIs de Fiji (.zip con archivos .roi)
# ---------------------------------------------------------------------------

def leer_roi_zip(ruta_zip):
    """
    Lee un zip de ROIs de Fiji y devuelve lista de dicts con los endpoints
    de cada linea: {x1, y1, x2, y2, nombre}
    Usa roifile, que maneja correctamente todos los tipos y versiones de Fiji.
    """
    try:
        import roifile
    except ImportError:
        raise ImportError("Falta 'roifile'. Correr instalar_dependencias.bat")

    ROI_LINE = 3   # tipo LINE en la convencion de Fiji/roifile

    resultado = []
    try:
        todos = roifile.roiread(ruta_zip)
        for r in todos:
            if r.roitype != ROI_LINE:
                continue
            coords = r.coordinates()   # shape (2, 2): [[x1,y1],[x2,y2]]
            if coords is None or len(coords) < 2:
                continue
            x1, y1 = float(coords[0][0]), float(coords[0][1])
            x2, y2 = float(coords[1][0]), float(coords[1][1])
            resultado.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                               "nombre": r.name})
    except Exception as e:
        print("  AVISO al leer {}: {}".format(ruta_zip, e))
    return resultado


# ---------------------------------------------------------------------------
# Segmentacion de particulas
# ---------------------------------------------------------------------------

def segmentar_frame(frame_2d, min_area_px, max_area_px):
    """
    Segmenta particulas en un frame 2D usando threshold local adaptativo.
    Funciona para objetos pequeños y de intensidad moderada (ej: mitocondrias
    con MitoTracker donde el fondo no es negro sino gris bajo).
    Devuelve lista de regionprops filtradas por area.
    """
    f = frame_2d.astype(np.float32)

    # Threshold local: compara cada pixel con su vecindad
    # block_size debe ser impar y mayor que el objeto mas grande esperado
    bloque = max(min_area_px * 4 + 1, 15)
    if bloque % 2 == 0:
        bloque += 1

    # Normalizar a 0-1 para filtros de skimage
    fmin, fmax = f.min(), f.max()
    f_norm = (f - fmin) / (fmax - fmin + 1e-9)

    # Threshold local (Gaussian local mean)
    thresh_local = filters.threshold_local(f_norm, block_size=bloque, method="gaussian")
    binaria = f_norm > thresh_local

    # Remover ruido y hoyos
    binaria = morphology.remove_small_holes(binaria, area_threshold=max(4, min_area_px))
    binaria = morphology.remove_small_objects(binaria, min_size=max(2, min_area_px // 3))

    etiquetas = measure.label(binaria)
    props = measure.regionprops(etiquetas, intensity_image=frame_2d)
    return [p for p in props if min_area_px <= p.area <= max_area_px]


def extraer_particula_en_roi(frame_2d, roi_linea, radio=35):
    """
    Extrae directamente la particula alrededor de una linea ROI.
    Estrategia: tomar un parche centrado en la linea y encontrar
    el componente conectado mas brillante dentro del parche.
    Devuelve un regionprop o None.
    """
    cx = int(round((roi_linea["x1"] + roi_linea["x2"]) / 2))
    cy = int(round((roi_linea["y1"] + roi_linea["y2"]) / 2))

    H, W = frame_2d.shape
    r1 = max(0, cy - radio)
    r2 = min(H, cy + radio)
    c1 = max(0, cx - radio)
    c2 = min(W, cx + radio)

    if r2 <= r1 or c2 <= c1:
        return None

    parche = frame_2d[r1:r2, c1:c2].astype(np.float32)

    # Threshold local dentro del parche: percentil 60 del parche
    thresh = np.percentile(parche, 60)
    bin_parche = parche > thresh

    bin_parche = morphology.remove_small_holes(bin_parche, area_threshold=4)

    etiq = measure.label(bin_parche)
    if etiq.max() == 0:
        return None

    props_parche = measure.regionprops(etiq, intensity_image=parche)

    # Punto central dentro del parche
    py_local = cy - r1
    px_local = cx - c1
    py_local = min(py_local, parche.shape[0] - 1)
    px_local = min(px_local, parche.shape[1] - 1)

    # Buscar la region que contiene el punto central, o la mas cercana
    candidata = None
    min_dist = float("inf")
    for p in props_parche:
        cr, cc = p.centroid
        dist = (cr - py_local) ** 2 + (cc - px_local) ** 2
        # Preferir la region que contiene el punto
        mr, mc, Mr, Mc = p.bbox
        if mr <= py_local < Mr and mc <= px_local < Mc:
            if p.image[py_local - mr, px_local - mc]:
                candidata = p
                break
        if dist < min_dist:
            min_dist = dist
            candidata = p

    if candidata is None:
        return None

    # Crear un regionprop "virtual" con coordenadas globales para features_region
    # Usamos la intensidad del parche original en el area segmentada
    return candidata, parche, (r1, c1)


# ---------------------------------------------------------------------------
# Conversion de areas pixel <-> um2
# ---------------------------------------------------------------------------

def px2_a_um2(area_px, escala_um_px):
    return area_px * (escala_um_px ** 2)

def um2_a_px2(area_um2, escala_um_px):
    return area_um2 / (escala_um_px ** 2) if escala_um_px > 0 else area_um2


# ---------------------------------------------------------------------------
# Extraccion de features de una region
# ---------------------------------------------------------------------------

def features_region(prop, escala_um_px, frame_completo=None, offset=None):
    """
    Extrae features en unidades del perfil (um / um2).
    Si se pasan frame_completo y offset=(r1,c1) es porque prop viene de un parche;
    en ese caso la intensidad ya esta en prop.mean_intensity del parche.
    """
    area_um2 = px2_a_um2(prop.area, escala_um_px)
    feret_um = prop.major_axis_length * escala_um_px

    mean_val = float(prop.mean_intensity)

    # Mediana: calcular sobre los pixels del parche/region
    try:
        pixels = prop.intensity_image[prop.image]
        median_val = float(np.median(pixels)) if pixels.size > 0 else mean_val
    except Exception:
        median_val = mean_val

    intden    = mean_val * area_um2
    rawintden = mean_val * prop.area

    return {
        "area":      round(area_um2, 4),
        "mean":      round(mean_val, 4),
        "intden":    round(intden, 4),
        "median":    round(median_val, 4),
        "rawintden": round(rawintden, 4),
        "length":    round(feret_um, 4),
    }


# ---------------------------------------------------------------------------
# Matching: linea ROI <-> particula segmentada
# ---------------------------------------------------------------------------

def puntos_linea(x1, y1, x2, y2, pasos=60):
    """Genera 'pasos' puntos equidistantes sobre el segmento."""
    xs = np.linspace(x1, x2, pasos).astype(int)
    ys = np.linspace(y1, y2, pasos).astype(int)
    return list(zip(xs, ys))


def linea_cruza_region(roi_linea, prop, shape):
    """True si algun punto de la linea cae dentro del bounding box + mascara."""
    pts = puntos_linea(roi_linea["x1"], roi_linea["y1"],
                       roi_linea["x2"], roi_linea["y2"])
    min_r, min_c, max_r, max_c = prop.bbox  # (row_min, col_min, row_max, col_max)
    mascara = prop.image  # mascara booleana del bounding box

    for x, y in pts:
        # x = columna, y = fila
        if min_r <= y < max_r and min_c <= x < max_c:
            if mascara[y - min_r, x - min_c]:
                return True
    return False


# ---------------------------------------------------------------------------
# Perfil estadistico
# ---------------------------------------------------------------------------

FEATURES_PERFIL = {"area", "mean", "intden", "median", "rawintden", "length"}

def construir_perfil(lista_features):
    """Calcula media y SD por feature morfologica (excluye metadatos como 'imagen')."""
    if not lista_features:
        return {}
    perfil = {}
    for feat in FEATURES_PERFIL:
        vals = [f[feat] for f in lista_features if feat in f]
        if len(vals) < 2:
            continue
        arr = np.array(vals, dtype=float)
        perfil[feat] = {
            "media": round(float(arr.mean()), 4),
            "sd":    round(float(arr.std()), 4),
            "n":     len(vals),
        }
    return perfil


def guardar_perfil(perfil, ruta):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w") as f:
        json.dump(perfil, f, indent=2)
    print("  Perfil guardado en: " + ruta)


def cargar_perfil(ruta):
    with open(ruta) as f:
        return json.load(f)


def es_mitocondria(feat, perfil, n_sigma):
    """True si todas las features estan dentro de n_sigma del perfil."""
    for key, stats in perfil.items():
        if key not in feat:
            continue
        sd = stats["sd"]
        if sd < 1e-9:
            continue
        if abs(feat[key] - stats["media"]) > n_sigma * sd:
            return False
    return True


# ---------------------------------------------------------------------------
# Guardar resultados
# ---------------------------------------------------------------------------

def guardar_frame_tif(frame_2d, ruta):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    # Normalizar a 16-bit para compatibilidad con Fiji
    f32 = frame_2d.astype(np.float32)
    f32 = (f32 - f32.min()) / (f32.max() - f32.min() + 1e-9)
    img16 = (f32 * 65535).astype(np.uint16)
    tifffile.imwrite(ruta, img16, photometric="minisblack")


def guardar_overlay_tif(frame_2d, props_mito_offset, rois_lineas, escala, ruta):
    """
    Guarda un TIFF RGB.
    props_mito_offset: lista de (prop, r1_offset, c1_offset)
    Verde = mitocondrias detectadas, Rojo = lineas ROI manuales.
    """
    f32 = frame_2d.astype(np.float32)
    f32 = (f32 - f32.min()) / (f32.max() - f32.min() + 1e-9)
    img8 = (f32 * 255).astype(np.uint8)
    rgb = np.stack([img8, img8, img8], axis=-1)

    for prop, r1, c1 in props_mito_offset:
        min_r, min_c, max_r, max_c = prop.bbox
        # Dibujar punto central en verde (3x3) para objetos pequenos
        cr = int(prop.centroid[0]) + r1
        cc = int(prop.centroid[1]) + c1
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                y, x = cr + dy, cc + dx
                if 0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]:
                    rgb[y, x] = [0, 255, 0]

    for roi in rois_lineas:
        pts = puntos_linea(roi["x1"], roi["y1"], roi["x2"], roi["y2"])
        for x, y in pts:
            if 0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]:
                rgb[y, x] = [255, 0, 0]

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    tifffile.imwrite(ruta, rgb, photometric="rgb")


def guardar_roi_zip_fiji(props, escala_um_px, ruta_zip, nombre_imagen):
    """
    Guarda un zip de ROIs en formato Fiji para visualizar en ImageJ.
    Cada mitocondria se guarda como un ROI de tipo RECT (bounding box).
    """
    os.makedirs(os.path.dirname(ruta_zip), exist_ok=True)

    def escribir_roi_rect(nombre, top, left, bottom, right):
        """Serializa un ROI rect en formato binario Fiji."""
        data = bytearray(64)
        data[0:4]   = b"Iout"                         # magic
        data[4:6]   = struct.pack(">H", 227)           # version
        data[6]     = 1                                # type RECT
        data[8:10]  = struct.pack(">h", top)
        data[10:12] = struct.pack(">h", left)
        data[12:14] = struct.pack(">h", bottom)
        data[14:16] = struct.pack(">h", right)
        return bytes(data)

    with zipfile.ZipFile(ruta_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, prop in enumerate(props):
            min_r, min_c, max_r, max_c = prop.bbox
            nombre = "{}-mito-{:04d}.roi".format(nombre_imagen, i + 1)
            roi_bytes = escribir_roi_rect(nombre, min_r, min_c, max_r, max_c)
            zf.writestr(nombre, roi_bytes)


def exportar_csv(filas, ruta):
    if not filas:
        return
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        writer.writeheader()
        writer.writerows(filas)
    print("  CSV guardado: " + ruta)


# ---------------------------------------------------------------------------
# Procesamiento de una imagen
# ---------------------------------------------------------------------------

def buscar_roi_zip(carpeta_roi, numero):
    candidato = os.path.join(carpeta_roi, numero + ".zip")
    return candidato if os.path.exists(candidato) else None


def extraer_numero(nombre):
    base = os.path.splitext(os.path.basename(nombre))[0]
    nums = re.findall(r"\d+", base)
    return nums[-1] if nums else None


def procesar_imagen(ruta_czi, ruta_roi_zip, carpeta_salida,
                    min_area_um2, max_area_um2, perfil, n_sigma, modo,
                    log_fn=print):
    numero = extraer_numero(ruta_czi)
    nombre = os.path.splitext(os.path.basename(ruta_czi))[0]
    log_fn("\n--- {} ---".format(nombre))

    # Abrir CZI
    try:
        stack, escala = abrir_czi(ruta_czi)
    except Exception as e:
        log_fn("  ERROR al abrir: " + str(e))
        return [], []

    log_fn("  Frames: {}  |  Escala: {:.4f} um/px".format(len(stack), escala))

    if modo == "entrenamiento":
        # En entrenamiento usar frame 1: es donde el usuario dibujo los ROIs
        t_mejor = 0
        log_fn("  Frame usado para entrenamiento: 1/{} (donde se dibujaron los ROIs)".format(len(stack)))
    else:
        t_mejor = encontrar_frame_nitido(stack)
        log_fn("  Frame mas nitido: {}/{}".format(t_mejor + 1, len(stack)))

    frame = stack[t_mejor]

    # Guardar TIFF del frame
    ruta_tif = os.path.join(carpeta_salida, "frames_nitidos",
                            nombre + "_frame{}.tif".format(t_mejor + 1))
    guardar_frame_tif(frame, ruta_tif)

    mediciones = []
    props_mito = []
    rois_lineas = []

    if modo == "entrenamiento" and ruta_roi_zip:
        # --- ENTRENAMIENTO: un parche por cada ROI manual ---
        rois_lineas = leer_roi_zip(ruta_roi_zip)
        log_fn("  ROIs manuales (lineas): {}".format(len(rois_lineas)))

        for roi in rois_lineas:
            resultado = extraer_particula_en_roi(frame, roi)
            if resultado is None:
                continue
            prop_parche, parche, (r1, c1) = resultado
            feat = features_region(prop_parche, escala)
            feat["imagen"] = numero
            mediciones.append(feat)
            props_mito.append((prop_parche, r1, c1))

        log_fn("  Mitocondrias identificadas: {}".format(len(mediciones)))

    else:
        # --- PREDICCION: maximos locales + el mismo parche que entrenamiento ---
        from skimage.feature import peak_local_max

        # Umbral minimo de intensidad: percentil 70 del frame (evita ruido de fondo)
        umbral_min = float(np.percentile(frame, 70))

        # Distancia minima entre maximos: radio del parche / 2
        # (para no detectar el mismo objeto dos veces)
        dist_min = 18

        coords = peak_local_max(
            frame,
            min_distance=dist_min,
            threshold_abs=umbral_min,
        )
        log_fn("  Maximos locales encontrados: {}".format(len(coords)))

        vistos = set()
        for (cy, cx) in coords:
            roi_seed = {"x1": float(cx), "y1": float(cy),
                        "x2": float(cx), "y2": float(cy)}
            resultado = extraer_particula_en_roi(frame, roi_seed)
            if resultado is None:
                continue
            prop_p, parche, (r1, c1) = resultado

            # Evitar medir la misma particula desde dos maximos distintos
            key = (r1 + int(prop_p.centroid[0]),
                   c1 + int(prop_p.centroid[1]))
            if key in vistos:
                continue
            vistos.add(key)

            feat = features_region(prop_p, escala)
            if not perfil or es_mitocondria(feat, perfil, n_sigma):
                feat["imagen"] = numero
                mediciones.append(feat)
                props_mito.append((prop_p, r1, c1))

        log_fn("  Mitocondrias detectadas: {}".format(len(mediciones)))

        # Guardar ROIs zip (para abrir en Fiji)
        if props_mito:
            ruta_rois = os.path.join(carpeta_salida, "rois_detectados",
                                     numero + "_mito.zip")
            guardar_roi_zip_fiji([p for p, _, _ in props_mito],
                                 escala, ruta_rois, numero)

    # Guardar overlay de verificacion visual
    ruta_overlay = os.path.join(carpeta_salida, "overlays",
                                nombre + "_overlay.tif")
    guardar_overlay_tif(frame, props_mito, rois_lineas, escala, ruta_overlay)

    return mediciones, props_mito


# ---------------------------------------------------------------------------
# Interfaz grafica (tkinter)
# ---------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Analizador de Mitocondrias")
        self.resizable(False, False)
        self._construir_ui()

    def _carpeta(self, var):
        ruta = filedialog.askdirectory()
        if ruta:
            var.set(ruta)

    def _construir_ui(self):
        pad = {"padx": 8, "pady": 4}

        # --- Carpetas ---
        frame_carpetas = ttk.LabelFrame(self, text="Carpetas", padding=8)
        frame_carpetas.grid(row=0, column=0, sticky="ew", **pad)

        base = r"C:\Users\JIMac\OneDrive\Escritorio\Analisis"
        self.v_czi      = tk.StringVar(value=os.path.join(base, "TODOSLOSCZI"))
        self.v_roi      = tk.StringVar(value=os.path.join(base, "Todos_los_roi_medidos"))
        self.v_salida   = tk.StringVar(value=os.path.join(base, "Resultados"))

        for i, (etiq, var) in enumerate([
            ("Carpeta CZI:",         self.v_czi),
            ("Carpeta ROIs (zips):", self.v_roi),
            ("Carpeta de salida:",   self.v_salida),
        ]):
            ttk.Label(frame_carpetas, text=etiq).grid(row=i, column=0, sticky="w")
            ttk.Entry(frame_carpetas, textvariable=var, width=55).grid(row=i, column=1)
            ttk.Button(frame_carpetas, text="...",
                       command=lambda v=var: self._carpeta(v)).grid(row=i, column=2)

        # --- Parametros ---
        frame_params = ttk.LabelFrame(self, text="Parametros de clasificacion", padding=8)
        frame_params.grid(row=1, column=0, sticky="ew", **pad)

        self.v_n_sigma = tk.DoubleVar(value=2.5)

        ttk.Label(frame_params, text="N sigma (tolerancia):").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame_params, textvariable=self.v_n_sigma, width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(frame_params,
                  text="2.0 = estricto   2.5 = normal   3.0 = permisivo",
                  foreground="gray").grid(row=0, column=2, sticky="w", padx=8)

        # --- Modo ---
        frame_modo = ttk.LabelFrame(self, text="Modo de operacion", padding=8)
        frame_modo.grid(row=2, column=0, sticky="ew", **pad)

        self.v_modo = tk.StringVar(value="prediccion")
        ttk.Radiobutton(frame_modo, text="Prediccion  (analizar imagenes nuevas con el perfil aprendido)",
                        variable=self.v_modo, value="prediccion").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(frame_modo, text="Entrenamiento  (refinar perfil superponiendo CZI + ROIs manuales)",
                        variable=self.v_modo, value="entrenamiento").grid(row=1, column=0, sticky="w")

        # --- Botones ---
        frame_btns = ttk.Frame(self, padding=8)
        frame_btns.grid(row=3, column=0, sticky="e")
        ttk.Button(frame_btns, text="Iniciar analisis",
                   command=self._iniciar).grid(row=0, column=0, padx=4)
        ttk.Button(frame_btns, text="Salir",
                   command=self.destroy).grid(row=0, column=1, padx=4)

        # --- Log ---
        frame_log = ttk.LabelFrame(self, text="Progreso", padding=8)
        frame_log.grid(row=4, column=0, sticky="nsew", **pad)
        self.log_txt = scrolledtext.ScrolledText(frame_log, height=14, width=80,
                                                  state="disabled", font=("Courier", 9))
        self.log_txt.pack(fill="both", expand=True)

    def _log(self, msg):
        self.log_txt.config(state="normal")
        self.log_txt.insert("end", msg + "\n")
        self.log_txt.see("end")
        self.log_txt.config(state="disabled")
        self.update_idletasks()

    def _iniciar(self):
        self.log_txt.config(state="normal")
        self.log_txt.delete("1.0", "end")
        self.log_txt.config(state="disabled")

        carpeta_czi  = self.v_czi.get()
        carpeta_roi  = self.v_roi.get()
        carpeta_out  = self.v_salida.get()
        min_area     = 0.0   # ya no se usa, prediccion usa peak_local_max
        max_area     = 9999.0
        n_sigma      = self.v_n_sigma.get()
        modo         = self.v_modo.get()

        ruta_perfil  = os.path.join(carpeta_out, "perfil_mitocondrias.json")
        ruta_csv_out = os.path.join(carpeta_out, "mediciones_automaticas.csv")
        os.makedirs(carpeta_out, exist_ok=True)

        # Cargar perfil
        perfil = {}
        if os.path.exists(ruta_perfil):
            perfil = cargar_perfil(ruta_perfil)
            self._log("Perfil cargado ({} features, n={}):".format(
                len(perfil), sum(v.get("n", 0) for v in perfil.values())))
            for k, v in sorted(perfil.items()):
                self._log("  {:12s}: {:.4f} ± {:.4f}".format(
                    k, v["media"], v["sd"]))
        elif modo == "prediccion":
            messagebox.showerror("Error",
                "No se encontro perfil_mitocondrias.json en:\n" + carpeta_out +
                "\n\nCopia el archivo ahi o usa modo Entrenamiento primero.")
            return

        # Listar CZIs
        try:
            czis = sorted([f for f in os.listdir(carpeta_czi)
                           if f.lower().endswith(".czi")])
        except Exception as e:
            messagebox.showerror("Error", "No se pudo leer la carpeta CZI:\n" + str(e))
            return

        self._log("\nCZIs encontrados: {}".format(len(czis)))
        if not czis:
            messagebox.showwarning("Aviso", "No se encontraron archivos .czi en:\n" + carpeta_czi)
            return

        todas_mediciones = []

        for archivo in czis:
            numero = extraer_numero(archivo)
            ruta_czi = os.path.join(carpeta_czi, archivo)
            ruta_zip = buscar_roi_zip(carpeta_roi, numero) if numero else None

            if modo == "entrenamiento" and not ruta_zip:
                self._log("\nSin ROI zip para {} - omitido".format(archivo))
                continue

            try:
                meds, _ = procesar_imagen(
                    ruta_czi, ruta_zip, carpeta_out,
                    min_area, max_area, perfil, n_sigma, modo,
                    log_fn=self._log
                )
                todas_mediciones.extend(meds)
            except Exception:
                self._log("  ERROR procesando {}: {}".format(archivo, traceback.format_exc()))

        # Post-procesamiento
        if modo == "entrenamiento" and todas_mediciones:
            perfil_nuevo = construir_perfil(todas_mediciones)
            guardar_perfil(perfil_nuevo, ruta_perfil)
            self._log("\n=== PERFIL ACTUALIZADO ===")
            for k, v in sorted(perfil_nuevo.items()):
                self._log("  {:12s}: {:.4f} ± {:.4f}  (n={})".format(
                    k, v["media"], v["sd"], v["n"]))

        exportar_csv(todas_mediciones, ruta_csv_out)

        self._log("\n=== ANALISIS COMPLETADO ===")
        self._log("Total mitocondrias: {}".format(len(todas_mediciones)))
        self._log("Resultados en: " + carpeta_out)
        self._log("\nOutputs:")
        self._log("  frames_nitidos/  -> TIFF de cada frame seleccionado")
        self._log("  overlays/        -> TIFF con mitocondrias marcadas en verde")
        self._log("  rois_detectados/ -> ZIP de ROIs para abrir en Fiji")
        self._log("  mediciones_automaticas.csv")

        messagebox.showinfo("Listo",
            "Analisis completado.\n{} mitocondrias procesadas.\n\nResultados en:\n{}".format(
                len(todas_mediciones), carpeta_out))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = App()
    app.mainloop()
