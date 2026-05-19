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
from skimage.measure import profile_line
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
            # Leer escala del XML de metadata (Zeiss guarda en metros, Distance Id="Y")
            escala = 1.0
            try:
                import xml.etree.ElementTree as ET
                meta_xml = czi.metadata()
                root = ET.fromstring(meta_xml)
                for elem in root.iter():
                    tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                    if tag == 'Distance' and elem.get('Id') == 'Y':
                        for child in elem:
                            ctag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                            if ctag == 'Value' and child.text:
                                val = float(child.text)
                                if val < 1e-3:   # metros -> um
                                    escala = val * 1e6
                                break
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
            # position es el frame (1-indexado) donde se dibujo el ROI
            frame_pos = int(getattr(r, "position", 1) or 1)
            resultado.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                               "nombre": r.name, "frame": frame_pos})
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


def medir_sobre_linea(frame_2d, x1, y1, x2, y2, escala_um_px, ancho=1):
    """
    Mide a lo largo de una linea, identico a la herramienta Line de Fiji.
    Samplea los pixeles con interpolacion bilineal y calcula:
      length  = distancia euclidea en um
      mean    = media de intensidad a lo largo de la linea
      median  = mediana de intensidad
      angle   = angulo en grados (convencion Fiji: eje X positivo = 0)
      intden  = mean * length  (en um)
      rawintden = suma de todos los valores del perfil
    Devuelve None si la linea tiene longitud < 1 pixel.
    """
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    length_px = math.sqrt(dx * dx + dy * dy)
    if length_px < 1.0:
        return None

    length_um  = length_px * escala_um_px
    angle_deg  = math.degrees(math.atan2(dy, dx))

    # profile_line espera (row, col) = (y, x)
    perfil = profile_line(
        frame_2d,
        (float(y1), float(x1)),
        (float(y2), float(x2)),
        linewidth=ancho,
        order=1,
        mode="reflect",
    )

    mean_val   = float(np.mean(perfil))
    median_val = float(np.median(perfil))
    intden     = mean_val * length_um
    rawintden  = float(np.sum(perfil))

    return {
        "length":    round(length_um, 4),
        "mean":      round(mean_val, 4),
        "median":    round(median_val, 4),
        "angle":     round(angle_deg, 4),
        "intden":    round(intden, 4),
        "rawintden": round(rawintden, 4),
    }


def endpoints_eje_mayor(prop, r1_offset, c1_offset):
    """
    Calcula los endpoints del eje mayor de una region segmentada
    en coordenadas globales del frame.
    Convencion skimage: orientation = angulo del eje mayor desde eje X.
    """
    cy, cx = prop.centroid
    cy_g = cy + r1_offset
    cx_g = cx + c1_offset
    half = prop.major_axis_length / 2.0
    ori  = prop.orientation

    # Segun documentacion de skimage regionprops
    x1 = cx_g - math.sin(ori) * half
    y1 = cy_g - math.cos(ori) * half
    x2 = cx_g + math.sin(ori) * half
    y2 = cy_g + math.cos(ori) * half

    return x1, y1, x2, y2


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

FEATURES_PERFIL = {"mean", "intden", "median", "rawintden", "length"}

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


def guardar_overlay_tif(frame_2d, lineas_detectadas, rois_lineas, escala, ruta):
    """
    Guarda un TIFF RGB.
    lineas_detectadas: lista de (x1, y1, x2, y2) - lineas verdes de punta a punta.
    rois_lineas: lista de dicts {x1,y1,x2,y2,...} - lineas rojas (ROI manuales).
    """
    from PIL import Image, ImageDraw

    f32 = frame_2d.astype(np.float32)
    f32 = (f32 - f32.min()) / (f32.max() - f32.min() + 1e-9)
    img8 = (f32 * 255).astype(np.uint8)

    pil_img = Image.fromarray(np.stack([img8, img8, img8], axis=-1), mode="RGB")
    draw = ImageDraw.Draw(pil_img)

    for (x1, y1, x2, y2) in lineas_detectadas:
        draw.line([(x1, y1), (x2, y2)], fill=(0, 255, 0), width=2)

    for roi in rois_lineas:
        draw.line([(roi["x1"], roi["y1"]), (roi["x2"], roi["y2"])],
                  fill=(255, 0, 0), width=2)

    directorio = os.path.dirname(ruta)
    if directorio:
        os.makedirs(directorio, exist_ok=True)
    tifffile.imwrite(ruta, np.array(pil_img), photometric="rgb")


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
                    log_fn=print, revision=False, app_root=None):
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
    props_mito = []       # solo para guardar_roi_zip_fiji en prediccion
    lineas_detectadas = []
    rois_lineas = []

    if modo == "entrenamiento" and ruta_roi_zip:
        # --- ENTRENAMIENTO: medir sobre cada linea ROI manual (como Fiji Line tool) ---
        rois_lineas = leer_roi_zip(ruta_roi_zip)
        log_fn("  ROIs manuales (lineas): {}".format(len(rois_lineas)))

        for roi in rois_lineas:
            feat = medir_sobre_linea(frame,
                                     roi["x1"], roi["y1"],
                                     roi["x2"], roi["y2"],
                                     escala)
            if feat is None:
                continue
            feat["imagen"] = numero
            mediciones.append(feat)
            lineas_detectadas.append((roi["x1"], roi["y1"], roi["x2"], roi["y2"]))

        log_fn("  Mitocondrias identificadas: {}".format(len(mediciones)))

    else:
        # --- PREDICCION: maximos locales + eje mayor como linea ---
        from skimage.feature import peak_local_max

        umbral_min = float(np.percentile(frame, 70))
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

            key = (r1 + int(prop_p.centroid[0]),
                   c1 + int(prop_p.centroid[1]))
            if key in vistos:
                continue
            vistos.add(key)

            vx1, vy1, vx2, vy2 = endpoints_eje_mayor(prop_p, r1, c1)
            feat = medir_sobre_linea(frame, vx1, vy1, vx2, vy2, escala)
            if feat is None:
                continue

            if not perfil or es_mitocondria(feat, perfil, n_sigma):
                feat["imagen"] = numero
                mediciones.append(feat)
                props_mito.append((prop_p, r1, c1))
                lineas_detectadas.append((vx1, vy1, vx2, vy2))

        log_fn("  Mitocondrias detectadas: {}".format(len(mediciones)))

        if props_mito:
            ruta_rois = os.path.join(carpeta_salida, "rois_detectados",
                                     numero + "_mito.zip")
            guardar_roi_zip_fiji([p for p, _, _ in props_mito],
                                 escala, ruta_rois, numero)

    # --- Revision manual opcional ---
    if revision and app_root is not None and lineas_detectadas:
        ventana = VentanaRevision(app_root, frame, lineas_detectadas, rois_lineas,
                                  escala, titulo="Revision: " + nombre)
        lineas_finales = ventana.lineas_finales
        if lineas_finales is not None:
            mediciones = []
            for (x1, y1, x2, y2) in lineas_finales:
                feat = medir_sobre_linea(frame, x1, y1, x2, y2, escala)
                if feat is not None:
                    feat["imagen"] = numero
                    mediciones.append(feat)
            lineas_detectadas = lineas_finales
            log_fn("  Tras revision manual: {} lineas".format(len(mediciones)))

    # Guardar overlay con lineas de punta a punta
    ruta_overlay = os.path.join(carpeta_salida, "overlays",
                                nombre + "_overlay.tif")
    guardar_overlay_tif(frame, lineas_detectadas, rois_lineas, escala, ruta_overlay)

    return mediciones, lineas_detectadas


# ---------------------------------------------------------------------------
# Ventana de revision manual por imagen
# ---------------------------------------------------------------------------

class VentanaRevision(tk.Toplevel):
    """
    Ventana modal para revisar las lineas detectadas en una imagen.
    Verde  = lineas detectadas/entrenadas (editables).
    Rojo   = ROIs manuales de referencia (solo lectura).
    Controles:
      Rueda del raton     -> zoom centrado en el cursor
      Click izquierdo     -> seleccionar linea verde mas cercana (amarillo)
      Tecla Delete        -> borrar linea seleccionada
      Arrastrar boton der -> dibujar nueva linea (cian mientras se dibuja)
      Confirmar           -> acepta los cambios y re-mide
      Saltar              -> descarta cambios, usa lineas originales
    """

    UMBRAL_SEL_PX = 8   # pixeles de pantalla para seleccionar una linea

    def __init__(self, parent, frame_2d, lineas_verdes, lineas_rojas, escala, titulo=""):
        super().__init__(parent)
        self.title(titulo or "Revision manual")
        self.grab_set()

        self._frame    = frame_2d
        self._verdes   = list(lineas_verdes)    # [(x1,y1,x2,y2), ...]
        self._rojas    = list(lineas_rojas)     # [{"x1":..,"y1":..,"x2":..,"y2":..}, ...]
        self._escala   = escala

        self._confirmado   = False
        self._zoom         = 1.0
        self._off_x        = 0
        self._off_y        = 0
        self._seleccionada = None   # indice en self._verdes
        self._drag_start   = None
        self._drag_end     = None
        self._photo        = None   # mantener referencia para tkinter

        self._construir_ui()
        self._render()
        self.wait_window()

    # ------------------------------------------------------------------
    def _construir_ui(self):
        h, w = self._frame.shape
        cw = min(w, 960)
        ch = min(h, 720)

        ttk.Label(self,
                  text="Verde=detectadas | Rojo=ROI referencia | "
                       "Click izq: seleccionar | Supr: borrar | "
                       "Arrastrar der: nueva linea | Rueda: zoom",
                  foreground="gray").pack(padx=4, pady=(4, 0))

        self._canvas = tk.Canvas(self, width=cw, height=ch, bg="black",
                                 cursor="crosshair")
        self._canvas.pack(padx=4, pady=4)

        frame_btns = ttk.Frame(self)
        frame_btns.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Button(frame_btns, text="Confirmar", command=self._confirmar).pack(side="left", padx=4)
        ttk.Button(frame_btns, text="Saltar",    command=self._saltar   ).pack(side="left", padx=4)

        c = self._canvas
        c.bind("<MouseWheel>",       self._on_zoom)
        c.bind("<Button-1>",         self._on_click)
        c.bind("<Button-3>",         self._on_drag_start)
        c.bind("<B3-Motion>",        self._on_drag_move)
        c.bind("<ButtonRelease-3>",  self._on_drag_end)
        self.bind("<Delete>",        self._on_delete)
        self.bind("<BackSpace>",     self._on_delete)

    # ------------------------------------------------------------------
    def _img_to_canvas(self, x, y):
        return x * self._zoom + self._off_x, y * self._zoom + self._off_y

    def _canvas_to_img(self, cx, cy):
        return (cx - self._off_x) / self._zoom, (cy - self._off_y) / self._zoom

    # ------------------------------------------------------------------
    def _render(self):
        from PIL import Image, ImageDraw, ImageTk

        f32 = self._frame.astype(np.float32)
        f32 = (f32 - f32.min()) / (f32.max() - f32.min() + 1e-9)
        img8 = (f32 * 255).astype(np.uint8)
        pil = Image.fromarray(np.stack([img8, img8, img8], axis=-1), mode="RGB")
        draw = ImageDraw.Draw(pil)

        # Lineas rojas (referencia, solo lectura)
        for roi in self._rojas:
            draw.line([(roi["x1"], roi["y1"]), (roi["x2"], roi["y2"])],
                      fill=(255, 50, 50), width=2)

        # Lineas verdes (editables)
        for i, (x1, y1, x2, y2) in enumerate(self._verdes):
            color = (255, 220, 0) if i == self._seleccionada else (0, 255, 0)
            draw.line([(x1, y1), (x2, y2)], fill=color, width=2)

        # Linea que se esta dibujando (cian)
        if self._drag_start and self._drag_end:
            draw.line([self._drag_start, self._drag_end], fill=(0, 200, 255), width=2)

        # Aplicar zoom
        h, w = self._frame.shape
        nw = max(1, int(w * self._zoom))
        nh = max(1, int(h * self._zoom))
        pil = pil.resize((nw, nh), Image.NEAREST)

        self._photo = ImageTk.PhotoImage(pil)
        self._canvas.delete("all")
        self._canvas.create_image(self._off_x, self._off_y, anchor="nw",
                                  image=self._photo)

    # ------------------------------------------------------------------
    def _on_zoom(self, event):
        factor = 1.15 if event.delta > 0 else 1.0 / 1.15
        cx, cy = event.x, event.y
        self._off_x = cx - (cx - self._off_x) * factor
        self._off_y = cy - (cy - self._off_y) * factor
        self._zoom = max(0.1, min(15.0, self._zoom * factor))
        self._render()

    # ------------------------------------------------------------------
    @staticmethod
    def _dist_pt_seg(px, py, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        return math.sqrt((px - (x1 + t * dx)) ** 2 + (py - (y1 + t * dy)) ** 2)

    def _on_click(self, event):
        ix, iy = self._canvas_to_img(event.x, event.y)
        umbral = self.UMBRAL_SEL_PX / self._zoom
        mejor, mejor_d = None, umbral
        for i, (x1, y1, x2, y2) in enumerate(self._verdes):
            d = self._dist_pt_seg(ix, iy, x1, y1, x2, y2)
            if d < mejor_d:
                mejor_d, mejor = d, i
        self._seleccionada = None if mejor == self._seleccionada else mejor
        self._render()

    def _on_delete(self, event=None):
        if self._seleccionada is not None:
            del self._verdes[self._seleccionada]
            self._seleccionada = None
            self._render()

    def _on_drag_start(self, event):
        ix, iy = self._canvas_to_img(event.x, event.y)
        self._drag_start = (ix, iy)
        self._drag_end   = (ix, iy)

    def _on_drag_move(self, event):
        if self._drag_start:
            self._drag_end = self._canvas_to_img(event.x, event.y)
            self._render()

    def _on_drag_end(self, event):
        if self._drag_start and self._drag_end:
            x1, y1 = self._drag_start
            x2, y2 = self._drag_end
            if math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) > 3:
                self._verdes.append((x1, y1, x2, y2))
        self._drag_start = self._drag_end = None
        self._render()

    # ------------------------------------------------------------------
    def _confirmar(self):
        self._confirmado = True
        self.destroy()

    def _saltar(self):
        self._confirmado = False
        self.destroy()

    @property
    def lineas_finales(self):
        """Devuelve la lista de lineas tras la edicion, o None si se salto."""
        return self._verdes if self._confirmado else None


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

        self.v_revision = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_modo,
                        text="Revision manual por imagen  "
                             "(mostrar overlay interactivo para corregir y agregar lineas)",
                        variable=self.v_revision).grid(row=2, column=0, sticky="w", pady=(6, 0))

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

    def _comparar_mediciones(self, ruta_manual, ruta_auto, ruta_salida):
        """
        Compara mediciones manuales vs automaticas por imagen.
        Calcula media de cada metrica en ambos sets y la diferencia relativa (%).
        Guarda comparacion.csv con una fila por imagen y metrica.
        """
        try:
            import csv as _csv

            def leer_csv(ruta):
                filas = []
                with open(ruta, newline="", encoding="utf-8") as f:
                    for row in _csv.DictReader(f):
                        filas.append(row)
                return filas

            manual = leer_csv(ruta_manual)
            auto   = leer_csv(ruta_auto)

            metricas = [k for k in manual[0].keys()
                        if k not in ("imagen",)]

            def agrupar_por_imagen(filas):
                grupos = {}
                for f in filas:
                    img = f.get("imagen", "?")
                    grupos.setdefault(img, []).append(f)
                return grupos

            man_grupos  = agrupar_por_imagen(manual)
            auto_grupos = agrupar_por_imagen(auto)
            imagenes    = sorted(set(man_grupos) | set(auto_grupos))

            filas_comp = []
            for img in imagenes:
                man_filas  = man_grupos.get(img, [])
                auto_filas = auto_grupos.get(img, [])
                for met in metricas:
                    def media(filas, m):
                        vals = []
                        for f in filas:
                            try:
                                vals.append(float(f[m]))
                            except (KeyError, ValueError):
                                pass
                        return round(sum(vals) / len(vals), 4) if vals else None

                    v_man  = media(man_filas, met)
                    v_auto = media(auto_filas, met)
                    if v_man is not None and v_auto is not None and v_man != 0:
                        dif_pct = round((v_auto - v_man) / abs(v_man) * 100, 2)
                    else:
                        dif_pct = None

                    filas_comp.append({
                        "imagen":      img,
                        "metrica":     met,
                        "manual":      v_man,
                        "automatico":  v_auto,
                        "dif_%":       dif_pct,
                    })

            exportar_csv(filas_comp, ruta_salida)
            self._log("  Comparacion guardada ({} imagenes, {} metricas)".format(
                len(imagenes), len(metricas)))
        except Exception as e:
            self._log("  AVISO: no se pudo generar comparacion: " + str(e))

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
        revision     = self.v_revision.get()

        ruta_perfil    = os.path.join(carpeta_out, "perfil_mitocondrias.json")
        ruta_csv_manual = os.path.join(carpeta_out, "mediciones_manuales.csv")
        ruta_csv_auto   = os.path.join(carpeta_out, "mediciones_automaticas.csv")
        ruta_csv_comp   = os.path.join(carpeta_out, "comparacion.csv")
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
                    log_fn=self._log,
                    revision=revision,
                    app_root=self,
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
            exportar_csv(todas_mediciones, ruta_csv_manual)
        else:
            exportar_csv(todas_mediciones, ruta_csv_auto)

        # Comparacion automatica si existen ambos archivos
        if os.path.exists(ruta_csv_manual) and os.path.exists(ruta_csv_auto):
            self._comparar_mediciones(ruta_csv_manual, ruta_csv_auto, ruta_csv_comp)
            self._log("  comparacion.csv  -> diferencias manual vs automatico")

        self._log("\n=== ANALISIS COMPLETADO ===")
        self._log("Total mitocondrias: {}".format(len(todas_mediciones)))
        self._log("Resultados en: " + carpeta_out)
        self._log("\nOutputs:")
        self._log("  frames_nitidos/       -> TIFF de cada frame seleccionado")
        self._log("  overlays/             -> TIFF con mitocondrias marcadas en verde")
        self._log("  rois_detectados/      -> ZIP de ROIs para abrir en Fiji")
        if modo == "entrenamiento":
            self._log("  mediciones_manuales.csv")
        else:
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
