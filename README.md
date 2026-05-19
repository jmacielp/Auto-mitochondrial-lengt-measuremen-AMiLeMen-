# AMiLeMen — Auto Mitochondrial Length Measurement

> Automatic detection and measurement of mitochondria in Zeiss `.czi` confocal images.

Drop one or more `.czi` files into a folder — the program finds every mitochondrion, measures it, and exports the results. No Fiji macros, no manual clicking through hundreds of images.

I built this on top of my own hand-drawn ROI measurements, but the goal is to make it collaborative: a shared space where researchers can pool measurements, refine the model together, and make this kind of analysis more open and reproducible.

---
## How it works
```
.czi → pick sharpest frame → detect candidates → filter by learned profile
     → overlay for manual correction → profile refines with each correction
```

Green lines = automatic detections. Red lines = your reference ROIs.
You remove false positives, add missed ones — every correction makes the next run better.

---
## Modes

**Training** — teach the program what a mitochondrion looks like:
- A folder of `.czi` files
- A folder of `.zip` files (Fiji ROI Manager exports) with hand-drawn line ROIs

**Prediction** — just measure:
- A folder of `.czi` files
- A trained profile (`perfil_mitocondrias.json`)

Enable **Manual Review** in prediction mode to correct detections image by image and actively refine the profile as you go.

---

## Output

Each run creates a timestamped subfolder — nothing ever gets overwritten:

```
Results/
└── 2026-05-19_14-30-00_prediction/
    ├── overlays/                  # RGB images with lines drawn on top
    ├── sharp_frames/              # sharpest frame from each CZI
    ├── detected_rois/             # detections in Fiji ROI format
    └── automatic_measurements.csv
```

---

## Requirements

```bash
pip install numpy scipy scikit-image tifffile pillow czifile matplotlib
```

Python 3.9+. Launch by double-clicking `ejecutar.bat`, or:

```bash
python mito_analyzer.py
```

---

## Contributing

If you work with mitochondrial imaging and want to contribute your own measurements to help improve the shared profile, you're very welcome. Open an issue, start a discussion, or submit a pull request.

---

*Not a professional programmer — just someone who finds this genuinely fun (even when it's frustrating). Be kind. 😊*
