AMiLeMen — Auto Mitochondrial Length Measurement
The idea is simple: drop one or more .czi files into a folder, and the program automatically detects and measures all the mitochondria in them.

I trained it using my own pre-existing measurements, but I'd love to turn this into a collaborative repository — a place where people can contribute their own measurements, discuss methods, and make this kind of analysis more open and transparent.

How it works
.czi → picks the sharpest frame → automatic detection → an overlay pops up so you can remove false positives or add missed ones by hand → those corrections keep refining the model over time.

What you need
Training mode — teach the program what a mitochondrion looks like:

A folder with .czi files
A folder with .zip files (Fiji ROI Manager exports) with your manually drawn line ROIs
Prediction mode — just measure:

A folder with .czi files
A trained profile (perfil_mitocondrias.json)
Output structure
Each run creates a timestamped subfolder so nothing gets overwritten:

Results/
└── 2026-05-19_14-30-00_prediction/
    ├── overlays/               # RGB images with detected lines drawn on top
    ├── sharp_frames/           # sharpest frame extracted from each CZI
    ├── detected_rois/          # detections exported in Fiji ROI format
    └── automatic_measurements.csv
Requirements
pip install numpy scipy scikit-image tifffile pillow czifile matplotlib
Python 3.9 or higher. Double-click ejecutar.bat to launch, or run:

python mito_analyzer.py
Note
I'm not a professional programmer — I do this because I find it genuinely fun (even when it's frustrating). Any feedback, contributions or comments are very welcome, as long as we keep it respectful. 😊

