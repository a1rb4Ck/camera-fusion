camera_fusion
==========

Multiple cameras correction calibration and fusion with OpenCV Python.
This package use ChAruco board to achieve accurate multi-cameras fusion.

Installation
----------

```bash
pip install camera_fusion
```
Depending of your environment, you could have to compile OpenCV from source with Python bindings.


Quickstart
----------

## Calibration
Print a [ChAruco board](https://www.uco.es/investiga/grupos/ava/node/26), for example the one in the *./resources* folder.

Measure the length of the length of the Aruco marker and the length of the black square and start example scripts from the *./scripts* folder.

## Usage examples

Generate the lens correction calibration file for a specific camera.

```bash
python3 ./scripts/camera_calibration.py
```

Generate homographies between many camera to fuse/blend on a specific plane. If no lens correction calibration exist for the cameras, they will be generate.

```bash
python3 ./scripts/camera_calibration.py
```

Actually, simple blending method are implemented:
- Blue channel to RGB blending
- Gray scale to RGB blending
- Weighted blending
- Difference

Use cases
----------

This project was made to create a super low-cost degree of linear polarization imager. With three cheap repurposed webcams, we achieve decent results and frame-rate.
