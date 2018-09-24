"""cameras_fusion package.

Perform multi-cameras lens correction, perspective correction and fusion.
This package use ChAruco board to achieve accurate and robust calibration.

Classes:
    Camera (object): Init and read a VideoCapture camera with lens correction.
    CamerasFusions (object):  Init, calibrate and read multi-cameras fusion.
"""

from .Camera import Camera
from .CamerasFusion import CamerasFusion

__version__ = '0.0.4'
__author__ = 'Pierre Nagorny'
__email__ = 'pierre.nagorny@univ-smb.fr'
