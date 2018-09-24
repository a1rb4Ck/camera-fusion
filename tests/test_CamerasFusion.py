"""camera_fusion CamerasFusion class tests."""

import os
import sys
import numpy as np
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import camera_fusion  # noqa


# Import tests
def test_import_CamerasFusion():
    """Test CamerasFusion class importation."""
    a = camera_fusion.CamerasFusion.__module__ == 'camera_fusion.CamerasFusion'
    assert a


# CamerasFusion tests
# TODO: test CamerasFusion points matching
