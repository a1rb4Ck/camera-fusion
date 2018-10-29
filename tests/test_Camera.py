"""camera_fusion Camera class tests."""

import cv2
import os
import sys
import filecmp
import pytest
import numpy as np
import shutil
import time
import unittest.mock as mock
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import camera_fusion  # noqa


class Vc(object):
    """VideoCapture mockup."""

    def __init__(self, parent, real_captured_frame=None):
        """Initialize VideoCapture mockup.

        Args:
            parent(Camera object): parent's Camera.

        """
        self.parent = parent
        self.real_captured_frame = real_captured_frame

    def get(self, setting):
        """Mock VideoCapture's get function only to get width and height."""
        if setting == 3:
            return 1280
        if setting == 4:
            return 720
        return setting

    def isOpened(self):
        """Mock VideoCapture's isOpened function."""
        return True

    def read(self):
        """Mock VideoCapture's read function."""
        time.sleep(0.33)
        self.parent.stop = True
        print('1 frame')
        return (True, self.real_captured_frame)

    def set(self, setting0, setting1):
        print(setting0, setting1)


# Import tests
def test_import_Camera():
    """Test Camera class importation."""
    assert camera_fusion.Camera.__module__ == 'camera_fusion.Camera'


def test_draw_fps():
    """Test draw_fps function."""
    with mock.patch('time.time', return_value=0):
        c = camera_fusion.Camera(0, 11)
    c.width = 1280
    c.height = 720
    frame = np.load('./tests/test_Camera/real_captured_frame.npy')
    with mock.patch('time.time', return_value=0.03):
        frame = c.draw_fps(frame)  # 33 fps
    np.testing.assert_array_equal(np.load(
        './tests/test_Camera/real_captured_frame_with_30fps.npy'), frame)


def test_draw_text():
    """Test draw_text function."""
    c = camera_fusion.Camera(0, 11)
    c.width = 1280
    c.height = 720
    frame = np.load('./tests/test_Camera/real_captured_frame.npy')
    frame = c.draw_text(frame, 'test')  # 33 fps
    np.save('./tests/test_Camera/real_captured_frame_withText.npy',
            frame)
    np.testing.assert_array_equal(
        np.load('./tests/test_Camera/real_captured_frame_withText.npy'), frame)


def test_initialize():
    """Test Camera's initialize function."""
    c = camera_fusion.Camera(0, 11)
    c.settings = [(0, 0), (1, 1), (3, 1280), (4, 720)]
    frame = np.load('./tests/test_Camera/real_captured_frame.npy')
    c.current_frame = frame
    with mock.patch('cv2.VideoCapture', return_value=Vc(c)):
        with mock.patch('camera_fusion.Camera.read', return_value=frame):
            c.initialize()


def test_test_camera():
    """Test the basic camera test."""
    # c = camera_fusion.Camera(0, 11)
    # shutil.rmtree('data')
    # shutil.copytree('./tests/test_Camera', 'data')
    c = camera_fusion.Camera(0, 11)
    c.settings = [(0, 0), (1, 1), (3, 1280), (4, 720)]
    frame = np.load('./tests/test_Camera/real_captured_frame.npy')
    c.current_frame = frame
    with mock.patch('cv2.VideoCapture', return_value=Vc(c)):
        # Testing camera setup
        with mock.patch('camera_fusion.Camera.read',
                        return_value=np.load(
                            './data/real_captured_frame.npy')):
            c.initialize()
            c.width = 1280
            c.height = 720
            c.test_camera()
    # shutil.rmtree('data')


def test__update_frame():
    """Test the _update_frame function."""
    c = camera_fusion.Camera(0, 11)
    c.stop = False
    shutil.rmtree('data')
    shutil.copytree('./tests/test_Camera', 'data')
    # Testing camera frame read and update
    real_captured_frame = np.load('./data/real_captured_frame.npy')
    c.cap = Vc(c, real_captured_frame)
    c._update_frame()
    np.testing.assert_array_equal(c.current_frame, real_captured_frame)
    shutil.rmtree('data')
