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


# PostureBuffer tests
def test_PostureBuffer():
    """Test PostureBuffer class definition."""
    c = camera_fusion.Camera(0, 11)
    assert c.board_post.window_length == 4


def test_PostureBuffer_pop():
    """Test PostureBuffer buffer abilities."""
    rvec = np.array([[1], [0], [0]])
    tvec = np.array([[1], [0], [0]])
    c = camera_fusion.Camera(0, 11)
    frvec, ftvec = c.board_post.update(rvec, tvec)

    b_rvecs_shape = c.board_post.buff_rvecs.shape
    b_tvecs_shape = c.board_post.buff_tvecs.shape

    assert (frvec.shape, ftvec.shape, b_rvecs_shape, b_tvecs_shape) == (
        (3,), (3,), (3, 1), (3, 1))


def test_PostureBuffer_filter():
    """Test PostureBuffer filtering."""
    rvec = np.array([[0.1], [0.2], [0]])
    tvec = np.array([[0.2], [0.1], [0]])
    c = camera_fusion.Camera(0, 0)
    frvec, ftvec = c.board_post.update(rvec, tvec)
    frvec, ftvec = c.board_post.update(rvec * 0.1, tvec * 0.1)
    frvec, ftvec = c.board_post.update(rvec, tvec)
    # This should trigger the filter default avg_max_std=0.1 maximal limit
    frvec, ftvec = c.board_post.update(rvec * 2, tvec * 2)
    frvec, ftvec = c.board_post.update(rvec * 3, tvec * 3)

    np.testing.assert_allclose([[0.3], [0.6], [0.0]], frvec)
    np.testing.assert_allclose([[0.6], [0.3], [0.]], ftvec)


# Camera tests
def test_calibrate_camera_correction():
    """Test calibrate_camera_correction function."""
    c = camera_fusion.Camera(0, 11)
    assert os.path.isdir('./data')
    shutil.rmtree('data')
    shutil.copytree('./tests/test_Camera', 'data')
    c.calibrate_camera_correction()
    assert c.aruco_dict_num == 11
    assert c.charuco_square_length == 3.7999999999999999e-02
    assert c.charuco_marker_size == 2.9000000000000001e-02
    assert c.width == 1280
    assert c.height == 720
    np.testing.assert_allclose(
        [[1.0824122780443031e+03, 0., 6.4165850036653376e+02],
         [0., 1.0824122780443031e+03, 3.5960861017399100e+02],
         [0., 0., 1.]],
        c.camera_matrix)
    np.testing.assert_allclose(
        [[7.6732549196567842e-02, -4.1976860824194072e-02, 0., 0.,
         -1.8028155099783838e-01]], c.dist_coeffs)
    shutil.rmtree('data')


def test_detect_markers():
    """Test the detect_markers function."""
    c = camera_fusion.Camera(0, 11)
    shutil.rmtree('data')
    shutil.copytree('./tests/test_Camera', 'data')
    c.calibrate_camera_correction()
    real_captured_frame = np.load('./data/real_captured_frame.npy')
    with mock.patch('camera_fusion.Camera.read',
                    return_value=real_captured_frame):
        frame, corners, ids = c.detect_markers()
    np.testing.assert_array_equal(frame, real_captured_frame)

    correct_corners = np.array([
        [[[1112., 506.], [1111., 374.], [1245., 368.], [1245., 500.]]],
        [[[22., 194.], [11., 57.], [144., 51.], [158., 189.]]],
        [[[744., 164.], [739., 23.], [878., 17.], [879., 157.]]],
        [[[243., 715.], [236., 585.], [366., 580.], [373., 708.]]],
        [[[591., 699.], [584., 570.], [714., 565.], [720., 694.]]],
        [[[940.,  688.], [934., 558.], [1067., 552.], [1072., 684.]]],
        [[[57., 549.], [45., 419.], [178., 413.], [189., 543.]]],
        [[[407., 534.], [399., 405.], [529., 399.], [538., 528.]]],
        [[[757., 519.], [752., 390.], [884., 384.], [888., 514.]]],
        [[[220., 367.], [207., 234.], [341., 228.], [351., 362.]]],
        [[[573., 353.], [565., 219.], [699., 213.], [705., 347.]]],
        [[[930.,  337.], [927.,  201.], [1062., 195.], [1065., 330.]]],
        [[[383., 180.], [372., 42.], [508., 34.], [517., 175.]]]])
    np.testing.assert_array_equal(corners, correct_corners)

    correct_ids = np.array(
        [[15], [1], [11], [2], [7], [12], [0], [5], [10], [3], [8], [13], [6]])
    np.testing.assert_array_equal(ids, correct_ids)
    shutil.rmtree('data')


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


def test_estimate_markers_posture():
    """Test the estimate_markers_posture function."""
    c = camera_fusion.Camera(0, 11)
    shutil.rmtree('data')
    shutil.copytree('./tests/test_Camera', 'data')
    c.calibrate_camera_correction()
    real_captured_frame = np.load('./data/real_captured_frame.npy')
    with mock.patch('camera_fusion.Camera.read',
                    return_value=real_captured_frame):
        frame = c.estimate_markers_posture()
    correct_markers_posture_frame = np.load(
        './data/correct_markers_posture_frame.npy')
    np.testing.assert_array_equal(frame, correct_markers_posture_frame)
    shutil.rmtree('data')


def test_estimate_board_posture():
    """Test the estimate_board_posture function."""
    c = camera_fusion.Camera(0, 11)
    shutil.rmtree('data')
    shutil.copytree('./tests/test_Camera', 'data')
    c.calibrate_camera_correction()
    real_captured_frame = np.load('./data/real_captured_frame.npy')
    with mock.patch('camera_fusion.Camera.read',
                    return_value=real_captured_frame):
        frame = c.estimate_board_posture()
    correct_board_posture_frame = np.load(
        './data/correct_board_posture_frame.npy')
    np.testing.assert_array_equal(frame, correct_board_posture_frame)
    shutil.rmtree('data')


def test_estimate_board_and_markers_posture():
    """Test the estimate_estimate_board_and_markers_posture function."""
    c = camera_fusion.Camera(0, 11)
    shutil.rmtree('data')
    shutil.copytree('./tests/test_Camera', 'data')
    c.calibrate_camera_correction()
    real_captured_frame = np.load('./data/real_captured_frame.npy')
    with mock.patch('camera_fusion.Camera.read',
                    return_value=real_captured_frame):
        frame = c.estimate_board_and_markers_posture()
    np.save('./tests/test_Camera/correct_board_and_markers_posture_frame.npy',
            frame)
    np.save('./data/correct_board_and_markers_posture_frame.npy',
            frame)
    correct_board_and_markers_posture_frame = np.load(
        './data/correct_board_and_markers_posture_frame.npy')
    np.testing.assert_array_equal(
        frame, correct_board_and_markers_posture_frame)
    shutil.rmtree('data')


def test_initialize():
    """Test Camera's initialize function."""
    c = camera_fusion.Camera(0, 11)
    c.settings = [(0, 0), (1, 1), (3, 1280), (4, 720)]
    c.current_frame = np.load('./tests/test_Camera/real_captured_frame.npy')
    with mock.patch('cv2.VideoCapture', return_value=Vc(c)):
        with mock.patch('camera_fusion.Camera.calibrate_camera_correction'):
            c.initialize()


def test_read_undistort():  # monkeypatch
    """Test the read_undistort function."""
    c = camera_fusion.Camera(0, 11)
    shutil.rmtree('data')
    shutil.copytree('./tests/test_Camera', 'data')
    c.calibrate_camera_correction()

    with mock.patch('camera_fusion.Camera.read',
                    return_value=np.load('./data/real_captured_frame.npy')):
        frame_undistored = c.read_undistort()
    valid_frame_undistored = np.load('./data/real_undistored_frame.npy')
    np.testing.assert_array_equal(valid_frame_undistored, frame_undistored)
    shutil.rmtree('data')


def test_test_camera():  # monkeypatch
    """Test the basic camera test."""
    c = camera_fusion.Camera(0, 11)
    shutil.rmtree('data')
    shutil.copytree('./tests/test_Camera', 'data')
    c.calibrate_camera_correction()
    # Testing camera setup
    with mock.patch('camera_fusion.Camera.read',
                    return_value=np.load('./data/real_captured_frame.npy')):
        c.test_camera()
    shutil.rmtree('data')


def test__update_frame():
    """Test the _update_frame function."""
    c = camera_fusion.Camera(0, 11)
    c.stop = False
    shutil.rmtree('data')
    shutil.copytree('./tests/test_Camera', 'data')
    c.calibrate_camera_correction()
    # Testing camera frame read and update
    real_captured_frame = np.load('./data/real_captured_frame.npy')
    c.cap = Vc(c, real_captured_frame)
    c._update_frame()
    np.testing.assert_array_equal(c.current_frame, real_captured_frame)


def test_write_defaultConfig():  # monkeypatch  # fake_input_float):
    """Test write_defaultConfig function."""
    c = camera_fusion.Camera(0, 11)
    c.width = 1280
    c.height = 720
    with mock.patch('builtins.input', return_value=0.03):
        c.write_defaultConfig()
    assert os.path.isfile('./data/defaultConfig.xml')
    assert filecmp.cmp(
        './data/defaultConfig.xml',
        './tests/test_Camera/defaultConfig_assert.xml')
    shutil.rmtree('data')
