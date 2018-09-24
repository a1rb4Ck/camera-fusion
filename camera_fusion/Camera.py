"""OpenCV Camera class for lens correction with Charuco calibration."""

from enum import Enum
from pathlib import Path
import numpy as np
from threading import Thread
import time
import subprocess
import os
import sys
try:
    import cv2
    from cv2 import aruco
except ImportError:
    raise ImportError('ERROR opencv-contrib-python must be installed!')

# TODO: implement height transform correction
# https://github.com/O-C-R/maproom-robots/tree/master/skycam

# TODO: AR example
# https://github.com/avmeer/ComputerVisionAugmentedReality
# Averaging
# ○ ArUco tags are hard to pick out perfectly each time
# ○ Position of the marker is noisy and subsequently the models would shake
# ○ Averaging the last three position matrices helped to stabilize the models.


def input_float(prompt=''):
    """Ask for a human float input.

    Args:
        prompt (string): Text to prompt as input.
    """
    # try:
    #     return raw_input(prompt)
    # except NameError:
    #     return input(prompt)

    while True:
        try:
            float_input = float(input(prompt))
        except ValueError:
            print('Please enter a float.\n')
            continue
        else:
            break
    return float_input


class CV_CAP_PROP(Enum):
    """CV_CAP_PROP enumeration."""

    # https://github.com/opencv/opencv/blob/master/modules/videoio/include/opencv2/videoio/videoio_c.h#L160
    # modes of the controlling registers (can be: auto, manual,
    # auto single push, absolute Latter allowed with any other mode)
    # every feature can have only one mode turned on at a time
    CV_CAP_PROP_POS_MSEC = 0
    CV_CAP_PROP_POS_FRAMES = 1
    CV_CAP_PROP_POS_AVI_RATIO = 2
    CV_CAP_PROP_FRAME_WIDTH = 3
    CV_CAP_PROP_FRAME_HEIGHT = 4
    CV_CAP_PROP_FPS = 5
    CV_CAP_PROP_FOURCC = 6
    CV_CAP_PROP_FRAME_COUNT = 7
    CV_CAP_PROP_FORMAT = 8
    CV_CAP_PROP_MODE = 9
    CV_CAP_PROP_BRIGHTNESS = 10
    CV_CAP_PROP_CONTRAST = 11
    CV_CAP_PROP_SATURATION = 12
    CV_CAP_PROP_HUE = 13
    CV_CAP_PROP_GAIN = 14
    CV_CAP_PROP_EXPOSURE = 15
    CV_CAP_PROP_CONVERT_RGB = 16
    CV_CAP_PROP_WHITE_BALANCE_BLUE_U = 17
    CV_CAP_PROP_RECTIFICATION = 18
    CV_CAP_PROP_MONOCHROME = 19
    CV_CAP_PROP_SHARPNESS = 20
    CV_CAP_PROP_AUTO_EXPOSURE = 21
    # exposure control done by camera,
    # user can adjust reference level
    # using this feature
    CV_CAP_PROP_GAMMA = 22
    CV_CAP_PROP_TEMPERATURE = 23
    CV_CAP_PROP_TRIGGER = 24
    CV_CAP_PROP_TRIGGER_DELAY = 25
    CV_CAP_PROP_WHITE_BALANCE_RED_V = 26
    CV_CAP_PROP_ZOOM = 27
    CV_CAP_PROP_FOCUS = 28
    CV_CAP_PROP_GUID = 29
    CV_CAP_PROP_ISO_SPEED = 30
    CV_CAP_PROP_MAX_DC1394 = 31
    CV_CAP_PROP_BACKLIGHT = 32
    CV_CAP_PROP_PAN = 33
    CV_CAP_PROP_TILT = 34
    CV_CAP_PROP_ROLL = 35
    CV_CAP_PROP_IRIS = 36
    CV_CAP_PROP_SETTINGS = 37
    CV_CAP_PROP_BUFFERSIZE = 38
    CV_CAP_PROP_AUTOFOCUS = 39
    CV_CAP_PROP_SAR_NUM = 40
    CV_CAP_PROP_SAR_DEN = 41

    #  Properties of cameras available through GStreamer interface
    CV_CAP_GSTREAMER_QUEUE_LENGTH = 200  # default is 1

    #  PVAPI
    CV_CAP_PROP_PVAPI_MULTICASTIP = 300  # ip for anable multicast master mode. 0 for disable multicast  # noqa
    CV_CAP_PROP_PVAPI_FRAMESTARTTRIGGERMODE = 301  # FrameStartTriggerMode: Determines how a frame is initiated  # noqa
    CV_CAP_PROP_PVAPI_DECIMATIONHORIZONTAL = 302  # Horizontal sub-sampling of the image  # noqa
    CV_CAP_PROP_PVAPI_DECIMATIONVERTICAL = 303  # Vertical sub-sampling of the image  # noqa
    CV_CAP_PROP_PVAPI_BINNINGX = 304  # Horizontal binning factor
    CV_CAP_PROP_PVAPI_BINNINGY = 305  # Vertical binning factor
    CV_CAP_PROP_PVAPI_PIXELFORMAT = 306  # Pixel format

    # Properties for Android cameras
    CV_CAP_PROP_ANDROID_FLASH_MODE = 8001
    CV_CAP_PROP_ANDROID_FOCUS_MODE = 8002
    CV_CAP_PROP_ANDROID_WHITE_BALANCE = 8003
    CV_CAP_PROP_ANDROID_ANTIBANDING = 8004
    CV_CAP_PROP_ANDROID_FOCAL_LENGTH = 8005
    CV_CAP_PROP_ANDROID_FOCUS_DISTANCE_NEAR = 8006
    CV_CAP_PROP_ANDROID_FOCUS_DISTANCE_OPTIMAL = 8007
    CV_CAP_PROP_ANDROID_FOCUS_DISTANCE_FAR = 8008
    CV_CAP_PROP_ANDROID_EXPOSE_LOCK = 8009
    CV_CAP_PROP_ANDROID_WHITEBALANCE_LOCK = 8010

    # Properties of cameras available through AVFOUNDATION interface
    CV_CAP_PROP_IOS_DEVICE_FOCUS = 9001
    CV_CAP_PROP_IOS_DEVICE_EXPOSURE = 9002
    CV_CAP_PROP_IOS_DEVICE_FLASH = 9003
    CV_CAP_PROP_IOS_DEVICE_WHITEBALANCE = 9004
    CV_CAP_PROP_IOS_DEVICE_TORCH = 9005


class PostureBuffer(object):
    """PostureBuffer class used to setup and use camera with lens correction.

    Attributes:
        window_length (int): Moving average window size (number of frame).
        avg_max_std (float): Maximum moving average standard deviation.
        buff_tvecs (Numpy array): Buffer of rotation vecs moving avg filter.
        buff_rvecs (Numpy array): Buffer of translation vecs moving avg filter.

    """

    def __init__(self, window_length=4, avg_max_std=0.1):
        """Initialize PostureBuffer class.

        Args:
            window_length (int): Moving average window size (number of frame).
            avg_max_std (float): Maximum moving average standard deviation.
        """
        self.window_length = window_length
        self.avg_max_std = avg_max_std
        self.buff_tvecs = None  # TODO: pre-allocate array of window_length
        self.buff_rvecs = None

    def update(self, rvecs, tvecs):
        """Update the moving average posture buffer and do the filtering.

        Arguments:
            rvecs (Numpy array): Posture rotation vectors (3x1).
            tvecs (Numpy array): Posture translation vectors (3x1).

        Returns:
            rvecs (Numpy array): Filtered (averaged) posture rotation vectors.
            tvecs (Numpy array): Filtered (avg) posture translation vectors.

        """
        # Notes:
        # https://github.com/avmeer/ComputerVisionAugmentedReality
        # ○ ArUco tags are hard to pick out perfectly each time.
        # ○ Position of the marker is noisy and the models would shake.
        # ○ Averaging the last THREE position matrices helped to stabilize.

        # Appending rvec and tvec postures to buffer
        if self.buff_rvecs is None:
            self.buff_rvecs = rvecs
        else:
            self.buff_rvecs = np.append(self.buff_rvecs, rvecs, axis=1)
        if self.buff_tvecs is None:
            self.buff_tvecs = tvecs
        else:
            self.buff_tvecs = np.append(self.buff_tvecs, tvecs, axis=1)

        if self.buff_rvecs.shape[1] > self.window_length:
            self.buff_rvecs = np.delete(self.buff_rvecs, 0, 1)

        if self.buff_tvecs.shape[1] > self.window_length:
            self.buff_tvecs = np.delete(self.buff_tvecs, 0, 1)
        # TODO: optimize delete without copying? But np.array are immutable..

        # Standard deviation filtering, if the board had a to big displacement.
        stdm = self.avg_max_std  # Moving/Rolling average filter max std
        rvecs_std = np.std(self.buff_rvecs, axis=1)
        if rvecs_std[0] > stdm or rvecs_std[1] > stdm or rvecs_std[2] > stdm:
            self.buff_rvecs = rvecs
        else:
            rvecs = np.mean(self.buff_rvecs, axis=1)

        tvecs_std = np.std(self.buff_tvecs, axis=1)
        if tvecs_std[0] > stdm or tvecs_std[1] > stdm or tvecs_std[2] > stdm:
            self.buff_tvecs = tvecs
        else:
            tvecs = np.mean(self.buff_tvecs, axis=1)

        return rvecs, tvecs


class Camera(object):
    """Camera class used used to setup and use camera with lens correction.

    Attributes:
        aruco_dict_num (int): ChAruco dictionnary number used for calibr.
        board (CharucoBoard): ChAruco board object used for calibration.
        cap (VideoCapture): OpenCV VideoCapture element.
        cam_id (string): Camera or V4L id (ex: /dev/video0 /dev/v4l_by_id/...).
        charuco_marker_size (float): black square length on the printed board.
        charuco_square_length (float): Aruco marker length on the print.
        height (int): Camera frame height in pixels.
        width (int): Camera frame width in pixels.
        camera_matrix (OpenCV matrix): OpenCV camera correction matrix.
        dist_coeffs (OpenCV matrix): OpenCV distance correction coefficients.
        corners (list): List of detected corners positions as a buffer.
        ids (list): List of detected corners ids as a buffer.
        board_post (PostureBuffer): Buffer to filter the posture of the board.
        thread (threading.Thread): VideoCapture reading thread.
        settings (list): List of OpenCV VideoCapture (v4l) settings.
        t0 (time.time): Time counter buffer.

    """

    def __init__(self, cam_id, aruco_dict_num, settings=None):
        """Initialize the Camera object variables.

        Args:
            cam_id (string): Camera or V4L id.
            aruco_dict_num (int): ChAruco dictionnary number used for calibr.
            settings (list): list of tuple with specific camera settings.
        """
        self.cam_id = cam_id
        self.aruco_dict_num = aruco_dict_num
        self.settings = settings
        self.t0 = time.time()

        # Corners points and identifiers buffers
        self.corners = None
        self.ids = None

        # Moving/Rolling average posture filtering
        # TODO: Low pass filtering on translation and rotation
        self.board_post = PostureBuffer()

        # VideoCapture reading Thread
        self.thread = Thread(target=self._update_frame, args=())

        # Parameter files folder
        if not Path('./data').exists():
            os.makedirs('./data')

    def initialize(self):
        """Set up camera and launch calibration routing."""
        self.cap = cv2.VideoCapture(self.cam_id)

        if not self.cap.isOpened():
            raise ValueError('Camera', self.cam_id, 'not found!')
        if self.settings:
            print('Camera settings:')
            for setting in self.settings:
                self.cap.set(setting[0], setting[1])
            for setting in self.settings:
                print('  %s: %d' % (
                      CV_CAP_PROP(setting[0]).name, self.cap.get(setting[0])))

        # Current camera recording frame size
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Camera correction
        self.calibrate_camera_correction()

        # Start read thread
        self.stop = False
        self.start_camera_thread()

        # Quick test
        self.test_camera()
        print('Corrected camera %s initialization done!\n' % self.cam_id)

    def calibrate_camera_correction(self):
        """Calibrate the camera lens correction."""
        # Hints:
        # https://github.com/opencv/opencv/blob/master/samples/python/calibrate.py
        # https://longervision.github.io/2017/03/16/OpenCV/opencv-internal-calibration-chessboard/
        # http://www.peterklemperer.com/blog/2017/10/29/opencv-charuco-camera-calibration/
        # http://www.morethantechnical.com/2017/11/17/projector-camera-calibration-the-easy-way/
        # https://mecaruco2.readthedocs.io/en/latest/notebooks_rst/Aruco/sandbox/ludovic/aruco_calibration_rotation.html
        defaultConfig_path = Path('./data/defaultConfig.xml')
        if defaultConfig_path.exists():
            print('  Found defaultConfig.xml.\nCAUTION: be sure settings in d'
                  'efaultConfig.xml match the current hardware configuration.')
            default_config = cv2.FileStorage(
                str(defaultConfig_path), cv2.FILE_STORAGE_READ)
            self.aruco_dict_num = int(
                default_config.getNode('charuco_dict').real())
            self.charuco_square_length = default_config.getNode(
                'charuco_square_lenght').real()  # ARGH, spelling mistake!
            self.charuco_marker_size = default_config.getNode(
                'charuco_marker_size').real()
            self.width = int(default_config.getNode(
                'camera_resolution').at(0).real())
            self.height = int(default_config.getNode(
                'camera_resolution').at(1).real())
            default_config.release()
        else:
            self.write_defaultConfig()
        aruco_dict = cv2.aruco.Dictionary_get(self.aruco_dict_num)

        # Create specific camera calibration if no one already exists
        # using the opencv_interactive-calibration program.
        cameraParameters_path = Path(
            './data/cameraParameters_%s.xml' % self.cam_id)
        if not cameraParameters_path.exists():
            print('\nStarting the %s lens calibration routine.' % self.cam_id)
            subprocess.call(
                ['opencv_interactive-calibration', '-d=0.25', '-h=7', '-w=5',
                 '--sz=%f' % self.charuco_square_length, '--t=charuco',
                 '--pf=' + str(defaultConfig_path),
                 '--of=' + str(cameraParameters_path)])
        # Load the camera calibration file.
        if cameraParameters_path.exists():
            print('  Found cameraParameters_%s.xml' % self.cam_id)
            calibration_file = cv2.FileStorage(
                str(cameraParameters_path), cv2.FILE_STORAGE_READ)
            self.camera_matrix = calibration_file.getNode('cameraMatrix').mat()
            self.dist_coeffs = calibration_file.getNode('dist_coeffs').mat()
            self.width = int(calibration_file.getNode(
                'cameraResolution').at(0).real())
            self.height = int(calibration_file.getNode(
                'cameraResolution').at(1).real())

            # Specific Fish-Eye parameters
            # self.r = calibrationParams.getNode("R").mat()
            # self.new_camera_matrix = calibrationParams.getNode(
            #     "newCameraMatrix").mat()
            calibration_file.release()
        else:
            raise ValueError(
                "cameraParameters_%s.xml not found!\n\t"
                "Please finish the calibration and press 's' to save to file."
                % self.cam_id)

        self.board = cv2.aruco.CharucoBoard_create(
                5, 7, self.charuco_square_length, self.charuco_marker_size,
                aruco_dict)
        print('Camera %s calibration correction done!' % self.cam_id)

    def detect_markers(self):
        """Detect ChAruco markers.

        Returns:
            frame (OpenCV Mat): A frame read from the VideoCapture method.
            corners (Numpy array): list of corners 2D coordinates.
            ids (Numpy array): list of detected marker identifiers.

        """
        parameters = cv2.aruco.DetectorParameters_create()
        frame = self.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rej = cv2.aruco.detectMarkers(
            gray, self.board.dictionary, parameters=parameters)
        corners, ids, rej, recov = cv2.aruco.refineDetectedMarkers(
            gray, self.board, corners, ids, rej,
            cameraMatrix=self.camera_matrix, distCoeffs=self.dist_coeffs)
        return frame, corners, ids

    def draw_fps(self, frame):
        """Compute and draw fps on frame.

        Return:
            frame (OpenCV Mat): A frame read from the VideoCapture method.

        """
        frame = self.draw_text(
            frame, '%d fps' % (1.0 / (time.time() - self.t0)),
            x=self.width/35, y=self.height - self.height/20)
        self.t0 = time.time()
        return frame

    def draw_text(self, frame, text, x=None, y=None, color=(0, 255, 0),
                  thickness=1, size=0.75):
        """Draw text on frame.

        Arguments:
            frame (OpenCV Mat): A frame read from the VideoCapture method.
            text (string): The string to be written.
            x (int): Written text horizontal coordinate.
            y (int): Written text vertical coordinate.
            color (int tuple): RGB text color.
            thickness (int): Lines thickness.
            size (float): Police size.
        Return:
            frame (OpenCV Mat): new frame with the text written.

        """
        # Hint: https://stackoverflow.com/a/42694604
        if x is None:
            x = self.width/35
        if y is None:
            y = self.height/20
        return cv2.putText(
            frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size,
            color, int(thickness), lineType=cv2.LINE_AA)

    def estimate_board_posture(self, frame=None, corners=None, ids=None):
        """Estimate ChAruco board posture.

        Arguments:
            frame (OpenCV Mat): A frame read from the VideoCapture method.
            corners (Numpy array): list of corners 2D coordinates.
            ids (Numpy array): list of detected marker identifiers.

        Return:
            frame (OpenCV Mat): Frame with the board posture drawn
        """
        # If we do not already have detect markers:
        if frame is None:
            frame, corners, ids = self.detect_markers()
        if ids is None:  # No detected marker
            frame = self.draw_text(frame, 'No ChAruco marker detected !')
            # time.sleep(0.1)  # Sleep to give the time to move the panel
        else:  # if there is at least one marker detected
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Draw axis for the global board
            retval, cha_corns, cha_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, self.board,
                cameraMatrix=self.camera_matrix, distCoeffs=self.dist_coeffs)

            if retval:
                frame_with_board = cv2.aruco.drawDetectedCornersCharuco(
                    frame, cha_corns, cha_ids, (0, 255, 0))
                # Posture estimation of the global ChAruco board
                retval, rvecs, tvecs = cv2.aruco.estimatePoseCharucoBoard(
                    cha_corns, cha_ids, self.board,
                    self.camera_matrix, self.dist_coeffs)

                if retval is True:
                    rvecs, tvecs = self.board_post.update(rvecs, tvecs)

                    frame = cv2.aruco.drawAxis(
                        frame_with_board, self.camera_matrix, self.dist_coeffs,
                        rvecs, tvecs, 4 * self.charuco_square_length)
                else:
                    frame = self.draw_text(
                        frame,  'Not enough Charuco markers detected.')
            else:
                frame = self.draw_text(
                    frame, 'Not enough resolution. Board is too far.')
        return frame

    def estimate_markers_posture(self, frame=None, corners=None, ids=None):
        """Estimate ChAruco markers posture.

        Arguments:
            frame (OpenCV Mat): A frame read from the VideoCapture method.
            corners (Numpy array): list of corners 2D coordinates.
            ids (Numpy array): list of detected marker identifiers.

        Return:
            frame (OpenCV Mat): Frame with all detected markers posture drawn.

        """
        # If we do not already have detect markers:
        if frame is None:
            frame, corners, ids = self.detect_markers()

        if ids is None:  # No detected marker
            frame = self.draw_text(frame, 'No ChAruco marker detected !')
            # time.sleep(0.1)  # Sleep to give the time to move the panel
        else:  # if there is at least one marker detected
            # Draw each detected marker
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.charuco_square_length,
                self.camera_matrix, self.dist_coeffs)
            # Draw axis for each marker
            for rvec, tvec in zip(rvecs, tvecs):
                frame = cv2.aruco.drawAxis(
                    frame, self.camera_matrix, self.dist_coeffs,
                    rvec, tvec, self.charuco_square_length)
        return frame

    def estimate_board_and_markers_posture(self):
        """Estimate posture of ChAruco markers and posture of global board.

        Return:
            frame (OpenCV Mat): Frame with the board and markers postures.

        """
        frame, corners, ids = self.detect_markers()
        frame = self.estimate_markers_posture(frame, corners, ids)
        frame = self.estimate_board_posture(frame, corners, ids)
        return frame

    # def py_charuco_camera_calibration(self):
    #     """TODO: camera calibration with Python."""
    #     parameters = cv2.aruco.DetectorParameters_create()
    #     corners_list = []
    #     ids_list = []
    #     print('Move the charuco board in front of the', self.cam_id)
    #     while len(corners_list) < 50:
    #         frame = self.read()
    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         corners, ids, rej = cv2.aruco.detectMarkers(
    #             gray, dictionary=aruco_dict, parameters=parameters)
    #         corners, ids, rej, recovered = cv2.aruco.refineDetectedMarkers(
    #             gray, cv2.aruco, corners, ids, rej,
    #             cameraMatrix=self.camera_matrix, distCoeffs=self.dist_coef)
    #         if corners is None or len(corners) == 0:
    #             print('No ChAruco corner detected!')
    #             continue
    #         ret, corners, ids = cv2.aruco.interpolateCornersCharuco(
    #             corners, ids, gray, cb)
    #         corners_list.append(corners)
    #         ids_list.append(ids)
    #         time.sleep(0.1)  # Sleep to give the time to move the panel

    #     print('Enough frames for %s calibration!' % self.cam_id)
    #     # Calibrate camera
    #     ret, K, dist_coef, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    #         corners_list, ids_list, cv2.aruco, (w, h), K,
    #         dist_coef, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    #     print('camera calib mat after\n%s' % K)
    #     print('camera dist_coef %s' % dist_coef.T)
    #     print('calibration reproj err %s' % ret)

    #     distCoeffsInit = np.zeros((5, 1))
    #     flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)  # noqa
    #     # flags = (cv2.CALIB_RATIONAL_MODEL)
    #     (ret, camera_matrix, distortion_coefficients0,
    #      rotation_vectors, translation_vectors,
    #      stdDeviationsIntrinsics, stdDeviationsExtrinsics,
    #      perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
    #      charucoCorners=allCorners, charucoIds=allIds, board=board,
    #      imageSize=imsize, cameraMatrix=cameraMatrixInit,
    #      distCoeffs=distCoeffsInit, flags=flags, criteria=(
    #         cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    def read(self):
        """Read the current camera frame.

        Return:
            frame (OpenCV Mat): A frame read from the VideoCapture method.

        """
        return self.current_frame

    # create thread for capturing images
    def start_camera_thread(self):
        """Start the Camera frame update Thread."""
        self.thread.start()
        time.sleep(0.2)

    def _update_frame(self):
        """Read VideoCapture to update Camera current frame."""
        while(True):
            if self.stop:
                break
            ret, frame = self.cap.read()
            if ret is False:
                print('%s | Error reading frame!' % self.cam_id)
            self.current_frame = frame

    def read_undistort(self):
        """Read an undistored camera frame."""
        return cv2.undistort(
            src=self.read(), cameraMatrix=self.camera_matrix,
            distCoeffs=self.dist_coeffs)

    def release(self):
        """Release the VideoCapture object."""
        self.stop = True
        time.sleep(0.1)  # 0.05
        self.cap.release()

    def set_focus(self, focus):
        """Set camera focus."""
        self.cap.set(28, focus * 0.02)
        # min: 0.0 (infinity), max: 1.0 (1cm), increment:0.02 for C525 & C920
        print(self.cam_id, '| Focus set:', self.cap.get(28))

    def test_camera(self):
        """Basic camera test."""
        # Testing camera setup
        test_frame = self.read()
        # Simple self-test
        if test_frame.shape[0] != self.height:
            print('WARNING: Camera height is different from the setted one!'
                  'Check the defaultConfig.xml camera_resolution.')
        if test_frame.shape[1] != self.width:
            raise ValueError('Camera width is different from the setted one!'
                             'Check the defaultConfig.xml camera_resolution.')

    def write_defaultConfig(self):
        """Write defaultConfig.xml with the ChAruco specific parameters."""
        print('\n')
        self.charuco_square_length = input_float(
                    'Enter the black square length in cm ')
        self.charuco_marker_size = input_float(
                    'Enter the Aruco marker length in cm: ')
        defaultConfig_path = Path('./data/defaultConfig.xml')
        file = cv2.FileStorage(
            str(defaultConfig_path), cv2.FILE_STORAGE_WRITE)
        file.write('charuco_dict', self.aruco_dict_num)
        file.write('charuco_square_lenght', self.charuco_square_length)
        # ARGH, spelling mistake in the opencv_interactive-calibration app..
        # https://github.com/opencv/opencv/blob/master/apps/interactive-calibration/parametersController.cpp#L40
        file.write('charuco_marker_size', self.charuco_marker_size)
        file.write('max_frames_num', 40)
        file.write('min_frames_num', 20)

        # To write a right <camera_resolution> element we need to update
        # OpenCV to add std::vect<int> support, see my fork and discussion:
        # https://github.com/a1rb4Ck/opencv/commit/58a9adf0dd8ed5a7f1f712e99bf0f7b1340f39a8
        # http://answers.opencv.org/question/199743/write-sequence-of-int-with-filestorage-in-python/
        #
        # Working code with the fork:
        # file.write('camera_resolution', (
        #     [self.width, self.height]))
        #
        # <camera_resolution> is an Seq of Integers. In C++ it is written by <<
        # Python bindings must be added to support seq of int as std::vect<int>

        file.release()

        # Without updating OpenCV, we seek to append <camera_resolution>
        f = open(str(defaultConfig_path), 'r+')
        ln = f.readline()
        while ln != '</opencv_storage>\n':
            ln = f.readline()
        f.seek(f.tell() - 18)
        f.write('<camera_resolution>\n  %d %d</camera_resolution>\n'
                '</opencv_storage>\n' % (self.width, self.height))
        f.close()
