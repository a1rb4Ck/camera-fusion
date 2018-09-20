"""OpenCV Camera class for lens correction with Charuco calibration."""

from enum import Enum
from pathlib import Path
import numpy as np
from threading import Thread
import time
import subprocess
import sys
try:
    import cv2
except ImportError:
    print("ERROR python3-opencv must be installed!")
    exit(1)

# TODO: implement height transform
# https://github.com/O-C-R/maproom-robots/tree/master/skycam

# TODO: AR example
# https://github.com/avmeer/ComputerVisionAugmentedReality
# Averaging
# ○ ArUco tags are hard to pick out perfectly each time
# ○ Position of the marker is noisy and subsequently the models would shake
# ○ Averaging the last three position matrices helped to stabilize the models.


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


class Camera(object):
    """Camera class used for setup, advanced ChAruco correction and usage.

    Attributes:
        aruco_dict_num (int): ChAruco dictionnary number used for calibr.
        board (CharucoBoard): ChAruco board object used for calibration.
        cap (VideoCapture): OpenCV VideoCapture element.
        cam_id (string): Camera or V4L id (ex: /dev/video0 /dev/v4l_by_id/...).
        charuco_marker_size (float): black square lenght on the printed board.
        charuco_square_length (float): Aruco marker lenght on the print.
        height (int): Camera frame height in pixels.
        width (int): Camera frame width in pixels.
        camera_matrix (OpenCV matrix): OpenCV camera correction matrix.
        dist_coeffs (OpenCV matrix): OpenCV distance correction coefficients.
        corners (list): List of detected corners positions as a buffer.
        ids (list): List of detected corners ids as a buffer.
        avg_lenght (int):  Moving average window size (number of frame).
        avg_max_std (int): Maximum moving average standard deviation.
        avg_rvecs_board (list): List buffer of rotation vecs avg moving filter.
        avg_tvecs_board (list): List buffer of translation vecs avg mov filter.

    """

    def __init__(self, cam_id, aruco_dict_num, settings=None):
        """Set up camera and launch calibration routing.

        Args:
            cam_id (string): Camera or V4L id.
            aruco_dict_num (int): ChAruco dictionnary number used for calibr.
            settings (list): list of tuple with specific camera settings.
        """
        self.cam_id = cam_id
        self.aruco_dict_num = aruco_dict_num
        self.cap = cv2.VideoCapture(self.cam_id)
        if not self.cap.isOpened():
            raise ValueError(self.cam_id, 'not found!')
        if settings:
            print("Camera settings:")
            for setting in settings:
                self.cap.set(setting[0], setting[1])
            for setting in settings:
                print('  %s: %d' % (
                      CV_CAP_PROP(setting[0]).name, self.cap.get(setting[0])))

        # Current camera recording frame size
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Corners points and identifiers buffers
        self.corners = None
        self.ids = None

        # Moving/Rolling average posture filtering
        # TODO: Low pass filtering on translation and rotation
        self.avg_lenght = 4  # Moving avg number of frame window
        self.avg_max_std = 0.1  # Max moving avg standard deviation
        self.avg_rvecs_board = None
        self.avg_tvecs_board = None

        # Camera correction
        self.calibrate_camera_correction()
        # Start read thread
        self.stop = False
        self.start_camera_thread()
        # Quick test
        self.test_camera()
        print('Corrected camera %s initialization done!\n' % self.cam_id)

    def calibrate_camera_correction(self):
        """Please move the charuco board in front of the camera."""
        # Hints:
        # https://github.com/opencv/opencv/blob/master/samples/python/calibrate.py
        # https://longervision.github.io/2017/03/16/OpenCV/opencv-internal-calibration-chessboard/
        # http://www.peterklemperer.com/blog/2017/10/29/opencv-charuco-camera-calibration/
        # http://www.morethantechnical.com/2017/11/17/projector-camera-calibration-the-easy-way/
        # https://mecaruco2.readthedocs.io/en/latest/notebooks_rst/Aruco/sandbox/ludovic/aruco_calibration_rotation.html

        if Path('defaultConfig.xml').exists():
            print('  Found defaultConfig.xml\n  CAUTION: be sure settings in d'
                  'efaultConfig.xml are valid with the current configuration.')
            default_config = cv2.FileStorage(
                'defaultConfig.xml', cv2.FILE_STORAGE_READ)
            self.aruco_dict_num = int(
                default_config.getNode('charuco_dict').real())
            self.charuco_square_length = default_config.getNode(
                'charuco_square_lenght').real()
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

        # Load or create specific camera calibration:
        # using opencv_interactive-calibration
        if not Path('cameraParameters_%s.xml' % self.cam_id).exists():
            print('  Starting the %s camera calibration routine' % self.cam_id)
            subprocess.call(
                ['opencv_interactive-calibration', '-d=0.25', '-h=7', '-w=5',
                 '--sz=%f' % self.charuco_square_length, '--t=charuco',
                 '--pf=defaultConfig.xml',
                 '--of=cameraParameters_%s.xml' % self.cam_id])
        if Path('cameraParameters_%s.xml' % self.cam_id).exists():
            print('  Found cameraParameters_%s.xml' % self.cam_id)
            calibration_file = cv2.FileStorage(
                'cameraParameters_%s.xml' % self.cam_id, cv2.FILE_STORAGE_READ)
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
        print("Camera %s calibration correction done!" % self.cam_id)

    def detect_markers(self):
        """Detect ChAruco markers.

        return frame, corners, ids
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

    def draw_text(self, frame, text, x=None, y=None, color=(0, 255, 0),
                  thickness=1, size=0.75):
        """Draw text on image.

        https://stackoverflow.com/a/42694604
        """
        if x is None:
            x = self.width/35
        if y is None:
            y = self.height/20
        return cv2.putText(
            frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size,
            color, int(thickness), lineType=cv2.LINE_AA)

    def estimate_board_posture(self, frame=None, corners=None, ids=None):
        """Estimate ChAruco board posture.

        return frame with board posture drawn
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
                    rvecs, tvecs, self.avg_rvecs_board, self.avg_tvecs_board =\
                        self._update_single_posture_moving_average(
                            rvecs, tvecs, self.avg_rvecs_board,
                            self.avg_tvecs_board)

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

        return frame with markers posture drawn
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

        return frame with board and markers posture drawn
        """
        frame, corners, ids = self.detect_markers()
        frame = self.estimate_markers_posture(frame, corners, ids)
        frame = self.estimate_board_posture(frame, corners, ids)
        return frame

    def _update_single_posture_moving_average(
            self, rvecs, tvecs, avg_rvecs, avg_tvecs):
        """Update moving average posture filter.

        return rvecs, tvecs, avg_rvecs, avg_tvecs
        """
        # https://github.com/avmeer/ComputerVisionAugmentedReality
        # ○ ArUco tags are hard to pick out perfectly each time.
        # ○ Position of the marker is noisy and the models would shake.
        # ○ Averaging the last three position matrices helped to stabilize.

        if avg_rvecs is None:
            avg_rvecs = rvecs
        if avg_tvecs is None:
            avg_tvecs = tvecs

        # Moving/Rolling average filter on posture vectors rvecs and tvecs
        stdm = self.avg_max_std  # Moving/Rolling average filter max std

        avg_rvecs = np.append(avg_rvecs, rvecs, axis=1)
        if avg_rvecs.shape[1] > self.avg_lenght:
            avg_rvecs = np.delete(avg_rvecs, 0, 1)
        rvecs_std = np.std(avg_rvecs, axis=1)
        if rvecs_std[0] > stdm or rvecs_std[1] > stdm or rvecs_std[2] > stdm:
            avg_rvecs = rvecs
        else:
            rvecs = np.mean(avg_rvecs, axis=1)

        avg_tvecs = np.append(avg_tvecs, tvecs, axis=1)
        if avg_tvecs.shape[1] > self.avg_lenght:
            avg_tvecs = np.delete(avg_tvecs, 0, 1)
        tvecs_std = np.std(avg_tvecs, axis=1)
        if tvecs_std[0] > stdm or tvecs_std[1] > stdm or tvecs_std[2] > stdm:
            avg_tvecs = tvecs
        else:
            tvecs = np.mean(avg_tvecs, axis=1)
        return rvecs, tvecs, avg_rvecs, avg_tvecs

    def py_charuco_camera_calibration(self):
        """TODO: camera calibration with Python."""
        parameters = cv2.aruco.DetectorParameters_create()
        corners_list = []
        ids_list = []
        print('Move the charuco board in front of the', self.cam_id)
        while len(corners_list) < 50:
            frame = self.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rej = cv2.aruco.detectMarkers(
                gray, dictionary=aruco_dict, parameters=parameters)
            corners, ids, rej, recovered = cv2.aruco.refineDetectedMarkers(
                gray, cv2.aruco, corners, ids, rej,
                cameraMatrix=self.camera_matrix, distCoeffs=self.dist_coef)
            if corners is None or len(corners) == 0:
                print('No ChAruco corner detected!')
                continue
            ret, corners, ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, cb)
            corners_list.append(corners)
            ids_list.append(ids)
            time.sleep(0.1)  # Sleep to give the time to move the panel

        print('Enough frames for %s calibration!' % self.cam_id)
        # Calibrate camera
        ret, K, dist_coef, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            corners_list, ids_list, cv2.aruco, (w, h), K,
            dist_coef, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        print("camera calib mat after\n%s" % K)
        print("camera dist_coef %s" % dist_coef.T)
        print("calibration reproj err %s" % ret)

        distCoeffsInit = np.zeros((5, 1))
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)  # noqa
        # flags = (cv2.CALIB_RATIONAL_MODEL)
        (ret, camera_matrix, distortion_coefficients0,
         rotation_vectors, translation_vectors,
         stdDeviationsIntrinsics, stdDeviationsExtrinsics,
         perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
         charucoCorners=allCorners, charucoIds=allIds, board=board,
         imageSize=imsize, cameraMatrix=cameraMatrixInit,
         distCoeffs=distCoeffsInit, flags=flags, criteria=(
            cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    def read(self):
        """Read the current camera frame."""
        return self.current_frame

    # create thread for capturing images
    def start_camera_thread(self):
        """Start the Camera frame update Thread."""
        Thread(target=self._update_frame, args=()).start()
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
        time.sleep(0.2)
        # time.sleep(0.05)
        print('thread stopped')
        self.cap.release()
        print('VideoCapture released')

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
        while True:
            try:
                self.charuco_square_length = float(input(
                    'Enter the black square lenght in cm (default .038): '))
            except ValueError:
                print('Please enter a float.')
                continue
            else:
                break
        while True:
            try:
                self.charuco_marker_size = float(input(
                    'Enter the Aruco marker lenght in cm (default .029): '))
            except ValueError:
                print('Please enter a float.')
                continue
            else:
                break

        file = cv2.FileStorage(
            'defaultConfig.xml', cv2.FILE_STORAGE_WRITE)
        file.write('charuco_dict', self.aruco_dict_num)
        file.write('charuco_square_lenght', self.charuco_square_length)
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

        # Without updating OpenCV we use seek to append <camera_resolution>
        f = open('defaultConfig.xml', 'r+')
        ln = f.readline()
        while ln != '</opencv_storage>\n':
            ln = f.readline()
        f.seek(f.tell() - 18)
        f.write('<camera_resolution>\n  %d %d</camera_resolution>\n'
                '</opencv_storage>\n' % (self.width, self.height))
        f.close()
