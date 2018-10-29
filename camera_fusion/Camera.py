"""OpenCV Camera class setup and usage."""

from enum import Enum
from pathlib import Path
from threading import Event, Thread
import time
try:
    import cv2
    from cv2 import aruco
except ImportError:
    raise ImportError('ERROR opencv-contrib-python must be installed!')


class Camera(object):
    """Camera class.

    Attributes:
        cap (VideoCapture): OpenCV VideoCapture element.
        cam_id (string): Camera or V4L id (ex: /dev/video0 /dev/v4l_by_id/...).
        height (int): Camera frame height in pixels.
        width (int): Camera frame width in pixels.
        settings (list): List of OpenCV VideoCapture (v4l) settings.
        thread_ready (Event): Thread is ready Event.
        thread (threading.Thread): VideoCapture reading thread.
        t0 (time.time): Time counter buffer.

    """

    def __init__(self, cam_id, vertical_flip=None, settings=None):
        """Initialize the Camera object variables.

        Args:
            cam_id (string): Camera or V4L id.
            vertical_flip (bool): Trigger vertical frame flipping.
            settings (list): list of tuple with specific camera settings.
        """
        # Resolve cam_id v4l path
        if isinstance(cam_id, int):
            # self.cam_id = '/dev/video' + str(cam_id)
            self.cam_id = str(cam_id)
        elif 'v4l' in cam_id:
            self.cam_id = Path(cam_id).resolve()
            # self.cam_id = int(str(cam_id).replace('/dev/video', ''))
            print('  Found a v4l camera path, resolved to: %s'
                  ', cam_id: %s' % (cam_id, self.cam_id))
        else:
            self.cam_id = cam_id

        if vertical_flip is True:
            print('Set vertical flip.')
            self.vertical_flip = True
        else:
            self.vertical_flip = False

        self.settings = settings
        self.t0 = time.time()

        # VideoCapture reading Thread
        self.thread_ready = Event()
        self.thread = Thread(target=self._update_frame, args=())

    def initialize(self):
        """Initialize the camera Thread."""
        self._setup()

        # Start the VideoCapture read() thread
        self.stop = False
        self.start_camera_thread()
        self.thread_ready.wait()

        # Quick test
        self.test_camera()
        print('Camera %s initialization done!\n' % self.cam_id)

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

    def read(self):
        """Read the current camera frame.

        Return:
            frame (OpenCV Mat): A frame read from the VideoCapture method.

        """
        return self.current_frame

    def release(self):
        """Release the VideoCapture object."""
        self.stop = True
        time.sleep(0.1)  # 0.05
        self.cap.release()

    def set_camera_settings(self):
        """Set all the camera settings."""
        if self.settings:
            print('Camera settings:')
            for setting in self.settings:
                self.cap.set(setting[0], setting[1])
            for setting in self.settings:
                print('  %s: %d' % (
                      CV_CAP_PROP(setting[0]).name, self.cap.get(setting[0])))

    def _setup(self):
        """Set up the camera."""
        self.cap = cv2.VideoCapture(self.cam_id, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise ValueError('Camera', self.cam_id, 'not found!')

        self.set_camera_settings()

        # Current camera recording frame size
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # create thread for capturing images
    def start_camera_thread(self):
        """Start the Camera frame update Thread."""
        self.thread.start()
        self.thread_ready.wait()  # block until thread created a current_frame

    def test_camera(self):
        """Basic camera test."""
        # Testing camera setup
        test_frame = self.read()
        # Simple self-test
        if test_frame.shape[0] != self.height:
            print('WARNING: Camera height is different from the setted one!\n'
                  'Check the defaultConfig.xml camera_resolution.')
        if test_frame.shape[1] != self.width:
            raise ValueError('Camera width is different from the setted one!\n'
                             'Check the defaultConfig.xml camera_resolution.')

    def _update_frame(self):
        """Read VideoCapture to update Camera current frame."""
        while(True):
            if self.stop:
                break
            ret, frame = self.cap.read()
            if ret is False:
                print('Cam %s | Error reading frame!' % self.cam_id)
            if self.vertical_flip:
                frame = cv2.flip(frame, -1)
            self.current_frame = frame
            self.thread_ready.set()


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
