"""OpenCV CamerasFusion class for multiple cameras fusion."""

from Camera import Camera

import argparse
from enum import Enum
from pathlib import Path
import numpy as np
import time
import sortednp as snp
import subprocess
import sys
try:
    import cv2
except ImportError:
    print("ERROR python3-opencv must be installed!")
    exit(1)


class CamerasFusion(object):
    """CamerasFusion class used to match and fuse many corrected Camera.

    Attributes:
        cameras (list of Camera): list of corrected Camera to be fused.
        homographies (list): Homography matrix between 1st and current cam.

    """

    def __init__(self, cameras):
        """Set up cameras fusion.

        Args:
            cameras (list of Camera): Cameras to be blend.
        """
        if len(cameras) == 0:
            raise ValueError('Cameras list is empty!')
        self.cameras = cameras
        self.board = self.cameras[0].board
        self.homographies = []
        self.fusion_calibration_is_done = False
        self.i = 15  # 3 seconds delay before frame capture.
        self.running = True

    def calibrate_fusion(self):
        """Launch calibration routing for the fusion of all Cameras."""
        ratio = 0.99
        reprojThresh = 10  # 4
        if Path('fusionParameters.npy').exists():
            print("Found fusionParameters.npy.")
            self.homographies = np.load('fusionParameters.npy')
            if len(self.homographies) + 1 != len(self.cameras):
                raise ValueError(
                    'fusionParameters.npy cameras number (=%d) is different'
                    ' from the specified cameras number (=%d!)'
                    'Remove fusionParameters.npy to compute a new one.' % (
                        len(self.homographies) + 1, len(self.cameras)))
        else:
            print('Starting the fusion calibration routine.')
            keypoints_features_list = []
            # Loop thru Cameras and assure every Camera got the same keypoints.
            while self.running:
                for camera in self.cameras:
                    camera.corners, camera.ids = self.detect_keypoints(camera)
                if sum(x is not None for x in [
                        cam.ids for cam in self.cameras]) == len(self.cameras):
                    break
            if not self.running:
                return False
            # Match features between many Cameras detected corners keypoints
            for camera in self.cameras[1:]:
                H = self.match_keypoints(
                    self.cameras[0].corners, camera.corners,
                    self.cameras[0].ids, camera.ids, ratio, reprojThresh)
                if H is None:
                    raise ValueError('In fusion calibration: Can not match at '
                                     ' least 4 keypoints between cameras.\n'
                                     'Please re-do the calibration.')
                    self.fusion_calibration_is_done = False
                    self.homographies = []
                self.homographies.append(H)
        self.fusion_calibration_is_done = True
        np.save('fusionParameters.npy', self.homographies)
        print("Fusion calibration done!")

    def detect_keypoints(self, camera):
        """Detect ChAruco corner keypoints in a corrected Camera frame."""
        # https://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
        # http://www.morethantechnical.com/2017/11/17/projector-camera-calibration-the-easy-way/
        # frame = camera.read()
        frame = camera.read_undistort()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect Aruco markers
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.board.dictionary, parameters=parameters)
        if ids is None:  # No detected marker
            frame = camera.draw_text(
                frame, '[Camera %d] No ChAruco marker detected !'
                % camera.cam_id)
        else:  # if there is at least one marker detected
            retval, cha_corns, cha_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, self.board,
                cameraMatrix=camera.camera_matrix,
                distCoeffs=camera.dist_coeffs)
            if retval:
                frame = cv2.aruco.drawDetectedCornersCharuco(
                    frame, cha_corns, cha_ids, (0, 255, 0))
                # Posture estimation of the global ChAruco board
                retval, rvecs, tvecs = cv2.aruco.estimatePoseCharucoBoard(
                    cha_corns, cha_ids, self.board,
                    camera.camera_matrix, camera.dist_coeffs)

                if retval is True:
                    frame = camera.draw_text(
                        frame, '[Camera %d] STAY STILL %ds'
                        % (camera.cam_id, self.i / 5))
                    cv2.imshow("Live camera", frame)
                    k = cv2.waitKey(200) % 256

                    if self.i < 0:
                        return cha_corns, cha_ids
                    self.i = self.i - 1
                    return None, None
                else:
                    frame = camera.draw_text(
                        frame, '[Camera %d] Not enough Charuco markers'
                        'detected.' % camera.cam_id)
            else:
                frame = camera.draw_text(
                    frame, '[Camera %d] Not enough resolution.'
                    'Board is too far.' % camera.cam_id)

        frame = camera.draw_text(
            frame, 'Calibration mode running..',
            y=camera.height - (camera.height/20), thickness=2)
        cv2.imshow("Live camera", frame)
        k = cv2.waitKey(400) % 256
        if k == 27 or k == ord('q'):
            self.running = False
            return None, None
        # camera.release()  # DO NOT FORGET TO RELEASE!
        time.sleep(0.1)  # Sleep to give the time to move the panel
        self.i = 15  # 3 seconds still delay
        return None, None

    def match_keypoints(self, kps_0, kps_1, ids_0, ids_1, ratio, reprojThresh):
        """Match keypoints between 2 frames and find the homography matrix."""
        intersection, indices = snp.intersect(
            ids_0.reshape(-1), ids_1.reshape(-1), indices=True)
        kps_0 = kps_0.reshape(-1, 2)  # 2D keypoints
        kps_1 = kps_1.reshape(-1, 2)  # 2D keypoints

        # At least 4 matches to compute an homography
        if ids_0[indices[0]].shape[0] > 12:
            # Homography between points in kps_0 and kps_1
            # homography, status = cv2.findHomography(
            #     kps_0[indices[0]], kps_1[indices[1]], cv2.RANSAC)
            homography, status = cv2.findHomography(
                kps_1[indices[1]], kps_0[indices[1]], cv2.RANSAC)
            # reprojThresh)

            if status.sum() != kps_0[indices[0]].shape[0]:
                print('WARNING: cv2.findHomography can not match all detected'
                      ' keypoints (%d/%d matches).' % (
                        status.sum(), kps_0[indices[0]].shape[0]))
            return homography
        raise ValueError(
            'cv2.findHomography can not match at least 12 of 24'
            ' corners (%d/24 matches).' % (kps_0[indices[0]].shape[0]))
        return None

    def read_blue2rgb_fused(self):
        """Fuse 3 cameras blue channel to rgb."""
        # if len(self.cameras) < 3:
        #     raise ValueError('Cameras number must be >3 to RGB merge!')
        # TODO: threaded warpPerspective transform?
        frames = list([self.cameras[0].read_undistort()])  # refcam wthout homo
        for camera, homography in zip(self.cameras[1:], self.homographies):
            frame = camera.read_undistort()
            # Perspective transform to fuse multiple camera frames
            # TODO: better than perspective transform ?
            frame = cv2.warpPerspective(
                frame, homography, (frame.shape[1], frame.shape[0]))
            frames.append(frame)
        return cv2.merge(  # Blue channel
            (frames[0][:, :, 0], frames[1][:, :, 0], frames[2][:, :, 0]))

    def read_gray2rgb_fused(self):
        """Fuse 3 cameras grayed frames to rgb."""
        # if len(self.cameras) < 3:
        #     raise ValueError('Cameras number must be >3 to RGB merge!')
        frames = list([self.cameras[0].read_undistort()])  # refcam wthout homo
        # TODO: threaded warpPerspective transform?
        for camera, homography in zip(self.cameras[1:], self.homographies):
            frame = camera.read_undistort()
            # Perspective transform to fuse multiple camera frames
            frame = cv2.warpPerspective(
                frame, homography, (frame.shape[1], frame.shape[0]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        return cv2.merge((frame[0], frame[1], frame[2]))

    def read_weighted_fused(self, blending_ratio):
        """Fuse all cameras with weighted blend."""
        if blending_ratio is None:
            blending_ratio = 1 / len(self.cameras)
        frames = list([self.cameras[0].read_undistort()])  # refcam wthout homo
        # TODO: threaded warpPerspective transform?
        for camera, homography in zip(self.cameras[1:], self.homographies):
            frame = camera.read_undistort()
            # Perspective transform to fuse multiple camera frames
            frame = cv2.warpPerspective(
                frame, homography, (frame.shape[1], frame.shape[0]))
            frames.append(frame)
        for frame in frames:
            frame_out = cv2.addWeighted(
                frame_out, 1 - blending_ratio, frame, blending_ratio, 0)
        return frame_out

    def read_sub_fused(self):
        """Difference fusion."""
        blending_ratio = 1 / (len(self.cameras) - 1)
        frames = list([self.cameras[0].read_undistort()])  # refcam wthout homo
        # TODO: threaded reading and transform
        for camera, homography in zip(self.cameras[1:], self.homographies):
            frame = camera.read_undistort()
            # Perspective transform to fuse multiple camera frames
            # TODO: better than perspective transform ?
            frame = cv2.warpPerspective(
                frame, homography, (frame.shape[1], frame.shape[0]))
            frames.append(frame)
        for frame in frames[1:]:
            frame_out = cv2.addWeighted(
                frame_out, 1 - blending_ratio, frame - frames[0],
                blending_ratio, 0)
        return frame_out

    def release(self):
        """Release all Camera VideoCapture instances."""
        print('Quit..')
        for camera in self.cameras:
            camera.cap.release()
            time.sleep(1)
            print('Camera %d released' % camera.cam_id)


def main(argv):
    """Setup, calibrate and live display two blended Cameras."""
    # Get default camera id based on current platform.
    if sys.platform == 'linux' or sys.platform == 'linux2':
        default_cam_ids = ['/dev/video0', '/dev/video1', '/dev/video2']
    else:  # darwin win32 win64
        default_cam_ids = [0, 1, 2]

    # Parse CLI arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--cam_ids', default=default_cam_ids,
                    help="camera ids list (ex: ='[/dev/video0, /dev/video1]'")
    ap.add_argument('-s', '--settings',
                    help="camera settings list "
                    "(ex:[[(3, 640), (4, 480)], [(3, 640), (4, 480)]]")
    args = vars(ap.parse_args())

    # Default camera settings
    if args["settings"]:
        settings = args["settings"]
    else:
        settings = [[(cv2.CAP_PROP_FRAME_WIDTH, 1280),
                     (cv2.CAP_PROP_FRAME_HEIGHT, 720),
                     (cv2.CAP_PROP_FPS, 30),
                     (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')),
                     (cv2.CAP_PROP_AUTOFOCUS, 1)],
                    [(cv2.CAP_PROP_FRAME_WIDTH, 1280),
                     (cv2.CAP_PROP_FRAME_HEIGHT, 720),
                     (cv2.CAP_PROP_FPS, 30),
                     (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')),
                     (cv2.CAP_PROP_AUTOFOCUS, 1)],
                    [(cv2.CAP_PROP_FRAME_WIDTH, 1280),
                     (cv2.CAP_PROP_FRAME_HEIGHT, 720),
                     (cv2.CAP_PROP_FPS, 30),
                     (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')),
                     (cv2.CAP_PROP_AUTOFOCUS, 1)]]

    aruco_dict_num = cv2.aruco.DICT_6X6_1000
    # also available: DICT_5X5_1000, DICT_4X4_50, DICT_ARUCO_ORIGINAL

    # Initialize Cameras objects with calibration and lens correction
    cam_ids = args['cam_ids']
    if sys.platform != 'linux' and sys.platform != 'linux2':
        cam_ids = [int(cam_id) for cam_id in cam_ids]
    cameras = []
    for cam_id, setting in zip(cam_ids, settings):
        print('Setting up camera %s.' % cam_id)
        cam = Camera(
            cam_id=cam_id, aruco_dict_num=aruco_dict_num, settings=setting)
        cameras.append(cam)

    cameras_fusion = CamerasFusion(cameras)
    cameras_fusion.calibrate_fusion()

    # Open basic live view
    print('Live view running...')
    print('  k to calibrate correction')
    print('  m to save frame')
    print('  v loop between gray2rgb and blue2rgb fusion')
    print('  ESC or q to exit.')
    selected_fused = cameras_fusion.read_blue2rgb_fused
    while True:
        if cameras_fusion.fusion_calibration_is_done:
            frame = selected_fused()
        else:
            for camera in cameras_fusion.cameras:
                frame = camera.read_undistort()
                frame = camera.draw_text(
                    frame, 'Please manually adjust Cameras overlapping, then c'
                    'alibrate.', y=camera.height - (camera.height/20),
                    thickness=2)
                # time.sleep(0.5)
                k = cv2.waitKey(50) % 256
                if k == 27 or k == ord('q'):
                    break
        cv2.imshow("Live camera", frame)
        k = cv2.waitKey(40) % 256
        if k == 27 or k == ord('q'):
            break
        elif k == ord('k'):
            if cameras_fusion.calibrate_fusion():
                print('Calibration done!')
        elif k == ord('m'):
            cv2.imwrite('frame_fused_%s.png' % cam.cam_id, frame)
        elif k == ord('v'):
            if selected_fused == cameras_fusion.read_blue2rgb_fused:
                selected_fused = cameras_fusion.read_gray2rgb_fused
            else:
                selected_fused = cameras_fusion.read_blue2rgb_fused

    cameras_fusion.release()  # DO NOT FORGET TO RELEASE!
    print('All released!')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv[1:])
