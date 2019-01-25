"""OpenCV CamerasFusion class for multiple cameras fusion."""

from camera_fusion.CameraCorrected import CameraCorrected

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
    from cv2 import aruco
except ImportError:
    raise ImportError('ERROR opencv-contrib-python must be installed!')


class CamerasFusion(object):
    """CamerasFusion class used to match and fuse many corrected vamera.

    Attributes:
        cameras (list of CameraCorrected): list of corrected cam to be fused.
        homographies (list): Homography matrix between 1st and current cam.

    """

    def __init__(self, cameras):
        """Initialize the CameraCorrected object variables.

        Args:
            cameras (list of CameraCorrected): cameras to be blend.
        """
        if len(cameras) == 0:
            raise ValueError('Cameras list is empty!')
        self.cameras = cameras
        self.board = self.cameras[0].board
        self.homographies = []
        self.fusion_calibration_is_done = False
        self.calibrate_stereo_is_done = False
        self.i = 15  # 3 seconds delay before frame capture.
        self.running = True

    def initialize(self):
        """Set up cameras fusion. Launch calibration if necessary."""
        self.ratio = 0.99
        self.reprojThresh = 10  # 4
        fusionParameters_path = Path('./data/fusionParameters.npy')
        if fusionParameters_path.exists():
            print("Found fusionParameters.npy.\n")
            self.homographies = np.load(str(fusionParameters_path))
            if len(self.homographies) + 1 != len(self.cameras):
                raise ValueError(
                    'fusionParameters.npy cameras number (=%d) is different'
                    ' from the specified cameras number (=%d!)'
                    'Remove fusionParameters.npy to compute a new one.' % (
                        len(self.homographies) + 1, len(self.cameras)))
            self.fusion_calibration_is_done = True
        else:
            self.calibrate_fusion()

    def calibrate_fusion(self):
        """Launch calibration routing for the fusion of all cameras."""
        print('Starting the fusion calibration routine.')
        keypoints_features_list = []
        self.homographies = []
        # Loop thru cameras and assure every one matches the same keypoints.
        while self.running:
            for camera in self.cameras:
                camera.corners, camera.ids = self.detect_keypoints(camera)
            if sum(x is not None for x in [
                    cam.ids for cam in self.cameras]) == len(self.cameras):
                break
        if not self.running:
            return False
        # Match features between many cameras detected corners keypoints
        for camera in self.cameras[1:]:
            H = self.match_keypoints(
                self.cameras[0].corners, camera.corners,
                self.cameras[0].ids, camera.ids, self.ratio, self.reprojThresh)
            if H is None:
                raise ValueError('In fusion calibration: Can not match at '
                                 ' least 4 keypoints between cameras.\n'
                                 'Please re-do the calibration.')
                self.fusion_calibration_is_done = False
                self.homographies = []
            self.homographies.append(H)
        fusionParameters_path = Path('./data/fusionParameters.npy')
        np.save(str(fusionParameters_path), self.homographies)
        self.fusion_calibration_is_done = True
        print("Fusion calibration done!")

    def calibrate_stereo(self):
        """Launch the routing for the stereo calibration of all cameras."""
        stereoParameters_path = Path('./data/stereoParameters.npy')
        self.img_size = (
                self.cameras[0].current_frame.shape[1],
                self.cameras[0].current_frame.shape[0])
        if stereoParameters_path.exists():
            print('Found a stereo calibration file.')
            camera_model = np.load(str(stereoParameters_path))
            self.stereocalibration_retval = camera_model.item().get(
                'rms_stereo')
            self.cameraMatrix1 = camera_model.item().get('cameraMatrix1')
            self.cameraMatrix2 = camera_model.item().get('cameraMatrix2')
            self.distCoeffs1 = camera_model.item().get('distCoeffs1')
            self.distCoeffs2 = camera_model.item().get('distCoeffs2')
            self.R = camera_model.item().get('R')
            self.T = camera_model.item().get('T')
            self.E = camera_model.item().get('E')
            self.F = camera_model.item().get('F')
        else:
            print('Starting the stereo calibration routine.')
            # https://stackoverflow.com/questions/50630769/how-to-get-a-good-cv2-stereocalibrate-after-successful-cv2-calibratecamera
            # https://github.com/bvnayak/stereo_calibration/blob/master/camera_calibrate.py
            # https://github.com/groundmelon/camera_calibration_frontend/blob/master/scripts/camera_calibration_frontend/calibrator.py
            # https://github.com/tobybreckon/python-examples-cv/blob/master/stereo_sgbm.py
            # https://github.com/Algomorph/AMBR/blob/master/tracker.py
            # https://github.com/jimchenhub/Smart-Car/blob/master/source/implementation/avoiding/stereomatch.py
            # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_depthmap/py_depthmap.html
            # https://erget.wordpress.com/2014/04/27/producing-3d-point-clouds-with-a-stereo-camera-in-opencv/
            # https://stackoverflow.com/questions/38653354/straightforward-solution-on-how-to-stereo-calibration-and-rectifications-opencv
            stereocalibration_criteria = (
                cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
            flags = 0
            flags = cv2.CALIB_FIX_INTRINSIC
            # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS
            flags |= cv2.CALIB_FIX_FOCAL_LENGTH
            # flags |= cv2.CALIB_FIX_ASPECT_RATIO
            flags |= cv2.CALIB_ZERO_TANGENT_DIST
            # flags |= cv2.CALIB_RATIONAL_MODEL
            # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
            # flags |= cv2.CALIB_FIX_K3
            # flags |= cv2.CALIB_FIX_K4
            # flags |= cv2.CALIB_FIX_K5

            # Loop thru cameras and assure every one matches same keypoints.
            corners = []
            corners_buff = []

            # Set up a set of real-world "object points" for the chessboard.
            patternX = 4
            patternY = 6
            square_size_in_m = self.cameras[0].charuco_marker_size
            objp = np.zeros((patternX * patternY, 3), np.float32)
            objp[:,:2] = np.mgrid[0:patternX, 0:patternY].T.reshape(-1, 2)
            objp = objp * square_size_in_m  # in meters
            print('Will calibrate for a board X%d*Y%d markers sized: %fm' % (
                patternX, patternY, square_size_in_m))

            objpoints = []
            for j in range(0, len(self.cameras)):
                corners.append([])
            for i in range(0, 10):  # Takes 10 differents board position
                while self.running:
                    corners_buff = []
                    all_corners = True
                    for camera in self.cameras:
                        camera.corners, camera.ids = self.detect_keypoints(
                            camera)
                        corners_buff.append(camera.corners)
                        if camera.corners is not None:
                            if len(camera.corners) != (4 * 6):
                                # Check we detect all corners:
                                all_corners = False
                                print('Not all 24 corners detected: '
                                      '%d/24. Check occlusions' % len(
                                        camera.corners))
                    if sum(
                        x is not None for x in [
                        cam.ids for cam in self.cameras]) == len(
                            self.cameras):
                        if all_corners is True:
                            break
                if not self.running:
                    return False
                corners_buff = corners_buff[-3:]  # .reshape((3, 24, 2))
                for j in range(0, len(self.cameras)):
                    # Convert Numpy positions to OpenCV tuple Point2f
                    # TODO: RLY UF?
                    for k in range(0, len(corners_buff[j])):
                        corners_buff[j][k] = np.array(
                            tuple(corners_buff[j][k][0].ravel()))
                    corners[j].append(np.array(corners_buff[j]))
                objpoints.append(objp)
                print('Position %d done!' % i)
                time.sleep(2)

            # print(np.array(corners).shape)
            # (3, 10, 24, 1, 2)  # Then it works
            imgpoints_right = corners[1]
            imgpoints_left = corners[2]
            cameraMatrix1 = self.cameras[1].camera_matrix
            distCoeffs1 = self.cameras[1].dist_coeffs
            cameraMatrix2 = self.cameras[2].camera_matrix
            distCoeffs2 = self.cameras[2].dist_coeffs

            self.stereocalibration_retval, self.cameraMatrix1, self.distCoeffs1, self.cameraMatrix2, self.distCoeffs2, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
                objpoints, imgpoints_left, imgpoints_right, cameraMatrix1,
                distCoeffs1, cameraMatrix2, distCoeffs2, self.img_size,
                criteria = stereocalibration_criteria, flags = flags)

            camera_model = dict(
                [('rms_stereo', self.stereocalibration_retval),
                 ('cameraMatrix1', self.cameraMatrix1),
                 ('cameraMatrix2', self.cameraMatrix2),
                 ('distCoeffs1', self.distCoeffs1), ('distCoeffs2', self.distCoeffs2),
                 ('R', self.R), ('T', self.T), ('E', self.E), ('F', self.F)])
            np.save(str(stereoParameters_path), camera_model)

        print('Stereo calibration done!\n')
        print('RMS left to  right re-projection error:', self.stereocalibration_retval)
        print('IntrinsicCameraMtx1\n', self.cameraMatrix1)
        print('distCoeffs1\n', self.distCoeffs1)
        print('IntrinsicCameraMtx2\n', self.cameraMatrix2)
        print('distCoeffs2\n', self.distCoeffs2)
        print('R\n', self.R)
        print('T\n', self.T)
        print('E\n', self.E)
        print('F\n', self.F)
        R1 = np.zeros(shape=(3, 3))
        R2 = np.zeros(shape=(3, 3))
        P1 = np.zeros(shape=(3, 3))  # (3, 4)
        P2 = np.zeros(shape=(3, 3))  # (3, 4)

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.cameraMatrix1, self.distCoeffs1, self.cameraMatrix2, self.distCoeffs2, self.img_size,
            self.R, self.T, R1, R2, P1, P2, Q=None, flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=-1, newImageSize=(0, 0))
        # cv2.stereoRectify(
        #     cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, self.img_size,
        #     R, T, R1, R2, P1, P2, Q=None, flags=cv2.CALIB_ZERO_DISPARITY,
        #     alpha=-1, newImageSize=(0, 0))

        # stereoRectify(
        #     cameraMatrix1, distCoeffs1, cameraMatrix2,
        #     distCoeffs2,(width, height), R, T)

        # fs1 << "R1" << R1;
        # fs1 << "R2" << R2;
        # fs1 << "P1" << P1;
        # fs1 << "P2" << P2;
        # fs1 << "Q" << Q;

        print('roi1:')
        print(roi1)
        print('roi2:')
        print(roi2)

        print('Done Rectification')
        print('Applying Undistort')

        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.cameraMatrix1, self.distCoeffs1, R1, P1, self.img_size, cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.cameraMatrix2, self.distCoeffs2, R2, P2, self.img_size, cv2.CV_32FC1)

        print('Undistort complete')

        # Create stereo BM class
        # self.stereoBM = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        # self.stereoBM = cv2.StereoBM_create(numDisparities=112, blockSize=31)
        self.stereoBM = cv2.StereoBM_create(numDisparities=192, blockSize=5)
        # self.stereoBM = cv2.StereoBM_create(numDisparities=256, blockSize=7)
        # self.stereoBM = cv2.StereoBM_create(numDisparities=1024, blockSize=7)
        # self.stereoBM = cv2.StereoBM_create(numDisparities=1536, blockSize=5)
        self.stereoBM.setPreFilterSize(31)  # LD 41
        self.stereoBM.setPreFilterType(cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE)
        self.stereoBM.setPreFilterCap(40)  # 31
        self.stereoBM.setTextureThreshold(10)
        self.stereoBM.setMinDisparity(0)
        self.stereoBM.setSpeckleWindowSize(10)  # 100
        self.stereoBM.setSpeckleRange(16)  # 64
        self.stereoBM.setUniquenessRatio(10)  # 
        # self.stereoBM.setROI1(roi1)
        # self.stereoBM.setROI1(roi2)

        # SGBM Parameters
        # http://timosam.com/python_opencv_depthimage
        self.window_size = 3 
        # wsize default 3; 5; 7 for SGBM reduced size image
        # 15 for SGBM full size image (1300px and above); 5 works nicely

        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=160,  # max_disp has to be dividable by 16 f. E. HH 192, 256
            blockSize=5,
            P1=8 * 3 * self.window_size ** 2,
            P2=32 * 3 * self.window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)

        # FILTER Parameters
        self.lmbda = 80000
        self.sigma = 1.2
        self.visual_multiplier = 1.0

        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(
            matcher_left=self.left_matcher)
        self.wls_filter.setLambda(self.lmbda)
        self.wls_filter.setSigmaColor(self.sigma)

        # cv2.namedWindow('imageL', cv2.WINDOW_NORMAL)  # cv2.WINDOW_NORMAL
        # cv2.resizeWindow('imageL', 1680, 1050)
        # cv2.namedWindow('imageR', cv2.WINDOW_NORMAL)  # cv2.WINDOW_NORMAL
        # cv2.resizeWindow('imageR', 1680, 1050)
        # cv2.namedWindow('image1L', cv2.WINDOW_NORMAL)  # cv2.WINDOW_NORMAL
        # cv2.resizeWindow('image1L', 1680, 1050)
        # cv2.namedWindow('image2R', cv2.WINDOW_NORMAL)  # cv2.WINDOW_NORMAL
        # cv2.resizeWindow('image2R', 1680, 1050)
        self.calibrate_stereo_is_done = True

    def detect_keypoints(self, camera):
        """Detect ChAruco corner keypoints in a corrected camera frame."""
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
                    cv2.imshow('Live', frame)
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
        cv2.imshow('Live', frame)
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
            # Homography between points in kps_1 and kps_0
            homography, status = cv2.findHomography(
                kps_1[indices[1]], kps_0[indices[0]], cv2.RANSAC)
            # reprojThresh)

            if status.sum() != kps_0[indices[0]].shape[0]:
                print('WARNING: cv2.findHomography can not match all detected'
                      ' keypoints (%d/%d matches).' % (
                        status.sum(), kps_0[indices[0]].shape[0]))
            return homography
        self.release()
        raise ValueError(
            'cv2.findHomography can not match at least the half of the'
            ' corners (%d/24 matched).' % (kps_0[indices[0]].shape[0]))
        return None

    def read_blue2rgb_fused(self):
        """Fuse 3 cameras blue channel to rgb."""
        if len(self.cameras) < 3:
            raise ValueError('Cameras number must be >=3 to RGB merge!')
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
        if len(self.cameras) < 3:
            raise ValueError('Cameras number must be >=3 to RGB merge!')
        # TODO: threaded warpPerspective transform?
        frames = cv2.cvtColor(self.cameras[0].read_undistort(),
                              cv2.COLOR_BGR2GRAY)  # Get first cam in grayscale
        frames = list([frames])  # refcam without homography
        for camera, homography in zip(self.cameras[1:], self.homographies):
            frame = camera.read_undistort()
            # Perspective transform to fuse multiple camera frames
            frame = cv2.warpPerspective(
                frame, homography, (frame.shape[1], frame.shape[0]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        return cv2.merge(  # Gray channel
            (frames[0], frames[1], frames[2]))

    def read_stereoBM_disparity(self):
        """Fuse 2 cameras with stereoBM and display disparity map."""
        if self.calibrate_stereo_is_done is False:
            self.calibrate_stereo()

        img1 = self.cameras[1].current_frame
        img2 = self.cameras[2].current_frame
        imgU1 = img1
        imgU2 = img2
        # imgU1 = np.zeros(self.cameras[1].current_frame.shape, np.uint8)
        # imgU1 = cv2.remap(
        #     img1, self.map1x, self.map1y, cv2.INTER_LINEAR, imgU1,
        #     cv2.BORDER_CONSTANT, 0)
        # imgU2 = cv2.remap(img2, self.map2x, self.map2y, cv2.INTER_LINEAR)

        # cv2.imshow('imageL', img1)
        # cv2.imshow('imageR', img2)
        # cv2.imshow('image1L', imgU1)
        # cv2.imshow('image2R', imgU2)
        disparity_map = self.stereoBM.compute(
                cv2.cvtColor(imgU1, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(imgU2, cv2.COLOR_BGR2GRAY))
        disparity_map = cv2.normalize(
            src=disparity_map, dst=disparity_map, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        disparity_map = np.uint8(disparity_map)
        frame_out = cv2.applyColorMap(disparity_map, cv2.COLORMAP_HOT)
        return frame_out

    def read_stereoSGBM_disparity(self):
        """Fuse 2 cameras with stereoSGBM and display disparity map."""
        if self.calibrate_stereo_is_done is False:
            self.calibrate_stereo()

        img1 = self.cameras[1].current_frame
        img2 = self.cameras[2].current_frame
        imgU1 = img1
        imgU2 = img2
        # imgU1 = np.zeros(self.cameras[1].current_frame.shape, np.uint8)
        # imgU1 = cv2.remap(
        #     img1, self.map1x, self.map1y, cv2.INTER_LINEAR, imgU1,
        #     cv2.BORDER_CONSTANT, 0)
        # imgU2 = cv2.remap(img2, self.map2x, self.map2y, cv2.INTER_LINEAR)

        displ = self.left_matcher.compute(imgU1, imgU2)  # .astype(np.float32)/16
        dispr = self.right_matcher.compute(imgU2, imgU1)  # .astype(np.float32)/16
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        filteredImg = self.wls_filter.filter(displ, imgU1, None, dispr)  # important to put "imgL" here!!!
        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        filteredImg = np.uint8(filteredImg)
        frame_out = cv2.applyColorMap(filteredImg, cv2.COLORMAP_HOT)
        return frame_out

    def read_weighted_fused(self, blending_ratio=None):
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
        frame_out = frames[0]
        for frame in frames[1:]:
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
        frame_out = frames[0]
        for frame in frames[1:]:
            frame_out = cv2.addWeighted(
                frame_out, 1 - blending_ratio, frame - frames[0],
                blending_ratio, 0)
        return frame_out

    def read_sub2rgb_fused(self):
        """Substract fusion to RGB colormap."""
        if len(self.cameras) != 3:
            raise ValueError('Cameras number must be 3 to RGBsubstract merge!')
        # TODO: threaded warpPerspective transform?
        frames = cv2.cvtColor(self.cameras[0].read_undistort(),
                              cv2.COLOR_BGR2GRAY)  # Get first cam in grayscale

        frames = list([frames])  # refcam without homography
        for camera, homography in zip(self.cameras[1:], self.homographies):
            frame = camera.read_undistort()
            # Perspective transform to fuse multiple camera frames
            frame = cv2.warpPerspective(
                frame, homography, (frame.shape[1], frame.shape[0]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frames[0] - frame)
        return cv2.merge(  # Gray channels to BGR
            (frames[0], frames[1], frames[2]))

    def release(self):
        """Release all Camera VideoCapture instances."""
        print('\nQuit..')
        for camera in self.cameras:
            camera.release()
            while(camera.thread.is_alive()):
                time.sleep(0.05)
            print('Camera %d released' % camera.cam_id)
