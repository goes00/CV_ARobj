import numpy as np
import cv2 as cv

def select_img_from_video(video_file, board_pattern, select_all=False, wait_msec=10, wnd_name='Camera Calibration'):
    # Open a video
    video = cv.VideoCapture(video_file)
    assert video.isOpened()

    # Select images
    img_select = []

    while True:
        # Grab an images from the video
        valid, img = video.read()
        if not valid:
            break

        if select_all:
            img_select.append(img)
        else:
            # Show the image
            display = img.copy()
            cv.putText(display, f'NSelect: {len(img_select)}', (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
            cv.imshow(wnd_name, display)

            # Process the key event
            key = cv.waitKey(wait_msec)
            if key == ord(' '):             # Space: Pause and show corners
                complete, pts = cv.findChessboardCorners(img, board_pattern)
                cv.drawChessboardCorners(display, board_pattern, pts, complete)
                cv.imshow(wnd_name, display)
                key = cv.waitKey()
                if key == ord('\r'):
                    img_select.append(img) # Enter: Select the image
            if key == 27:                  # ESC: Exit (Complete image selection)
                break

    cv.destroyAllWindows()
    return img_select

def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    # Find 2D corner points from given images
    img_points = []
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            img_points.append(pts)
    assert len(img_points) > 0 

    # Prepare 3D points of the chess board
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points) # Must be `np.float32`

    # Calibrate the camera
    return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)

def draw_ARobj(img, center, radius, height, color):
    num_pts = 50  # Number of points to approximate the ARobj
    for i in range(num_pts):
        angle = 2 * np.pi * i / num_pts
        x1 = int(center[0] + radius * np.cos(angle))
        y1 = int(center[1] + radius * np.sin(angle))
        x2 = int(center[0] + radius * np.cos(angle + np.pi))
        y2 = int(center[1] + radius * np.sin(angle + np.pi))
        cv.line(img, (x1, y1), (x2, y2), color, 2)
        cv.circle(img, (x1, y1), 2, color, -1)  # Draw small circles at the ends of the ARobj
        cv.circle(img, (x2, y2), 2, color, -1)

if __name__ == '__main__':
    video_file = 'chess_input.avi'
    board_pattern = (10, 7)
    board_cellsize = 0.025

    img_select = select_img_from_video(video_file, board_pattern)
    assert len(img_select) > 0, 'There is no selected images!'
    rms, K, dist_coeff, rvecs, tvecs = calib_camera_from_chessboard(img_select, board_pattern, board_cellsize)

    # Open the video again
    video = cv.VideoCapture(video_file)
    assert video.isOpened(), 'Cannot read the given input, ' + video_file

    while True:
        # Read an image from the video
        valid, img = video.read()
        if not valid:
            break

        # Estimate the camera pose
        success, img_points = cv.findChessboardCorners(img, board_pattern)
        if success:
            # Draw a ARobj in the center of the chessboard
            center = tuple(np.mean(img_points, axis=0)[0])
            ARobj_radius = 50
            ARobj_height = 100
            ARobj_color = (0, 255, 0)
            draw_ARobj(img, center, ARobj_radius, ARobj_height, ARobj_color)

        # Show the image and process the key event
        cv.imshow('AR Object on Chessboard', img)
        key = cv.waitKey(10)
        if key == 27: # ESC
            break

    video.release()
    cv.destroyAllWindows()
