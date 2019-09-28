import math
import imutils
import logging

import numpy as np
import cv2 as cv

from matplotlib import pyplot as plt

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger('User similarity calculator')


def show_image(image_mat, dpi=250):
    nchannels = len(image_mat.shape)
    if (nchannels == 2):
        plt.imshow(image_mat, cmap='gray')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.figure(dpi=dpi)
        plt.show()
    else:
        b_, g_, r_ = cv.split(image_mat)
        image_mat_rgb = cv.merge([r_, g_, b_])
        plt.imshow(image_mat_rgb)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.figure(dpi=dpi)
        plt.show()


def distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    m = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, m, (max_width, max_height))

    # return the warped image
    return warped


class PreprocesingImage(object):

    @staticmethod
    def scale_image_with_path(image_path, height, save_path=None):
        image = cv.imread(image_path, cv.IMREAD_COLOR)
        image_resize = imutils.resize(image, height=height)
        if save_path is not None:
            cv.imwrite(save_path, image_resize)
        return image_resize

    @staticmethod
    def scale_image_wit_image(image, height, save_path=None):
        image_resize = imutils.resize(image, height=height)
        if save_path is not None:
            cv.imwrite(save_path, image_resize)
        return image_resize

    @staticmethod
    def crop_card(image_path, height, save_path=None):
        image = cv.imread(image_path, cv.IMREAD_COLOR)
        image_resize = imutils.resize(image, height=height)
        ratio = image.shape[0] / 500.0
        orig_image = image.copy()

        gray = cv.GaussianBlur(image_resize, (3, 3), 3)
        dst = cv.Canny(gray, 50, 100, None, 3)

        logger.info("Copy edges to the images that will display the results in BGR")
        cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
        lines = cv.HoughLines(dst, 1, np.pi / 180, 135, None, 0, 0)

        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

        show_image(cdst)
        pts = []

        for i in range(lines.shape[0]):
            (rho1, theta1) = lines[i, 0]
            m1 = -1 / np.tan(theta1)
            c1 = rho1 / np.sin(theta1)
            for j in range(i + 1, lines.shape[0]):
                (rho2, theta2) = lines[j, 0]
                m2 = -1 / np.tan(theta2)
                c2 = rho2 / np.sin(theta2)
                if np.abs(m1 - m2) <= 1e-8:
                    continue
                x = (c2 - c1) / (m1 - m2)
                y = m1 * x + c1
                if 0 <= x < cdst.shape[1] and 0 <= y < cdst.shape[0]:
                    pts.append((int(x), int(y)))

        pts_corner = []
        sorted_pts_x_axis = sorted(pts, key=lambda tup: tup[0])
        sorted_pts_y_axis = sorted(pts, key=lambda tup: tup[1])

        x_thres = (sorted_pts_x_axis[0][0] + sorted_pts_x_axis[-1][0]) / 2
        y_thres = (sorted_pts_y_axis[0][1] + sorted_pts_y_axis[-1][1]) / 2

        right = []
        left = []

        for i in range(len(pts)):
            if pts[i][0] <= x_thres:
                right.append(pts[i])
            else:
                left.append(pts[i])

        centre_pt = (x_thres, y_thres)

        right_up = []
        right_down = []
        for i in range(len(right)):
            if right[i][1] <= y_thres:
                right_up.append(right[i])
            else:
                right_down.append(right[i])

        thresh = -1
        right_up_corner = None
        for i in range(len(right_up)):
            dist = distance(right_up[i], centre_pt)
            if dist > thresh:
                thresh = dist
                right_up_corner = right_up[i]
        if right_up_corner is None:
            return False
        else:
            pts_corner.append(right_up_corner)

        thresh = -1
        right_down_corner = None
        for i in range(len(right_down)):
            dist = distance(right_down[i], centre_pt)
            if dist > thresh:
                thresh = dist
                right_down_corner = right_down[i]
        if right_down_corner is None:
            return False
        else:
            pts_corner.append(right_down_corner)

        left_up = []
        left_down = []
        for i in range(len(left)):
            if left[i][1] <= y_thres:
                left_up.append(left[i])
            else:
                left_down.append(left[i])

        thresh = -1
        left_down_corner = None
        for i in range(len(left_down)):
            dist = distance(left_down[i], centre_pt)
            if dist > thresh:
                thresh = dist
                left_down_corner = left_down[i]
        if left_down_corner is None:
            return False
        else:
            pts_corner.append(left_down_corner)

        thresh = -1
        left_up_corner = None
        for i in range(len(left_up)):
            dist = distance(left_up[i], centre_pt)
            if dist > thresh:
                thresh = dist
                left_up_corner = left_up[i]
        if left_up_corner is None:
            return False
        else:
            pts_corner.append(left_up_corner)

        pts_np = np.array(pts_corner)
        pts_use = pts_np[:, None]  # We need to convert to a 3D numpy array with a singleton 2nd dimension
        hull = cv.convexHull(pts_use)

        out2 = np.dstack([dst, dst, dst])
        for pt in hull[:, 0]:
            cv.circle(out2, tuple(pt), 2, (0, 255, 0), 2)
        # show_image(out2)

        corner_arr = np.asarray(pts_corner, dtype="float32")
        # warped = four_point_transform(image_resize, corner_arr)
        warped = four_point_transform(orig_image, corner_arr * ratio)
        warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
        if save_path is not None:
            cv.imwrite(save_path, warped)
        return warped
