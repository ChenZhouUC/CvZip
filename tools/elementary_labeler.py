import cv2
import numpy as np
import sys
import os
module_ = os.path.abspath(__file__)
for layer_ in range(2):
    module_ = os.path.dirname(module_)
sys.path.append(module_)
from image.geometrics_marker import text_marker


class ElementaryLabeler(object):
    def __init__(self, window_name, window_size, adsorb_opt=False, adsorb_ratio=0.005, dist_norm=2):

        def __MinkowskiDistance(x, y):
            return np.linalg.norm(np.array(x) - np.array(y), ord=dist_norm)

        def __CannyAdsorber(x, y, img, radius):
            if adsorb_opt:
                img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)
                center = np.array([radius, radius])
                perimeter_left = x - radius
                perimeter_right = np.clip(x + radius, 0, img.shape[1] - 1)
                perimeter_top = y - radius
                perimeter_down = np.clip(y + radius, 0, img.shape[0] - 1)
                if perimeter_left < 0:
                    center[0] += perimeter_left
                    perimeter_left = 0
                if perimeter_top < 0:
                    center[1] += perimeter_top
                    perimeter_top = 0
                edges = cv2.Canny(img[perimeter_top:(perimeter_down + 1), perimeter_left: (perimeter_right + 1)], 100, 200)
                max_val = np.max(edges)
                shift_x, shift_y = 0, 0
                if max_val < 200:
                    return shift_x, shift_y
                ys_, xs_ = np.where(edges == max_val)
                min_dist = np.inf
                for _i in range(len(ys_)):
                    shift_x_ = xs_[_i] - center[0]
                    shift_y_ = ys_[_i] - center[1]
                    dist_ = __MinkowskiDistance([shift_x_, shift_y_], [0, 0])
                    if dist_ < min_dist:
                        shift_x, shift_y = shift_x_, shift_y_
                        min_dist = dist_
                return shift_x, shift_y
            else:
                return 0, 0

        self.window_name = window_name
        self.window_size = window_size
        cv2.namedWindow(self.window_name, cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_size[0], self.window_size[1])

        self.point_cache = []
        self.point_current = (-1, -1)
        self.match = None

        self.adsorb_ratio = adsorb_ratio
        self.adsorb_thresh = np.mean(self.window_size) * self.adsorb_ratio

        self.dist_func = __MinkowskiDistance
        self.adsorber = __CannyAdsorber

    def handlePoint(self, event, x, y, flags, image):
        image_size = image.shape[:2][::-1]
        self.adsorb_thresh = np.mean(image_size) * self.adsorb_ratio
        self.point_current = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.match = None
            for _i, _pt in enumerate(self.point_cache):
                if self.dist_func(_pt, self.point_current) <= self.adsorb_thresh:
                    self.match = _i
                    break
            if self.match is not None:
                self.point_cache[self.match] = self.point_current
            else:
                shift_x, shift_y = self.adsorber(x, y, image, np.ceil(2 * self.adsorb_thresh).astype(int))
                self.point_cache.append((shift_x + x, shift_y + y))
                self.match = len(self.point_cache) - 1
        elif event == cv2.EVENT_LBUTTONUP:
            if self.match is not None:
                coord_x = np.clip(x, 0, image_size[0] - 1)
                coord_y = np.clip(y, 0, image_size[1] - 1)
                if coord_x != x or coord_y != y:
                    self.point_cache.pop(self.match)
                    self.match = None
                else:
                    # self.point_cache[self.match] = self.point_current
                    self.match = None
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            if self.match is not None:
                coord_x = np.clip(x, 0, image_size[0] - 1)
                coord_y = np.clip(y, 0, image_size[1] - 1)
                self.point_cache[self.match] = (coord_x, coord_y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.match is None:
                for _i, _pt in enumerate(self.point_cache):
                    if self.dist_func(_pt, self.point_current) <= self.adsorb_thresh:
                        self.point_cache.pop(_i)
                        break

    def renderImage(self, img, wait=30, pt_color=(0, 255, 0)):
        if img.shape[2] == 3:
            cv2.setMouseCallback(self.window_name, self.handlePoint, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        else:
            cv2.setMouseCallback(self.window_name, self.handlePoint, img)
        for _i, _pt in enumerate(self.point_cache):
            cv2.circle(img, _pt, np.ceil(self.adsorb_thresh).astype(int), pt_color, -1)
        text_marker(img, str(len(self.point_cache)), (0, 0), (0, 0), cv2.FONT_HERSHEY_TRIPLEX, 2, 3, (0, 255, 255))
        cv2.imshow(self.window_name, img)
        cv2.waitKey(wait)

    def clear(self):
        self.point_cache = []
        self.point_current = (-1, -1)
        self.match = None
        self.adsorb_thresh = np.mean(self.window_size) * self.adsorb_ratio


if __name__ == '__main__':
    window_name = "Elementary Labeler"
    window_size = (800, 600)
    labeler = ElementaryLabeler(window_name, window_size, False)

    img_path = "/home/chenzhou/Pictures/Concept/python-package.webp"
    colorful = cv2.imread(img_path)
    while True:
        labeler.renderImage(colorful.copy())
