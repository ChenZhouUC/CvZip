import cv2
import numpy as np
import sys
import os
from copy import deepcopy
module_ = os.path.abspath(__file__)
for layer_ in range(2):
    module_ = os.path.dirname(module_)
sys.path.append(module_)
from image.geometrics_marker import text_marker, rectangle_marker, polygon_marker
from elementary_labeler import ElementaryLabeler


class DetectionLabeler(ElementaryLabeler):
    def __init__(self, window_name, window_size, adsorb_opt=False, adsorb_ratio=0.005, dist_norm=2):
        super(DetectionLabeler, self).__init__(window_name, window_size, adsorb_opt, adsorb_ratio, dist_norm)
        self.region_shift = None
        self.region_cache = []

    def __mouse_handler(self, event, x, y, flags, image):
        image_size = image.shape[:2][::-1]
        self.adsorb_thresh = np.mean(image_size) * self.adsorb_ratio
        self.point_current = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.match = None
            self.region_shift = None
            for _i, _pt in enumerate(self.point_cache):
                if self.dist_func(_pt, self.point_current) <= self.adsorb_thresh:
                    self.match = _i
                    break
            if self.match is not None:
                self.point_cache[self.match] = self.point_current
            else:
                for _i, _rg in enumerate(self.region_cache):
                    for _j, _pt in enumerate(_rg):
                        if self.dist_func(_pt, self.point_current) <= self.adsorb_thresh:
                            self.match = _i
                            self.region_shift = _j
                            break
                    if self.match is None and cv2.pointPolygonTest(np.array(_rg), self.point_current, True) >= 0:
                        self.region_shift = [x, y]
                        self.match = _i
                        break
                if self.match is None:
                    shift_x, shift_y = self.adsorber(x, y, image, np.ceil(2 * self.adsorb_thresh).astype(int))
                    self.point_cache.append((shift_x + x, shift_y + y))
                    self.match = len(self.point_cache) - 1
                else:
                    if type(self.region_shift) == int:
                        self.region_cache[self.match][self.region_shift] = self.point_current
        elif event == cv2.EVENT_LBUTTONUP:
            if self.region_shift is None and self.match is not None:
                # adjusting cached point
                coord_x = np.clip(x, 0, image_size[0] - 1)
                coord_y = np.clip(y, 0, image_size[1] - 1)
                if coord_x != x or coord_y != y:
                    self.point_cache.pop(self.match)
                    self.match = None
                else:
                    self.match = None
            elif self.region_shift is not None:
                coord_x = np.clip(x, 0, image_size[0] - 1)
                coord_y = np.clip(y, 0, image_size[1] - 1)
                adjusted_pointer = (coord_x, coord_y)
                if type(self.region_shift) == int:
                    # adjusting region point
                    self.region_cache[self.match][self.region_shift] = adjusted_pointer
                    self.match = None
                    self.region_shift = None
                else:
                    # adjusting region
                    self.region_cache[self.match] = np.array(self.region_cache[self.match]) + np.array(adjusted_pointer) - np.array(self.region_shift)
                    self.region_cache[self.match] = [tuple(_pt) for _pt in self.region_cache[self.match]]
                    self.match = None
                    self.region_shift = None
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            if self.region_shift is None and self.match is not None:
                # adjusting cached point
                coord_x = np.clip(x, 0, image_size[0] - 1)
                coord_y = np.clip(y, 0, image_size[1] - 1)
                self.point_cache[self.match] = (coord_x, coord_y)
            elif self.region_shift is not None:
                coord_x = np.clip(x, 0, image_size[0] - 1)
                coord_y = np.clip(y, 0, image_size[1] - 1)
                adjusted_pointer = (coord_x, coord_y)
                if type(self.region_shift) == int:
                    # adjusting region point
                    self.region_cache[self.match][self.region_shift] = adjusted_pointer
                else:
                    # adjusting region
                    self.region_cache[self.match] = np.array(self.region_cache[self.match]) + np.array(adjusted_pointer) - np.array(self.region_shift)
                    self.region_cache[self.match] = [tuple(_pt) for _pt in self.region_cache[self.match]]
                    self.region_shift = list(adjusted_pointer)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.match is None:
                for _i, _pt in enumerate(self.point_cache):
                    if self.dist_func(_pt, self.point_current) <= self.adsorb_thresh:
                        self.point_cache.pop(_i)
                        return
                for _i, _rg in enumerate(self.region_cache):
                    for _j, _pt in enumerate(_rg):
                        if self.dist_func(_pt, self.point_current) <= self.adsorb_thresh:
                            if len(self.region_cache[_i]) >= 4:
                                self.region_cache[_i].pop(_j)
                                return
                    if cv2.pointPolygonTest(np.array(_rg), self.point_current, True) >= 0:
                        self.region_cache.pop(_i)
                        return

    def __status_handler(self, status):
        if status.lower() == "append":
            if self.match is not None and type(self.region_shift) != int:
                cand = (np.array(self.region_cache[self.match][0]) + np.array(self.region_cache[self.match][-1])) / 2
                center = np.mean(self.region_cache[self.match], axis=0)
                cand += (cand - center) * 0.1
                self.region_cache[self.match].append(tuple(cand.astype(int)))
                print(self.region_cache[self.match])
        elif status.lower() == "add":
            if self.match is None:
                if len(self.point_cache) >= 3:
                    self.region_cache.append(deepcopy(self.point_cache))
                    self.point_cache = []
        elif status.lower() == "save":
            if self.match is None:
                if len(self.point_cache) >= 3:
                    self.region_cache.append(deepcopy(self.point_cache))
                    self.point_cache = []
            # todo
            self.clear()

    def renderImage(self, img, status_dict, wait=10, pt_color=(0, 255, 0), plg_color=(255, 0, 0)):
        if img.shape[2] == 3:
            cv2.setMouseCallback(self.window_name, self.__mouse_handler, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        else:
            cv2.setMouseCallback(self.window_name, self.__mouse_handler, img)
        for _j, _rg in enumerate(self.region_cache):
            img = polygon_marker(img, _rg, plg_color, np.ceil(self.adsorb_thresh).astype(int), 0.5, True, 1 + np.ceil(self.adsorb_thresh).astype(int), True)
        try:
            img = polygon_marker(img, self.point_cache, pt_color, np.ceil(self.adsorb_thresh).astype(int), 0.5, True, 1 + np.ceil(self.adsorb_thresh).astype(int), False)
        except:
            pass
        text_marker(img, str(len(self.point_cache)), (0, 0), (0, 0), cv2.FONT_HERSHEY_TRIPLEX, 2, 3, (0, 255, 255))
        cv2.imshow(self.window_name, img)
        pressed_key = cv2.waitKey(wait) & 0xFF
        for k, v in status_dict.items():
            if pressed_key == ord(k):
                self.__status_handler(v)

    def clear(self):
        self.point_cache = []
        self.point_current = (-1, -1)
        self.match = None
        self.adsorb_thresh = np.mean(self.window_size) * self.adsorb_ratio
        self.region_shift = None
        self.region_cache = []


if __name__ == '__main__':
    window_name = "Elementary Labeler"
    window_size = (1600, 1200)
    labeler = DetectionLabeler(window_name, window_size, False)

    img_path = "/home/chenzhou/Pictures/Concept/python-package.webp"
    colorful = cv2.imread(img_path)
    status_dict = {"a": "APPEND", "s": "SAVE", "d": "ADD"}
    while True:
        labeler.renderImage(colorful.copy(), status_dict)
