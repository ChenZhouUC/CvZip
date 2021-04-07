import cv2
import numpy as np
from copy import deepcopy
import sys
import os
module_ = os.path.abspath(__file__)
for layer_ in range(3):
    module_ = os.path.dirname(module_)
sys.path.append(module_)
from image.geometrics_marker import text_marker, polygon_marker, rectangle_marker
from elementary_labeler import ElementaryLabeler

FONT, FONTSCALE, THICKNESS = cv2.FONT_HERSHEY_COMPLEX, 1.5, 2


class ClassifiedDetectionLabeler(ElementaryLabeler):
    def __init__(self, window_name, window_size, adsorb_opt=False, adsorb_ratio=0.005, dist_norm=2, region_type="rect", class_dict={0: "default"}, render_dict={0: (255, 0, 0)}):
        super(ClassifiedDetectionLabeler, self).__init__(window_name, window_size, adsorb_opt, adsorb_ratio, dist_norm)
        assert region_type.lower() in ["rect", "poly"]
        assert type(class_dict) == dict and len(class_dict.keys()) > 0
        assert type(render_dict) == dict and len(render_dict.keys()) == len(class_dict.keys())
        self.class_dict = class_dict
        self.render_dict = render_dict
        self.class_list = list(self.class_dict.keys())
        self.class_selected = self.class_list[0]
        self.region_shift = None
        self.region_cache = {k: [] for k in self.class_dict.keys()}
        self.region_type = region_type

        self.__legend_panel()

    def __legend_panel(self, cubic_shink=1, background=(255, 255, 255), text_color=(0, 0, 0), spacing=10):
        width = 0
        height = 0
        for k in self.class_list:
            (label_width, label_height), baseline = cv2.getTextSize(" " + str(k) + " " + self.class_dict[k], FONT, FONTSCALE, THICKNESS)
            width = max(width, label_width)
            height = max(height, label_height + baseline)
        cubic_size = height - 2 * cubic_shink
        cubic_shink = cubic_shink if cubic_size > 0 else 0
        cubic_size = cubic_size if cubic_size > 0 else height
        width = 2 * spacing + width + cubic_size
        total_height = height * len(self.class_list) + spacing * (len(self.class_list) + 1)
        channels = len(background)
        panel = (np.ones([total_height, width, channels]) * np.array(background)).astype(np.uint8)
        for _i, k in enumerate(self.class_list):
            row_start = _i * height + (_i + 1) * spacing
            panel = rectangle_marker(panel, [[cubic_shink + spacing, row_start], [cubic_shink + spacing + cubic_size, row_start + cubic_size]], self.render_dict[k], 1, 0.8, True, 1)
            panel = text_marker(panel, " " + str(k) + " " + self.class_dict[k], (spacing + height, row_start), (0, 0), FONT, FONTSCALE, THICKNESS, text_color)
        self.legend = panel

    def __patch_legend(self, img, background=(255, 255, 255)):
        height = img.shape[0]
        legend_height = self.legend.shape[0]
        if height < legend_height:
            patch_bottom = (np.ones([legend_height - height, img.shape[1], img.shape[2]]) * np.array(background)).astype(np.uint8)
            img = np.concatenate((img, patch_bottom), axis=0)
        else:
            patch_bottom = (np.ones([height - legend_height, self.legend.shape[1], self.legend.shape[2]]) * np.array(background)).astype(np.uint8)
            legend = np.concatenate((self.legend, patch_bottom), axis=0)
        img = np.concatenate((img, legend), axis=1)
        return img

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
            if self.match is None and self.region_type == "rect" and len(self.point_cache) == 4:
                if cv2.pointPolygonTest(np.array(self.point_cache), self.point_current, True) >= 0:
                    self.match = [x, y]
            if self.match is not None:
                if type(self.match) == int:
                    self.point_cache[self.match] = self.point_current
            else:
                for _i, _rg in enumerate(self.region_cache[self.class_selected]):
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
                    if self.region_type == "poly":
                        shift_x, shift_y = self.adsorber(x, y, image, np.ceil(2 * self.adsorb_thresh).astype(int))
                        self.point_cache.append((shift_x + x, shift_y + y))
                        self.match = len(self.point_cache) - 1
                    else:
                        if len(self.point_cache) == 0:
                            shift_x, shift_y = self.adsorber(x, y, image, np.ceil(2 * self.adsorb_thresh).astype(int))
                            self.point_cache.append((shift_x + x, shift_y + y))
                            self.match = len(self.point_cache) - 1
                        elif len(self.point_cache) == 1:
                            shift_x, shift_y = self.adsorber(x, y, image, np.ceil(2 * self.adsorb_thresh).astype(int))
                            this_cross = (shift_x + x, shift_y + y)
                            inter_1 = (self.point_cache[0][0], this_cross[1])
                            inter_2 = (this_cross[0], self.point_cache[0][1])
                            self.point_cache.extend([inter_1, this_cross, inter_2])
                            self.match = len(self.point_cache) - 2
                else:
                    if type(self.region_shift) == int:
                        self.region_cache[self.class_selected][self.match][self.region_shift] = self.point_current
        elif event == cv2.EVENT_LBUTTONUP:
            coord_x = np.clip(x, 0, image_size[0] - 1)
            coord_y = np.clip(y, 0, image_size[1] - 1)
            adjusted_pointer = (coord_x, coord_y)
            if self.region_shift is None and self.match is not None:
                # adjusting cached point
                if self.region_type == "poly":
                    if coord_x != x or coord_y != y:
                        self.point_cache.pop(self.match)
                        self.match = None
                    else:
                        self.match = None
                else:
                    if type(self.match) == int:
                        if coord_x != x or coord_y != y:
                            if len(self.point_cache) == 1:
                                self.point_cache.pop(self.match)
                                self.match = None
                            else:
                                # self.point_cache = [self.point_cache[(self.match + 2) % 4]]
                                self.match = None
                        else:
                            self.match = None
                    else:
                        self.point_cache = np.array(self.point_cache) + np.array(adjusted_pointer) - np.array(self.match)
                        self.point_cache = [tuple(_pt) for _pt in self.point_cache]
                        self.match = None
            elif self.region_shift is not None:
                if type(self.region_shift) == int:
                    # adjusting region point
                    if self.region_type == "poly":
                        self.region_cache[self.class_selected][self.match][self.region_shift] = adjusted_pointer
                        self.match = None
                        self.region_shift = None
                    else:
                        this_cross = self.region_cache[self.class_selected][self.match][(self.region_shift + 2) % 4]
                        inter_1 = (adjusted_pointer[0], this_cross[1])
                        inter_2 = (this_cross[0], adjusted_pointer[1])
                        self.region_cache[self.class_selected][self.match][(self.region_shift - 1) % 4] = inter_1
                        self.region_cache[self.class_selected][self.match][(self.region_shift + 1) % 4] = inter_2
                        self.region_cache[self.class_selected][self.match][self.region_shift] = adjusted_pointer
                        self.match = None
                        self.region_shift = None
                else:
                    # adjusting region
                    self.region_cache[self.class_selected][self.match] = np.array(self.region_cache[self.class_selected][self.match]) + np.array(adjusted_pointer) - np.array(self.region_shift)
                    self.region_cache[self.class_selected][self.match] = [tuple(_pt) for _pt in self.region_cache[self.class_selected][self.match]]
                    self.match = None
                    self.region_shift = None
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            coord_x = np.clip(x, 0, image_size[0] - 1)
            coord_y = np.clip(y, 0, image_size[1] - 1)
            adjusted_pointer = (coord_x, coord_y)
            if self.region_shift is None and self.match is not None:
                # adjusting cached point
                if self.region_type == "poly":
                    self.point_cache[self.match] = adjusted_pointer
                else:
                    if type(self.match) == int:
                        if len(self.point_cache) == 1:
                            self.point_cache[self.match] = adjusted_pointer
                        else:
                            this_cross = self.point_cache[(self.match + 2) % 4]
                            inter_1 = (adjusted_pointer[0], this_cross[1])
                            inter_2 = (this_cross[0], adjusted_pointer[1])
                            self.point_cache[(self.match - 1) % 4] = inter_1
                            self.point_cache[(self.match + 1) % 4] = inter_2
                            self.point_cache[self.match] = adjusted_pointer
                    else:
                        self.point_cache = np.array(self.point_cache) + np.array(adjusted_pointer) - np.array(self.match)
                        self.point_cache = [tuple(_pt) for _pt in self.point_cache]
                        self.match = list(adjusted_pointer)
            elif self.region_shift is not None:
                if type(self.region_shift) == int:
                    # adjusting region point
                    if self.region_type == "poly":
                        self.region_cache[self.class_selected][self.match][self.region_shift] = adjusted_pointer
                    else:
                        this_cross = self.region_cache[self.class_selected][self.match][(self.region_shift + 2) % 4]
                        inter_1 = (adjusted_pointer[0], this_cross[1])
                        inter_2 = (this_cross[0], adjusted_pointer[1])
                        self.region_cache[self.class_selected][self.match][(self.region_shift - 1) % 4] = inter_1
                        self.region_cache[self.class_selected][self.match][(self.region_shift + 1) % 4] = inter_2
                        self.region_cache[self.class_selected][self.match][self.region_shift] = adjusted_pointer
                else:
                    # adjusting region
                    self.region_cache[self.class_selected][self.match] = np.array(self.region_cache[self.class_selected][self.match]) + np.array(adjusted_pointer) - np.array(self.region_shift)
                    self.region_cache[self.class_selected][self.match] = [tuple(_pt) for _pt in self.region_cache[self.class_selected][self.match]]
                    self.region_shift = list(adjusted_pointer)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.match is None:
                if self.region_type == "poly":
                    for _i, _pt in enumerate(self.point_cache):
                        if self.dist_func(_pt, self.point_current) <= self.adsorb_thresh:
                            self.point_cache.pop(_i)
                            return
                    for _i, _rg in enumerate(self.region_cache[self.class_selected]):
                        for _j, _pt in enumerate(_rg):
                            if self.dist_func(_pt, self.point_current) <= self.adsorb_thresh:
                                if len(self.region_cache[self.class_selected][_i]) >= 4:
                                    self.region_cache[self.class_selected][_i].pop(_j)
                                    return
                        if cv2.pointPolygonTest(np.array(_rg), self.point_current, True) >= 0:
                            self.region_cache[self.class_selected].pop(_i)
                            return
                else:
                    if len(self.point_cache) == 1:
                        if self.dist_func(self.point_current, self.point_cache[0]) <= self.adsorb_thresh:
                            self.point_cache.pop(0)
                            return
                    else:
                        for _i, _pt in enumerate(self.point_cache):
                            if self.dist_func(_pt, self.point_current) <= self.adsorb_thresh:
                                self.point_cache = [self.point_cache[(_i + 2) % 4]]
                                return
                        if len(self.point_cache) == 4 and cv2.pointPolygonTest(np.array(self.point_cache), self.point_current, True) >= 0:
                            self.point_cache = []
                    for _i, _rg in enumerate(self.region_cache[self.class_selected]):
                        if cv2.pointPolygonTest(np.array(_rg), self.point_current, True) >= 0:
                            self.region_cache[self.class_selected].pop(_i)
                            return

    def __status_handler(self, status):
        if status.lower() == "append":
            if self.region_type == "poly":
                if self.match is not None and type(self.region_shift) != int:
                    cand = (np.array(self.region_cache[self.class_selected][self.match][0]) + np.array(self.region_cache[self.class_selected][self.match][-1])) / 2
                    center = np.mean(self.region_cache[self.class_selected][self.match], axis=0)
                    cand += (cand - center) * 0.1
                    self.region_cache[self.class_selected][self.match].append(tuple(cand.astype(int)))
            return False, None
        elif status.lower() == "add":
            if self.match is None:
                if len(self.point_cache) >= 3:
                    self.region_cache[self.class_selected].append(deepcopy(self.point_cache))
                    self.point_cache = []
            return False, None
        elif status.lower() == "save":
            if self.match is None and len(self.point_cache) == 0:
                rst = deepcopy(self.region_cache)
                self.clear()
                return True, rst
            return False, None

    def render_image(self, img, status_dict, task_name, wait=5, pt_color=(0, 255, 0), render_layers={"selected": [0.5, 2], "background": [0.3, 1]}):
        if img.shape[2] == 3:
            cv2.setMouseCallback(self.window_name, self.__mouse_handler, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        elif img.shape[2] == 1:
            cv2.setMouseCallback(self.window_name, self.__mouse_handler, img)
        else:
            raise NotImplementedError
        # render part
        for _c, _rgs in self.region_cache.items():
            if _c == self.class_selected:
                transparency = render_layers["selected"][0]
                thicky = np.ceil(self.adsorb_thresh).astype(int)
                radius = 1 + np.ceil(self.adsorb_thresh).astype(int)
                img = polygon_marker(img, self.point_cache, pt_color, thicky, transparency, True, radius, False)
            else:
                transparency = render_layers["background"][0]
                thicky = render_layers["background"][1]
                radius = render_layers["background"][1]
            for _j, _rg in enumerate(_rgs):
                img = polygon_marker(img, _rg, self.render_dict[_c], thicky, transparency, True, radius, True)
        text = "{}:{} pt:{} rg:{}".format(task_name, self.class_dict[self.class_selected], len(self.point_cache), len(self.region_cache[self.class_selected]))
        text_marker(img, text, (0, 0), (0, 0), FONT, FONTSCALE, THICKNESS, self.render_dict[self.class_selected], background=(255, 255, 255), bgthick=0.3)
        img = self.__patch_legend(img)
        cv2.imshow(self.window_name, img)
        pressed_key = cv2.waitKey(wait) & 0xFF
        if self.match is None:
            for k, v in self.class_dict.items():
                if pressed_key == ord(str(k)):
                    self.class_selected = k
                    return False, None
        for k, v in status_dict.items():
            if pressed_key == ord(k):
                flag, rst = self.__status_handler(v)
                if flag:
                    print("recorded: {}".format(rst))
                    return flag, rst
        return False, None

    def clear(self):
        self.point_cache = []
        self.point_current = (-1, -1)
        self.match = None
        self.adsorb_thresh = np.mean(self.window_size) * self.adsorb_ratio
        self.region_shift = None
        self.class_selected = self.class_list[0]
        self.region_cache = {k: [] for k in self.class_dict.keys()}


if __name__ == '__main__':
    window_name = "Elementary Labeler"
    window_size = (1000, 800)
    class_dict = {0: "people", 1: "dog", 2: "cat"}
    render_dict = {0: (139, 236, 255), 1: (58, 58, 139), 2: (226, 43, 138)}
    labeler = ClassifiedDetectionLabeler(window_name, window_size, False, region_type="rect", class_dict=class_dict, render_dict=render_dict)

    img_path = "/home/chenzhou/Pictures/Concept/python-package.webp"
    colorful = cv2.imread(img_path)
    status_dict = {"a": "APPEND", "s": "SAVE", "d": "ADD"}
    while True:
        labeler.render_image(colorful.copy(), status_dict, "test")
