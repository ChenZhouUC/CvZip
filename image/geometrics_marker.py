import cv2
import numpy as np


def polygon_marker(img, polygon, color, thickness, fill, point, radius, closed, epsilon=0.1):
    if len(polygon) == 0:
        return img
    polygon = np.array(polygon).round().astype(int)
    fill = np.clip(fill, 0.0, 1.0)
    if len(polygon.shape) == 2:
        assert polygon.shape[1] == 2, "coordinates must be of dim 2."
    elif len(polygon.shape) == 3:
        assert polygon.shape[2] == 2 and polygon.shape[1] == 1, "coordinates must be of dim 1 x 2"
    else:
        raise NotImplementedError
    if fill > 0:
        mask = np.zeros_like(img)
        masking = cv2.drawContours(mask.copy(), [polygon], 0, color, -1)
        fused = cv2.addWeighted(img, 1 - fill, masking, fill, 0)
        mask = cv2.drawContours(mask, [polygon], 0, 1, -1)
        locs = np.where(mask != 0)
        img[locs[0], locs[1]] = fused[locs[0], locs[1]]
    if thickness < 0:
        marked = cv2.drawContours(img, [polygon], 0, color, thickness)
    elif thickness == 0:
        marked = img
    else:
        marked = cv2.polylines(img, [polygon], closed, color, thickness)
    if point > 0:
        if point < 1:
            # aspect_length = cv2.arcLength(polygon, closed=closed) / 4
            aspect_length = np.sqrt(cv2.contourArea(polygon))
            polygon_simplified = cv2.approxPolyDP(polygon, aspect_length * epsilon, closed=closed)
            for p in polygon_simplified:
                img = cv2.circle(img, tuple(p[0]), radius, color, -1)
        else:
            for p in polygon:
                p = p.reshape(2)
                img = cv2.circle(img, tuple(p), radius, color, -1)
    return marked


def rectangle_marker(img, rectangle, color, thickness, fill, point, radius, epsilon=0.1):
    left, top = rectangle[0]
    right, down = rectangle[1]
    polygonize = [[left, top], [right, top], [right, down], [left, down]]
    return polygon_marker(img, polygonize, color, thickness, fill, point, radius, True, epsilon)


def text_marker(img, text, coordinate, locator, font, fontscale, thickness, color, background=None, bgthick=0.5):
    text_list = text.split("\n")
    if len(text_list) == 0:
        return img
    label_width_max = 0
    for text in text_list:
        (label_width, label_height), baseline = cv2.getTextSize(text, font, fontscale, thickness)
        label_width_max = max(label_width_max, label_width)
    locator_x, locator_y = np.array(locator)[:2] * np.array([label_width_max, (label_height + baseline) * len(text_list)])
    if background is not None:
        left_top_corner = [coordinate[0] - locator_x, coordinate[1] - locator_y]
        right_down_corner = [coordinate[0] - locator_x + label_width_max, coordinate[1] - locator_y + (label_height + baseline) * len(text_list)]
        img = rectangle_marker(img, [left_top_corner, right_down_corner], background, thickness, bgthick, False, None)
    for _i, text in enumerate(text_list):
        shift_ = np.array([-locator_x, (label_height + baseline) * _i + label_height - locator_y])
        coordinate_ = tuple((np.array(coordinate) + shift_).round().astype(int))
        cv2.putText(img, text, coordinate_, font, fontscale, color, thickness)
    return img


if __name__ == "__main__":

    img_path = "/home/chenzhou/Pictures/Concept/python-package.webp"
    colorful = cv2.imread(img_path)
    c = [(150, 150), (800, 800)]
    colorful = rectangle_marker(colorful, c, (0, 255, 0), 2, fill=0.5, point=0.5, radius=5)
    coord = (150, 150)
    # cv2.circle(colorful, coord, 5, (0, 255, 255), -1)
    text_marker(colorful, "draw\ntest\n newtest", coord, (0.5, 0.5), cv2.FONT_HERSHEY_TRIPLEX, 1, 1, (0, 0, 0), background=(255, 255, 255), bgthick=0.3)
    cv2.imshow("winname", colorful)
    cv2.waitKey(0)
