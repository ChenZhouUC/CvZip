import cv2


def edge_cropper(img, width, height):
    assert width > 0 and height > 0
    asp_ratio = height / width
    h, w = img.shape[:2]
    asp_ratio_ori = h / w
    if asp_ratio >= asp_ratio_ori:
        # width should be cropped
        new_w = round(h / asp_ratio)
        left_ = (w - new_w) // 2
        new_img = cv2.resize(img[:, left_:(left_ + new_w)], (width, height))
    else:
        # height should be cropped
        new_h = round(w * asp_ratio)
        left_ = (h - new_h) // 2
        new_img = cv2.resize(img[left_:(left_ + new_h), :], (width, height))
    return new_img


if __name__ == "__main__":
    img_path = "/home/chenzhou/Pictures/b.jpg"
    out_path = "/home/chenzhou/Downloads/cad_crop.png"

    img = cv2.imread(img_path)
    print(img.shape)
    new_img = edge_cropper(img, 1500, 300)
    cv2.imshow("winname", new_img)
    cv2.waitKey(0)
