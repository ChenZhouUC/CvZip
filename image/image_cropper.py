import cv2
import numpy as np

if __name__ == "__main__":
    img = "/home/chenzhou/Pictures/b.jpg"
    out = "/home/chenzhou/Downloads/cad_crop.png"

    img = cv2.imread(img)
    print(img.shape)
    print(np.mean(img.shape[:2]))
    # img = cv2.resize(img, (900, 657))
    # cv2.imwrite(out, img)