import cv2
import numpy as np
from multiprocessing import Process
import time
import os
import sys
module_ = os.path.abspath(__file__)
for layer_ in range(3):
    module_ = os.path.dirname(module_)
sys.path.append(module_)
from video.video_generator import VideoGenerator
from detection_labeler import DetectionLabeler


class VideoLabeler(object):
    def __init__(self, image_labeler, video_generator, target_dir, img_ext, cache_length=5, trackbar_name='Frame'):
        self.labeler = image_labeler
        self.video = video_generator
        self.target_dir = target_dir
        self.img_ext = img_ext
        self.result = {}
        self.process = []

        self.frame_cache = {}
        self.trackbar_name = trackbar_name

        def __TrackbarLoading(frame_index):
            start = np.clip(frame_index - cache_length, 0, self.video.input_frames() - 1).astype(int)
            end = np.clip(frame_index + cache_length, 0, self.video.input_frames() - 1).astype(int)
            for _f in range(start, end + 1):
                if _f not in self.frame_cache.keys():
                    img_path = os.path.join(self.target_dir, self.video.input_identity(), "images", str(_f) + self.img_ext)
                    while True:
                        frame_tmp = cv2.imread(img_path)
                        if frame_tmp is not None:
                            self.frame_cache[_f] = frame_tmp
                            break
                        else:
                            time.sleep(1)
                            print("retry obtain :{}".format(img_path))
            print(self.frame_cache.keys())

        cv2.createTrackbar(self.trackbar_name, self.labeler.window_name, 0, self.video.input_frames() - 1, __TrackbarLoading)
        cv2.setTrackbarPos(self.trackbar_name, self.labeler.window_name, 1)

    def start_label(self, status_dict):
        while True:
            frame_index = cv2.getTrackbarPos(self.trackbar_name, self.labeler.window_name)
            print(frame_index)
            if frame_index in self.frame_cache.keys():
                print(123)
                image = self.frame_cache[frame_index]
                cv2.imshow("winname", image)
                cv2.waitKey(0)
                flag, rst = self.labeler.render_image(image.copy(), status_dict)
                if flag:
                    print("check the rst", rst)
            else:
                time.sleep(1)

    def start(self, status_dict):
        labeling = Process(target=self.start_label, args=(status_dict,))
        labeling.start()
        # self.video.split_frames(self.target_dir, self.img_ext)
        labeling.join()


if __name__ == '__main__':
    window_name = "Elementary Labeler"
    window_size = (1600, 1200)
    video_source = "/mnt/cv/usr/chenzhou/synopsis/video_samples/190625_26_RainyStreets_HD_29.mp4"
    target_dir = "/mnt/cv/usr/chenzhou/synopsis/label_test"
    img_ext = ".png"
    status_dict = {"a": "APPEND", "s": "SAVE", "d": "ADD"}

    video_generator = VideoGenerator(video_source, False)
    labeler = DetectionLabeler(window_name, window_size, False, adsorb_ratio=0.003, region_type="rect")
    vl = VideoLabeler(labeler, video_generator, target_dir, img_ext)
    vl.start(status_dict)
