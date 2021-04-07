import cv2
import numpy as np
import time
import json
import os
import sys
module_ = os.path.abspath(__file__)
for layer_ in range(3):
    module_ = os.path.dirname(module_)
sys.path.append(module_)
from classified_detection_labeler import ClassifiedDetectionLabeler
from multiprocessing import Process, Manager

"""
this Labeler should be working with a Labeler Server to unpack the videos
"""


class VideoLabeler(object):
    def __init__(self, image_labeler, cache_dict, lock, img_ext=".png", cache_length=100, progress_trackbar_name='Progress Cuts @', cache_trackbar_name='Frame Cache +'):
        self.labeler = image_labeler
        self.img_ext = img_ext
        self.cache_length = cache_length
        self.progress_trackbar_name = progress_trackbar_name
        self.cache_trackbar_name = cache_trackbar_name
        self.cache_dict = cache_dict
        self.lock = lock
        self.empty = (np.ones_like(self.labeler.legend) * 255).astype(np.uint8)

    def assign_task(self, source_path, target_path):
        image_list_tmp = os.listdir(source_path)
        self.image_list = []
        for f in image_list_tmp:
            fname_, ext_ = os.path.splitext(f)
            if ext_ == self.img_ext:
                self.image_list.append(f)
        self.image_list = sorted(self.image_list, key=lambda x: int(os.path.splitext(x)[0]))
        self.target_path = target_path
        self.source_path = source_path
        dirpath, fname = os.path.split(source_path)
        self.task_name = fname

        self.total_frames = len(self.image_list)
        self.total_cache = min(self.cache_length, self.total_frames)
        self.progress_points = self.total_frames - (self.total_cache - 1)

        def __TrackbarLoading(frame_index):
            def __LoadCache(image_path, image_list, frame_index, cache_length, cache_dict, lock):
                with lock:
                    tmp_keys = list(cache_dict.keys())
                for _diff in range(cache_length):
                    if frame_index + _diff not in tmp_keys:
                        img = cv2.imread(os.path.join(image_path, image_list[frame_index + _diff]))
                        # lbl = json.loads(s: _LoadsString)
                        with lock:
                            cache_dict[frame_index + _diff] = {"image": img}
                    else:
                        tmp_keys.remove(frame_index + _diff)
                with lock:
                    for _t in tmp_keys:
                        cache_dict.pop(_t)
            loading = Process(target=__LoadCache, args=(self.source_path, self.image_list, frame_index, self.total_cache, self.cache_dict, self.lock))
            loading.start()
            loading.join()

        def __TrackbarDeactivate(num):
            pass

        cv2.createTrackbar(self.cache_trackbar_name, self.labeler.window_name, 0, self.total_cache - 1, __TrackbarDeactivate)
        cv2.createTrackbar(self.progress_trackbar_name, self.labeler.window_name, 0, self.progress_points - 1, __TrackbarLoading)
        cv2.setTrackbarPos(self.progress_trackbar_name, self.labeler.window_name, 1)
        cv2.setTrackbarPos(self.progress_trackbar_name, self.labeler.window_name, 0)

    def start_label(self, status_dict):
        self.labeler.render_image(self.empty.copy(), status_dict, "LOADING...")
        while True:
            progress_pt = cv2.getTrackbarPos(self.progress_trackbar_name, self.labeler.window_name)
            cache_pt = cv2.getTrackbarPos(self.cache_trackbar_name, self.labeler.window_name)
            frame_index = progress_pt + cache_pt
            if frame_index in self.cache_dict.keys():
                image = self.cache_dict[frame_index]["image"]
                flag, rst = self.labeler.render_image(image.copy(), status_dict, self.task_name)
                if flag:
                    print("check the rst", rst)
                    if cache_pt < self.total_cache - 1:
                        cv2.setTrackbarPos(self.cache_trackbar_name, self.labeler.window_name, cache_pt + 1)
                    elif progress_pt < self.progress_points - 1:
                        cv2.setTrackbarPos(self.progress_trackbar_name, self.labeler.window_name, progress_pt + 1)
            else:
                self.labeler.render_image(self.empty.copy(), status_dict, "LOADING......")
                time.sleep(1)


if __name__ == '__main__':
    window_name = "Elementary Labeler"
    window_size = (1600, 1200)
    class_dict = {0: "people", 1: "dog", 2: "cat"}
    render_dict = {0: (139, 236, 255), 1: (58, 58, 139), 2: (226, 43, 138)}
    status_dict = {"a": "APPEND", "s": "SAVE", "d": "ADD", "q": "QUIT"}

    MGR = Manager()
    CACHE_DICT = MGR.dict()
    LOCK = MGR.Lock()

    labeler = ClassifiedDetectionLabeler(window_name, window_size, False, region_type="rect", class_dict=class_dict, render_dict=render_dict)
    vl = VideoLabeler(labeler, CACHE_DICT, LOCK)
    vl.assign_task("/mnt/cv/usr/chenzhou/Labeler/image/01.mp4", "/mnt/cv/usr/chenzhou/Labeler/label/01.mp4")
    vl.start_label(status_dict)
