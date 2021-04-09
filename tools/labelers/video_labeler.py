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
from copy import deepcopy
import urllib.request
import getpass

"""
this Labeler should be working with a Labeler Server to unpack the videos
"""

FINISH_FLAG_FILE = "FINISHED.json"


class VideoLabeler(object):
    def __init__(self, image_labeler, cache_dict, lock, img_ext=".png", cache_length=100, load_task_length=200, step_jump=30, progress_trackbar_prefix='Progress Cuts x', cache_trackbar_name='Frame Cache +'):
        self.labeler = image_labeler
        self.img_ext = img_ext
        self.cache_length = cache_length
        self.load_task_length = load_task_length
        self.step_jump = step_jump
        self.progress_trackbar_prefix = progress_trackbar_prefix
        self.cache_trackbar_name = cache_trackbar_name
        self.cache_dict = cache_dict
        self.lock = lock
        self.empty = (np.ones_like(self.labeler.legend) * 255).astype(np.uint8)

    def assign_task(self, source_path, target_path, restart):
        image_list_tmp = os.listdir(source_path)
        self.image_list = []
        for f in image_list_tmp:
            fname_, ext_ = os.path.splitext(f)
            if ext_ == self.img_ext:
                self.image_list.append(f)
        self.image_list = sorted(self.image_list, key=lambda x: int(os.path.splitext(x)[0]))
        with open(os.path.join(source_path, FINISH_FLAG_FILE), "r") as f:
            tmp = json.loads(f.read())
        if not restart or tmp["last_visit_index"] == 0:
            if tmp["last_visit_index"] + self.load_task_length * 2 >= len(self.image_list):
                self.image_list = self.image_list[tmp["last_visit_index"]:]
                tmp["last_visit_index"] = 0
            else:
                self.image_list = self.image_list[tmp["last_visit_index"]:(tmp["last_visit_index"] + self.load_task_length)]
                tmp["last_visit_index"] += self.load_task_length
            with open(os.path.join(source_path, FINISH_FLAG_FILE), "w") as f:
                f.write(json.dumps(tmp))
        else:
            self.image_list = self.image_list[(tmp["last_visit_index"] - self.load_task_length):tmp["last_visit_index"]]

        self.target_path = target_path
        self.source_path = source_path
        dirpath, fname = os.path.split(source_path)
        self.task_name = fname

        self.total_frames = len(self.image_list)
        self.total_cache = min(self.cache_length, self.total_frames - 1)
        self.total_jump = min(self.step_jump, self.total_cache - 1)
        self.progress_trackbar_name = self.progress_trackbar_prefix + str(self.total_jump)
        assert self.total_jump >= 1
        self.progress_points = np.ceil((self.total_frames - self.total_cache) / self.total_jump).astype(int) + 1
        self.last_progress_cache = self.total_frames - (self.progress_points - 1) * self.total_jump

        def __TrackbarLoading(frame_index):
            def __transform_label(jsondict, class_dict):
                rst = {}
                for k in class_dict.keys():
                    rst[k] = []
                bbox_list = jsondict["annotations"]
                for b in bbox_list:
                    lt = tuple(b["bbox"][:2])
                    rd = tuple(b["bbox"][2:])
                    rt = (rd[0], lt[1])
                    ld = (lt[0], rd[1])
                    if b["category_id"] in rst:
                        rst[b["category_id"]].append([lt, rt, rd, ld])
                    else:
                        rst[b["category_id"]] = [[lt, rt, rd, ld]]
                return rst

            def __load_cache(image_path, image_list, label_path, frame_index, cache_length, cache_dict, lock):
                with lock:
                    tmp_keys = np.array((cache_dict.keys()))
                if frame_index not in tmp_keys:
                    adding_order = range(cache_length)
                    rmv_indeces = tmp_keys[tmp_keys >= frame_index + cache_length]
                else:
                    adding_order = range(cache_length - 1, -1, -1)
                    rmv_indeces = tmp_keys[tmp_keys < frame_index]
                for _diff in adding_order:
                    if frame_index + _diff >= self.total_frames:
                        continue
                    if frame_index + _diff not in tmp_keys:
                        img = cv2.imread(os.path.join(image_path, image_list[frame_index + _diff]))
                        with lock:
                            cache_dict[frame_index + _diff] = {"image": img}
                        label_name = os.path.join(label_path, os.path.splitext(image_list[frame_index + _diff])[0] + ".json")
                        if os.path.exists(label_name):
                            with open(label_name, "r") as f:
                                lbl = __transform_label(json.loads(f.read()), self.labeler.class_dict)
                            with lock:
                                tmp = cache_dict[frame_index + _diff]
                                tmp["label"] = lbl
                                cache_dict[frame_index + _diff] = tmp
                    else:
                        break
                with lock:
                    for _t in rmv_indeces:
                        cache_dict.pop(_t)
            loading = Process(target=__load_cache, args=(self.source_path, self.image_list, self.target_path, frame_index * self.total_jump, self.total_cache, self.cache_dict, self.lock))
            loading.start()
            loading.join()
            if frame_index < self.progress_points - 1:
                now_cache_at = cv2.getTrackbarPos(self.cache_trackbar_name, self.labeler.window_name)
                if now_cache_at < self.total_cache - 1:
                    cv2.setTrackbarPos(self.cache_trackbar_name, self.labeler.window_name, now_cache_at + 1)
                else:
                    cv2.setTrackbarPos(self.cache_trackbar_name, self.labeler.window_name, now_cache_at - 1)
                cv2.setTrackbarPos(self.cache_trackbar_name, self.labeler.window_name, now_cache_at)
            else:
                cv2.setTrackbarPos(self.cache_trackbar_name, self.labeler.window_name, 0)

        def __TrackbarDeactivate(num):
            pass

        def __TrackbarRender(frame_index):
            frame_index += cv2.getTrackbarPos(self.progress_trackbar_name, self.labeler.window_name) * self.total_jump
            if frame_index >= self.total_frames:
                cv2.setTrackbarPos(self.cache_trackbar_name, self.labeler.window_name, self.last_progress_cache - 1)
                return
            with self.lock:
                if "label" in self.cache_dict[frame_index].keys():
                    label = self.cache_dict[frame_index]["label"]
                    self.labeler.region_cache = deepcopy(label)
                else:
                    self.labeler.region_cache = {k: [] for k in self.labeler.class_dict.keys()}

        cv2.createTrackbar(self.cache_trackbar_name, self.labeler.window_name, 0, self.total_cache - 1, __TrackbarRender)
        cv2.createTrackbar(self.progress_trackbar_name, self.labeler.window_name, 0, self.progress_points - 1, __TrackbarLoading)
        cv2.setTrackbarPos(self.progress_trackbar_name, self.labeler.window_name, 1)
        cv2.setTrackbarPos(self.progress_trackbar_name, self.labeler.window_name, 0)

    def record_label(self, label_dict, image_abs_path):
        img_dict = {"image_name": image_abs_path,
                    "annotations": []}
        for k, v in label_dict.items():
            category_name = self.labeler.class_dict[k]
            for b in v:
                bbox = np.array(b)
                left_top = bbox.min(axis=0).tolist()
                right_down = bbox.max(axis=0).tolist()
                item = {"bbox": left_top + right_down,
                        "category_id": k,
                        "name": category_name
                        }
                img_dict["annotations"].append(item)
        return img_dict

    def start_label(self, status_dict):
        self.labeler.render_image(self.empty.copy(), status_dict, "LOADING...")
        while True:
            progress_pt = cv2.getTrackbarPos(self.progress_trackbar_name, self.labeler.window_name)
            cache_pt = cv2.getTrackbarPos(self.cache_trackbar_name, self.labeler.window_name)
            frame_index = progress_pt * self.total_jump + cache_pt
            if frame_index in self.cache_dict.keys():
                image = self.cache_dict[frame_index]["image"]
                flag, rst = self.labeler.render_image(image.copy(), status_dict, self.task_name)
                if flag:
                    tmp = self.cache_dict[frame_index]
                    tmp["label"] = rst
                    self.cache_dict[frame_index] = tmp
                    json_rst = self.record_label(rst, os.path.join(self.source_path, self.image_list[frame_index]))
                    fname_id, ext = os.path.splitext(self.image_list[frame_index])
                    with open(os.path.join(self.target_path, fname_id + ".json"), "w") as f:
                        f.write(json.dumps(json_rst))
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

    HOST_PORT = "http://0.0.0.0:1207"
    RESTART_TASK = 0
    RESTART_CACHE = 0

    assignment_url = HOST_PORT + "/assignment?user=" + getpass.getuser() + "&restart=" + str(RESTART_TASK)
    while(True):
        response = urllib.request.urlopen(assignment_url)
        data = json.loads(response.read().decode("utf-8"))
        if len(data["tasks"]) > 0:
            IMAGE_ROOT = os.path.join(data["image_root"], data["tasks"][0])
            LABEL_ROOT = os.path.join(data["label_root"], data["tasks"][0])
            render_dict = data["render_dict"]
            class_dict = data["class_dict"]
            status_dict = data["status_dict"]
            break
        else:
            print("not getting task, waiting 10 sec: {}".format(data))
            time.sleep(10)

    MGR = Manager()
    CACHE_DICT = MGR.dict()
    LOCK = MGR.Lock()

    labeler = ClassifiedDetectionLabeler(window_name, window_size, False, region_type="rect", class_dict=class_dict, render_dict=render_dict)
    vl = VideoLabeler(labeler, CACHE_DICT, LOCK)
    vl.assign_task(IMAGE_ROOT, LABEL_ROOT, RESTART_CACHE)
    vl.start_label(status_dict)
