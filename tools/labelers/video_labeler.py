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
from algorithms.Hungarian import Hungarian
from multiprocessing import Process, Manager
from copy import deepcopy
import urllib.request
import getpass

"""
this Labeler should be working with a Labeler Server to unpack the videos
"""

FINISH_FLAG_FILE = "FINISHED.json"


class VideoLabeler(object):
    def __init__(self, image_labeler, cache_dict, interp_list, lock, prefix, img_ext=".png", cache_length=100, load_task_length=500, step_jump=50, progress_trackbar_prefix='Progress Cuts x', cache_trackbar_name='Frame Cache +'):
        self.labeler = image_labeler
        self.img_ext = img_ext
        self.cache_length = cache_length
        self.load_task_length = load_task_length
        self.step_jump = step_jump
        self.progress_trackbar_prefix = progress_trackbar_prefix
        self.cache_trackbar_name = cache_trackbar_name
        self.cache_dict = cache_dict
        self.interp_list = interp_list
        self.interp_list["index"] = []
        self.lock = lock
        self.empty = (np.ones_like(self.labeler.legend) * 255).astype(np.uint8)
        self.prefix = prefix

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
                        pass
                        # rst[b["category_id"]] = [[lt, rt, rd, ld]]
                return rst

            def __load_cache(image_path, image_list, label_path, frame_index, cache_length, cache_dict, interp_list, lock):
                with lock:
                    tmp_keys = np.array(list(cache_dict.keys()))
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
                                order_list = np.array(interp_list["index"])
                                indeces_ = np.argwhere(frame_index + _diff <= order_list)
                                if len(indeces_) > 0:
                                    last_index_ = indeces_[0][0]
                                    interp_list["index"] = order_list[:last_index_].tolist() + [frame_index + _diff] + order_list[last_index_:].tolist()
                                else:
                                    interp_list["index"] = order_list.tolist() + [frame_index + _diff]
                    else:
                        break
                with lock:
                    for _t in rmv_indeces:
                        cache_dict.pop(_t)
                        try:
                            tmp = interp_list["index"]
                            tmp.remove(_t)
                            interp_list["index"] = tmp
                        except Exception:
                            pass
            loading = Process(target=__load_cache, args=(self.source_path, self.image_list, self.target_path, frame_index * self.total_jump, self.total_cache, self.cache_dict, self.interp_list, self.lock))
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
            def __calc_bbox_dist(bboxlist1, bboxlist2, interp):
                cost_mat = np.zeros([len(bboxlist1), len(bboxlist2)])
                bboxlist1 = bboxlist1
                bboxlist2 = bboxlist2
                for _i, _b1 in enumerate(bboxlist1):
                    lt_1, rd_1 = np.min(_b1, axis=0), np.max(_b1, axis=0)
                    for _j, _b2 in enumerate(bboxlist2):
                        lt_2, rd_2 = np.min(_b2, axis=0), np.max(_b2, axis=0)
                        ratio = (rd_2 - lt_2) / (rd_1 - lt_1)
                        cost_mat[_i, _j] = np.linalg.norm(((rd_1 + lt_1) - (rd_2 + lt_2)) / 2) * (1 + np.abs(np.log(ratio)).mean())
                hungarian = Hungarian()
                hungarian.calculate(cost_mat)
                paired = hungarian.get_results()
                rst_bbox = []
                for _p in paired:
                    _b1 = bboxlist1[_p[0]]
                    _b2 = bboxlist2[_p[1]]
                    lt_1, rd_1 = np.min(_b1, axis=0), np.max(_b1, axis=0)
                    lt_2, rd_2 = np.min(_b2, axis=0), np.max(_b2, axis=0)
                    lt_ = (lt_1 * (1 - interp) + lt_2 * interp).astype(int).tolist()
                    rd_ = (rd_1 * (1 - interp) + rd_2 * interp).astype(int).tolist()
                    new = [tuple(lt_), (lt_[0], rd_[1]), tuple(rd_), (rd_[0], lt_[1])]
                    rst_bbox.append(new)
                return rst_bbox

            frame_index += cv2.getTrackbarPos(self.progress_trackbar_name, self.labeler.window_name) * self.total_jump
            if frame_index >= self.total_frames:
                cv2.setTrackbarPos(self.cache_trackbar_name, self.labeler.window_name, self.last_progress_cache - 1)
                return
            with self.lock:
                if "label" in self.cache_dict[frame_index].keys():
                    label = self.cache_dict[frame_index]["label"]
                    self.labeler.region_cache = deepcopy(label)
                    self.labeler.prediction_status = False
                else:
                    ready = np.array(self.interp_list["index"])
                    index_ = np.argwhere(frame_index >= ready)
                    self.labeler.prediction_status = True
                    if 0 < len(index_) < len(ready):
                        begin_ = ready[index_[-1][0]]
                        end_ = ready[index_[-1][0] + 1]
                        interp_factor = (frame_index - begin_) / (end_ - begin_)
                        interp_1 = self.cache_dict[begin_]["label"]
                        interp_2 = self.cache_dict[end_]["label"]
                        rst = {}
                        for k in self.labeler.class_dict.keys():
                            interp_1_bboxes = interp_1[k]
                            interp_2_bboxes = interp_2[k]
                            if len(interp_1_bboxes) > 0 and len(interp_2_bboxes) > 0:
                                rst_bbox = __calc_bbox_dist(interp_1_bboxes, interp_2_bboxes, interp_factor)
                                rst[k] = rst_bbox
                            else:
                                rst[k] = []
                        self.labeler.region_cache = deepcopy(rst)
                    else:
                        self.labeler.region_cache = {k: [] for k in self.labeler.class_dict.keys()}

        cv2.createTrackbar(self.cache_trackbar_name, self.labeler.window_name, 0, self.total_cache - 1, __TrackbarRender)
        cv2.createTrackbar(self.progress_trackbar_name, self.labeler.window_name, 0, self.progress_points - 1, __TrackbarLoading)
        cv2.setTrackbarPos(self.progress_trackbar_name, self.labeler.window_name, 1)
        cv2.setTrackbarPos(self.progress_trackbar_name, self.labeler.window_name, 0)

    def record_label(self, label_dict, image_abs_path):
        img_dict = {"image_name": image_abs_path[len(self.prefix):].strip("/"),
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
                with self.lock:
                    image = self.cache_dict[frame_index]["image"]
                flag, rst = self.labeler.render_image(image.copy(), status_dict, self.task_name)
                if flag:
                    with self.lock:
                        tmp = self.cache_dict[frame_index]
                        tmp["label"] = rst
                        self.cache_dict[frame_index] = tmp
                        if frame_index not in self.interp_list["index"]:
                            order_list = np.array(self.interp_list["index"])
                            indeces_ = np.argwhere(frame_index <= order_list)
                            if len(indeces_) > 0:
                                last_index_ = indeces_[0][0]
                                self.interp_list["index"] = order_list[:last_index_].tolist() + [frame_index] + order_list[last_index_:].tolist()
                            else:
                                self.interp_list["index"] = order_list.tolist() + [frame_index]
                    json_rst = self.record_label(rst, os.path.join(self.source_path, self.image_list[frame_index]))
                    fname_id, ext = os.path.splitext(self.image_list[frame_index])
                    with open(os.path.join(self.target_path, fname_id + ".json"), "w") as f:
                        f.write(json.dumps(json_rst))
                    if cache_pt < self.total_cache - 1:
                        cv2.setTrackbarPos(self.cache_trackbar_name, self.labeler.window_name, cache_pt + 1)
                    elif progress_pt < self.progress_points - 1:
                        cv2.setTrackbarPos(self.progress_trackbar_name, self.labeler.window_name, progress_pt + 1)
                        cv2.setTrackbarPos(self.cache_trackbar_name, self.labeler.window_name, self.total_cache - self.total_jump)
            else:
                self.labeler.render_image(self.empty.copy(), status_dict, "LOADING......")
                time.sleep(1)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='HOST:PORT RestartTask RestartCache MountPrefix')
    parser.add_argument("--host", type=str, required=True, help="HOST:PORT")
    parser.add_argument("--task", type=int, required=True, help="1/0")
    parser.add_argument("--cache", type=int, required=True, help="1/0")
    parser.add_argument("--prefix", type=str, required=True, help="MountPrefix /data")

    args = parser.parse_args()

    window_name = "Elementary Labeler"
    window_size = (1600, 1200)

    HOST_PORT = args.host
    RESTART_TASK = args.task
    RESTART_CACHE = args.cache
    PREFIX = args.prefix

    assignment_url = HOST_PORT + "/assignment?user=" + getpass.getuser() + "&restart=" + str(RESTART_TASK)
    while(True):
        response = urllib.request.urlopen(assignment_url)
        data = json.loads(response.read().decode("utf-8"))
        if len(data["tasks"]) > 0:
            IMAGE_ROOT = os.path.join(PREFIX, data["image_root"][len(data["prefix"]):].strip('/'), data["tasks"][0].strip('/'))
            LABEL_ROOT = os.path.join(PREFIX, data["label_root"][len(data["prefix"]):].strip('/'), data["tasks"][0].strip('/'))
            print(IMAGE_ROOT, LABEL_ROOT)
            render_dict = data["render_dict"]
            class_dict = data["class_dict"]
            status_dict = data["status_dict"]
            break
        else:
            print("not getting task, waiting 10 sec: {}".format(data))
            time.sleep(10)

    MGR = Manager()
    CACHE_DICT = MGR.dict()
    INTERP_LIST = MGR.dict()
    LOCK = MGR.Lock()

    labeler = ClassifiedDetectionLabeler(window_name, window_size, False, region_type="rect", class_dict=class_dict, render_dict=render_dict, status_dict=status_dict)
    vl = VideoLabeler(labeler, CACHE_DICT, INTERP_LIST, LOCK, PREFIX)
    vl.assign_task(IMAGE_ROOT, LABEL_ROOT, RESTART_CACHE)
    vl.start_label(status_dict)
