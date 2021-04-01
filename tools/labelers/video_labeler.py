import os
import sys
module_ = os.path.abspath(__file__)
for layer_ in range(3):
    module_ = os.path.dirname(module_)
sys.path.append(module_)
from video.video_generator import VideoGenerator
from detection_labeler import DetectionLabeler


class VideoLabeler(object):
    def __init__(self, image_labeler, video_generator):
        self.labeler = image_labeler
        self.video = video_generator
        self.result = {}

    def start(self):
        counter = 0
        while True:
            ret, frame = self.video.read_frame()
            if ret:
                while True:
                    flag, rst = self.labeler.render_image(frame.copy(), status_dict)
                    if flag:
                        self.result[counter] = rst
                        counter += 1
                        break
            else:
                break


if __name__ == '__main__':
    window_name = "Elementary Labeler"
    window_size = (1600, 1200)
    video_source = "/mnt/cv/usr/chenzhou/synopsis/video_samples/1617080400392-1617084002678-2080040.mp4"
    status_dict = {"a": "APPEND", "s": "SAVE", "d": "ADD"}

    video_generator = VideoGenerator(video_source, False)
    labeler = DetectionLabeler(window_name, window_size, False, adsorb_ratio=0.003, region_type="rect")
    vl = VideoLabeler(labeler, video_generator)
    vl.start()
