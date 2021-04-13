import cv2
import os
import numpy as np
import sys
module_ = os.path.abspath(__file__)
for layer_ in range(2):
    module_ = os.path.dirname(module_)
sys.path.append(module_)
from tools.progress_bar import progressbar_urlretrieve


def decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])


class VideoGenerator:
    def __init__(self, video_input, visual=False):
        self._input = video_input
        self._input_name = os.path.basename(self._input)
        self._input_identity, self._input_ext = os.path.splitext(self._input_name)
        self._reader = cv2.VideoCapture(self._input)
        self._input_width = int(self._reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._input_height = int(self._reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._input_frames = int(self._reader.get(cv2.CAP_PROP_FRAME_COUNT))
        self._input_fps = int(self._reader.get(cv2.CAP_PROP_FPS))
        self._input_fourcc = decode_fourcc(self._reader.get(cv2.CAP_PROP_FOURCC))
        self.fourcc = None
        self._reader_counter = 0
        print("[VideoSize]:{}x{}x{} [FPS]:{} [FOURCC]:{}".format(self._input_frames, self._input_width, self._input_height, self._input_fps, self._input_fourcc))
        if visual:
            while self._reader.isOpened():
                ret, frame = self._reader.read()
                self._reader_counter += 1
                if ret:
                    cv2.imshow("VISUAL", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def __del__(self):
        self._reader.release()

    def __reinit_reader(self):
        self._reader = cv2.VideoCapture(self._input)
        self._reader_counter = 0

    def __reshape_frame(self, frame, shape_width, shape_height, method="resize", crop_locator=None):
        # this function is necessary since different size designated between VideoWriter and input frames would causing skipping while burning
        if method == "resize":
            return cv2.resize(frame, (shape_width, shape_height))
        elif method == "crop":
            left_top = (np.clip(crop_locator[0], 0, self._input_width), np.clip(crop_locator[1], 0, self._input_height))
            right_down = (np.clip(crop_locator[0] + shape_width, 0, self._input_width), np.clip(crop_locator[1] + shape_height, 0, self._input_height))
            # this_width = right_down[0] - left_top[0]
            # this_height = right_down[1] - left_top[1]
            return cv2.resize(frame[left_top[1]:right_down[1], left_top[0]:right_down[0]], (shape_width, shape_height))
        else:
            raise NotImplementedError

    def __generate(self, output_file, output_fps, output_width, output_height, output_frames, resample_ratio, starting_frame, crop_locator):
        assert int(output_fps) > 0 and int(output_width) > 0 and int(output_height) > 0 and resample_ratio > 0
        assert type(output_file) == str and len(output_file) > 0 and os.path.splitext(output_file)[1].replace(".", "") == self.output_format
        print("estimated output video duration: {} sec".format(output_frames / output_fps))

        if self.fourcc:
            self._writer = cv2.VideoWriter(output_file, self.fourcc, output_fps, (output_width, output_height))
        else:
            raise NotImplementedError

        __frame_counter = 0
        __weights_summary = 0.0
        __this_frame_index = 0
        while (self._reader.isOpened() and __frame_counter < output_frames):
            ret, frame = self._reader.read()
            __this_frame_index += 1
            if ret:
                if __this_frame_index > starting_frame:
                    __rep_num = int(__weights_summary + resample_ratio) - int(__weights_summary)
                    if crop_locator is None:
                        frame = self.__reshape_frame(frame, output_width, output_height, "resize")
                    else:
                        frame = self.__reshape_frame(frame, output_width, output_height, "crop", crop_locator)
                    for __r in range(__rep_num):
                        self._writer.write(frame)
                    __weights_summary += resample_ratio
                    __frame_counter += __rep_num
            else:
                print("video source interrupted!")
                break
        self._writer.release()

    def read_frame(self):
        if self._reader.isOpened():
            ret, frame = self._reader.read()
            self._reader_counter += 1
            if ret:
                return True, frame
            else:
                return False, None
        else:
            return False, None

    def input_frames(self):
        return self._input_frames

    def input_identity(self):
        return self._input_identity

    def split_frames(self, target_dir, ext=".png", start=None, num=None, fps=None, exists_callback=None, max_aspect=None):
        if start is None:
            start = 0
        else:
            start = np.clip(start, 0, self._input_frames - 1).astype(int)
        if num is None:
            end = self._input_frames
        else:
            end = np.clip(start + num, start, self._input_frames).astype(int)
        if fps is None:
            fps = 1.0
        else:
            fps = np.clip(fps, 0.0, self._input_fps).astype(int)

        frame_dir = os.path.join(target_dir)
        if os.path.exists(frame_dir):
            print("warning: this identity already used, please check if it is labeled!")
            if exists_callback is not None:
                if exists_callback(frame_dir):
                    file_list = os.listdir(frame_dir)
                    counter = 0
                    for _f in file_list:
                        if os.path.splitext(_f)[1] == ext:
                            counter += 1
                    print("warning: this identity packed with {} images, unpacking skipped!".format(counter))
                    return counter
        else:
            os.makedirs(frame_dir, exist_ok=True)
        carving_stats = 0
        acc_rate = 0.0
        fps_rate = fps / self._input_fps
        reporthook = progressbar_urlretrieve(prefix='Progress:', suffix='Unpacking', urlstr='Unpacked: ' + self._input, length=100, dynamic=True)
        while True:
            ret, frame = self.read_frame()
            this_frame_index = self._reader_counter - 1
            if ret:
                if start <= this_frame_index < end:
                    if int(acc_rate + fps_rate) > int(acc_rate):
                        height_, width_ = frame.shape[:2]
                        if max(height_, width_) > max_aspect:
                            if height_ >= width_:
                                new_height_ = max_aspect
                                new_width_ = round(width_ / height_ * new_height_)
                            else:
                                new_width_ = max_aspect
                                new_height_ = round(height_ / width_ * new_width_)
                            frame = cv2.resize(frame, (new_width_, new_height_))
                        img_path = os.path.join(frame_dir, str(this_frame_index) + ext)
                        cv2.imwrite(img_path, frame)
                        carving_stats += 1
                        reporthook(this_frame_index, 1, self._input_frames)
                    acc_rate += fps_rate
                elif this_frame_index >= end:
                    break
            else:
                print("error: reading video error, stopped at frame {}!".format(this_frame_index))
                break
        print("finish splitting: {} frames".format(carving_stats))
        return carving_stats

    def designate_format(self, output_format):
        self.output_format = output_format.lower()
        if self.output_format == 'avi':
            self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        elif self.output_format == 'mp4':
            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # self.fourcc = cv2.VideoWriter_fourcc(*'avc1')
        else:
            raise NotImplementedError

    def generate(self, output_file, output_fps=None, output_width=None, output_height=None, output_frames=None, resample_ratio=None, starting_frame=None, crop_locator=None):
        if output_fps is None:
            output_fps = self._input_fps
        if output_width is None:
            output_width = self._input_width
        if output_height is None:
            output_height = self._input_height
        if output_frames is None:
            output_frames = self._input_frames
        if resample_ratio is None:
            resample_ratio = 1.0
        if starting_frame is None:
            starting_frame = 0
        self.__generate(output_file, output_fps, output_width, output_height, output_frames, resample_ratio, starting_frame, crop_locator)


if __name__ == '__main__':
    video_source = "/mnt/cv/usr/chenzhou/synopsis/video_samples/1617080400392-1617084002678-2080040.mp4"
    video_generator = VideoGenerator(video_source, True)
    video_generator.designate_format("mp4")
    output_file = "/mnt/cv/usr/chenzhou/synopsis/video_samples/1617080400392-1617084002678-2080040_cropped.mp4"
    # output_fps = 15
    # output_width = 320
    # output_height = 240
    # output_frames = 150
    # resample_ratio = 0.5
    # video_generator.generate(output_file, output_width=676, output_height=573, output_frames=52350, starting_frame=0, crop_locator=(0, 147))
    # cvt_file = os.path.splitext(output_file)[0] + "_cvt" + ".mp4"
    # shell_cmd = "ffmpeg -y -i {} -vcodec h264 {}".format(output_file, cvt_file)
    # print(os.system(shell_cmd))
