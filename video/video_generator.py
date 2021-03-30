import cv2
import os
import numpy as np


def decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])


class VideoGenerator:
    def __init__(self, video_input, visual=False):
        self._input = video_input
        self._reader = cv2.VideoCapture(self._input)
        self._input_width = int(self._reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._input_height = int(self._reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._input_frames = int(self._reader.get(cv2.CAP_PROP_FRAME_COUNT))
        self._input_fps = int(self._reader.get(cv2.CAP_PROP_FPS))
        self._input_fourcc = decode_fourcc(self._reader.get(cv2.CAP_PROP_FOURCC))
        self.fourcc = None
        print("[VideoSize]:{}x{}x{} [FPS]:{} [FOURCC]:{}".format(self._input_frames, self._input_width, self._input_height, self._input_fps, self._input_fourcc))
        if visual:
            while self._reader.isOpened():
                ret, frame = self._reader.read()
                if ret:
                    cv2.imshow("VISUAL", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def __del__(self):
        self._reader.release()

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
        while(self._reader.isOpened() and __frame_counter < output_frames):
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
