import shutil, cv2, os
import numpy as np

# Map {image_number: [row_num, col_num]}
default_concat_mapping = {1: [1, 1],
                          2: [1, 2],
                          3: [1, 3],
                          4: [1, 4],
                          5: [1, 5],
                          6: [2, 3],
                          7: [2, 4],
                          8: [2, 4],
                          9: [3, 3],
                          10:[2, 5],
                          11:[3, 4],
                          12:[3, 4],
                          13:[4, 4],
                          14:[4, 4],
                          15:[4, 4],
                          16:[4, 4]}

default_model_postfix_mapping = {"pix2pixHD": "_fake_B.png",
                                 "cut"      : "_fake_B.png",
                                 "ugatit"   : "_fake_A2B.png",
                                 "5G"       : "_fake_B.png",
                                 "5.7G"     : "_fake_B.png",
                                 "900Mreorg": "_fake_B.png"}

class ConcatImages:
    
    def __init__(self, ):
        self.cv2_img_read_type = cv2.IMREAD_COLOR
        self.cv2_img_read_size = (512, 512)
        self.blank_img_size = (512, 512, 3)
        self.paired_image_list = []
        self.save_dir = None
        self.save_name = None
        self.blank_img = np.zeros(self.blank_img_size).astype(np.uint8)
        
        # Text config
        self.text_pos = (10, 30)
        self.text_font = cv2.FONT_HERSHEY_COMPLEX
        self.text_color = (0, 0, 255) # Red
        self.text_size = 1.0
        self.text_overstriking = 2
        
        
    def run_concat_images(self, save_dir, paired_image_path_list, paired_image_text_list=None, row_num=0, col_num=0):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        total_img_number = len(paired_image_path_list)
        assert total_img_number <= row_num * col_num, "Error image number {} > (row: {} x col: {})".format("%d"%total_img_number, "%d"%row_num, "%d"%col_num)
        
        flag_add_text = False
        if paired_image_text_list == None:
            flag_add_text = False
        else:
            assert len(paired_image_path_list) == len(paired_image_text_list), "Not match len(paired_image_path_list) %d and len(paired_image_text_list) %d".format(len(paired_image_path_list), len(paired_image_text_list))
            flag_add_text = True
        
        self.read_images_list(paired_image_path_list, paired_image_text_list, flag_add_text)
        self.concat_format_grids_byorder(row_num, col_num)
    
    
    def read_images_list(self, paired_image_path_list, paired_image_text_list, flag_add_text):
        self.paired_image_list = []
        if flag_add_text == True:
            for img_idx in range(len(paired_image_path_list)):
                self.paired_image_list += [self.add_text(self.read_image(paired_image_path_list[img_idx]),
                                                         paired_image_text_list[img_idx])]
        else:
            for img_idx in range(len(paired_image_path_list)):
                self.paired_image_list += [self.read_image(paired_image_path_list[img_idx])]
        self.save_name = paired_image_path_list[0].split("/")[-1]


    def concat_format_grids_byorder(self, row_num, col_num):
        # Stack format
        res = None
        total_img_number = len(self.paired_image_list)
        
        added_blank_img_number = row_num * col_num - total_img_number
        self.paired_image_list += [ self.blank_img for added_blank_img_idx in range(added_blank_img_number)]
        
        img_idx = 0
        row_res = None
        col_res = None
        for row_idx in range(row_num):
            for col_idx in range(col_num):
                if col_idx == 0:
                    col_res = self.paired_image_list[img_idx]
                else:
                    col_res = np.hstack([col_res, self.paired_image_list[img_idx]])
                img_idx += 1
            if row_idx == 0:
                row_res = col_res
            else:
                row_res = np.vstack([row_res, col_res])
        
        cv2.imwrite(os.path.join(self.save_dir, self.save_name), row_res)
    
    
    def concat_format_custom(self, ):
        pass

    
    def read_image(self, img_path):
        return cv2.resize(cv2.imread(img_path, self.cv2_img_read_type), self.cv2_img_read_size)
    
    
    def add_text(self, img, text):
        return cv2.putText(img, text, self.text_pos, self.text_font, self.text_size, self.text_color, self.text_overstriking)


class ConcatVideos:

    def __init__(self, ):
        self.save_dir = None
        self.save_name = None
        self.video_path_list = []
        self.total_video_number = 0
        self.ffmpeg_cmd_str = None
        self.alphabet_mapping_idx = 97
        self.flag_readd_video_text = True
        
        # Text config
        self.text_pos_x = 10
        self.text_pos_y = 30
        self.text_color = "red"
        self.text_size = 22
        self.text_font = self.set_text_font("/usr/share/fonts/")


    def run_concat_videos(self, save_dir, paired_video_path_list, paired_video_text_list=None, row_num=0, col_num=0, flag_readd_video_name=True):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        flag_add_text = False
        if paired_video_text_list == None:
            flag_add_text = False
        else:
            assert len(paired_video_path_list) == len(paired_video_text_list), "Not match len(paired_video_path_list) %d and len(paired_video_text_list) %d".format(len(paired_video_path_list), len(paired_video_path_list))
            flag_add_text = True
        
        if flag_add_text == True:
            self.video_path_list = self.add_video_text(paired_video_path_list, paired_video_text_list)
            assert len(self.video_path_list) == len(paired_video_path_list)
        else:
            self.video_path_list = paired_video_path_list
        
        self.flag_readd_video_text = flag_readd_video_name
        if self.flag_readd_video_text == True:
            assert paired_video_text_list is not None, "Must have paired_video_text_list."
        
        self.total_video_number = len(paired_video_path_list)
        assert self.total_video_number <= row_num * col_num, "Error video number {} > (row: {} x col: {})".format("%d"%self.total_video_number, "%d"%row_num, "%d"%col_num)
        
        self.concat_format_grids_byorder(row_num, col_num)


    def concat_format_grids_byorder(self, row_num, col_num):
        
        self.ffmpeg_cmd_str = "ffmpeg "
        
        for video_fn_path in self.video_path_list:
            self.ffmpeg_cmd_str += "-i " + video_fn_path + " "
        
        # Video concat format
        self.ffmpeg_cmd_str += "-filter_complex '"
        video_cnt = 0
        
        for t_row in range(row_num):
            for t_col in range(col_num):
                if video_cnt == 0:
                    self.ffmpeg_cmd_str += "[0:v]pad=iw*{}:ih*{}".format(col_num, row_num)
                else:
                    self.ffmpeg_cmd_str += "[{}];[{}][{}:v]overlay=w*{}:h*{}".format(chr(self.alphabet_mapping_idx + video_cnt), chr(self.alphabet_mapping_idx + video_cnt), video_cnt, t_col, t_row)
                video_cnt += 1
                if video_cnt >= self.total_video_number:
                    break
        
        self.ffmpeg_cmd_str += "'"
        
        # Video concat config
        self.ffmpeg_cmd_str += " -c:v libx264 -crf 22 -preset veryfast -y "
        self.ffmpeg_cmd_str += os.path.join(self.save_dir, "output.mp4")
        
        print("self.ffmpeg_cmd_str", self.ffmpeg_cmd_str)
        os.system(self.ffmpeg_cmd_str)


    def add_video_text(self, video_path_list, video_name_list):
        
        video_w_name_path_list = []
        
        for video_path, video_name in zip(video_path_list, video_name_list):
            assert os.path.isfile(video_path), "When adding video text, Not a video file path {}.".format(video_path)
            video_fn = video_path.split("/")[-1]
            video_w_name_fn = os.path.join(os.path.dirname(video_path), video_fn.split(".")[0] + "_w_name." + video_fn.split(".")[-1])
            if self.flag_readd_video_text == True:
                print("ffmpeg -i {} -vf drawtext=fontfile={}:fontcolor={}:fontsize={}:text='{}':x={}:y={} -y {}".format(video_path, self.text_font, self.text_color, self.text_size, video_name, self.text_pos_x, self.text_pos_y, video_w_name_fn))
                os.system("ffmpeg -i {} -vf drawtext=fontfile={}:fontcolor={}:fontsize={}:text='{}':x={}:y={} -y {}".format(video_path, self.text_font, self.text_color, self.text_size, video_name, self.text_pos_x, self.text_pos_y, video_w_name_fn))
            video_w_name_path_list += [video_w_name_fn]
        return video_w_name_path_list


    def set_text_font(self, font_path):

        for root, dirs, files in os.walk(font_path):
            if "DejaVuSans.ttf" in files:
                print("Successfully set text font in ffmpeg: {}".format("%s"%os.path.join(root, "DejaVuSans.ttf")))
                return os.path.join(root, "DejaVuSans.ttf")
        assert False, "Not exist DejaVuSans.ttf in {}".format("%s"%font_path)
