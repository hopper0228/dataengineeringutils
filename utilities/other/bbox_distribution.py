import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.annotation_utils.alg_annotation_utils import Bbox


class BboxResolution:
    def __init__(self, class_id:int=0):
            self.name = ""
            self.class_id = int(class_id)

            self.width_ratio = 0
            self.height_ratio = 0
            self.area_ratio = 0

            self.width_pixel = 0
            self.height_pixel = 0
            self.area_pixel = 0

            self.xmin = 0
            self.xmax = 0
            self.ymin = 0
            self.ymax = 0

            self.x_center = 0
            self.y_center = 0

class BboxHistogram:
    def __init__(self, supported_file_keys, path):
        self.SUPPORTED_FILE_KEYS = supported_file_keys
        self.PATH = path

        self.checkmatch = ""
        self.image_paths = {}
        self.annosvis_paths = {}
        self.imagepath = {}
        self.annosvispath = {}

        self.resolutions = {}
        self.resolutions_temp = {}
        self.resolution = {}

        self.tiny_img_resolution = []
        self.tiny_img_path = ""

    #每個資料夾中的每張圖片的長寬
    def image_resolution_by_folder(self, folderpath:str):

        try:
            contents =  os.listdir(folderpath)
            folder_path = folderpath.split(os.sep)
            #圖片可能包含image
            if any("label" in name.lower() for name in contents) and not any("image" in name.lower() and os.path.isdir(folderpath + os.sep + name) for name in contents):
                self.checkmatch = "tiny"
            for item in contents:
                lower_item = item.lower()
                item_path = os.path.join(folderpath, lower_item)
                root, extension = os.path.splitext(lower_item)

                if os.path.isdir(item_path):
                    self.image_resolution_by_folder(item_path)

                    if "label" in lower_item and self.resolutions_temp:
                        self.resolutions[folderpath] = self.resolutions_temp.copy()
                        self.resolutions_temp.clear()
                        self.resolution.clear()

                elif "annosvis" in folder_path[len(folder_path)-1].lower():
                    self.annosvispath[root] = item_path

                elif "image" in folder_path[len(folder_path)-1].lower():
                    self.checkmatch = ""
                    if extension.lower() in self.SUPPORTED_FILE_KEYS:
                        img = Image.open(item_path)
                        self.resolution[root] = img.size
                        self.imagepath[root] = item_path

                elif "label" in folder_path[len(folder_path)-1].lower():
                    if extension.lower() == ".txt":
                        if self.checkmatch == "" and self.resolution:
                            self.resolutions_temp[item_path] = self.resolution[root]
                            self.image_paths[lower_item] = self.imagepath[root]
                            if root in self.annosvispath.keys():
                                self.annosvis_paths[lower_item] = self.annosvispath[root]
                        elif self.checkmatch == "tiny" and self.tiny_img_resolution:
                            self.resolutions_temp[item_path] = self.tiny_img_resolution
                            self.image_paths[lower_item] = self.tiny_img_path

                elif extension.lower() in self.SUPPORTED_FILE_KEYS and self.checkmatch == "tiny":
                    img = Image.open(item_path)
                    self.tiny_img_resolution = img.size
                    self.tiny_img_path = item_path


        except Exception as e:
            print(e)


        return self.resolutions

    #每張圖片的長寬面積比例,每張圖片的面積加總比
    def bbox_resolution(self, resolutions):

        bbox_histograms = {}
        bbox_center_image = {}
        for i in resolutions.keys():
            bbox_images = {}
            bbox_his = {}
            for j in resolutions[i].keys():
                total_area_ratio = 0
                item = j.split(os.sep)
                name = item[len(item)-1]

                with open(file=j, mode='rb') as fin:
                    data = fin.read()

                try:
                    data = data.decode('utf-8')
                except Exception as e:
                    print(e)
                    break

                lines = data.split('\n')

                bbox_image = []
                for line in lines:
                    bbox_temp = BboxResolution()
                    line = line.strip() # remove /n and whitspace
                    if line == '':
                        continue

                    if line is None:
                        raise ValueError('line is empty')

                    bbox = Bbox()
                    bbox.parse_bbox_from_line(line, Bbox.BboxType.XY_CENTER_WH, True, resolutions[i][j][0], resolutions[i][j][1])

                    class_id = -1

                    class_id, x_center, y_center, width, height, confidence = (line.split() + [0.0])[:6]

                    width = float(width)
                    height = float(height)
                    bbox_temp.class_id = int(class_id)

                    bbox_temp.xmin = bbox.xmin
                    bbox_temp.xmax = bbox.xmax
                    bbox_temp.ymin = bbox.ymin
                    bbox_temp.ymax = bbox.ymax

                    bbox_temp.x_center = (bbox.xmax + bbox.xmin)/2
                    bbox_temp.y_center = (bbox.ymax + bbox.ymin)/2

                    bbox_temp.width_ratio = width
                    bbox_temp.height_ratio = height
                    bbox_temp.area_ratio = bbox_temp.width_ratio*bbox_temp.height_ratio

                    bbox_temp.width_pixel = (bbox.xmax-bbox.xmin)
                    bbox_temp.height_pixel = (bbox.ymax-bbox.ymin)
                    bbox_temp.area_pixel = bbox_temp.width_pixel*bbox_temp.height_pixel

                    bbox_temp.name = name

                    total_area_ratio += bbox_temp.area_ratio

                    if not bbox_temp.class_id in bbox_his:
                        bbox_his[bbox_temp.class_id] = {}
                    bbox_his[bbox_temp.class_id][len(bbox_his[bbox_temp.class_id])] = bbox_temp
                    bbox_image.append([bbox_temp.x_center, bbox_temp.y_center])

                bbox_images[j] = [bbox_image.copy(), total_area_ratio]
                #bbox_center_image[j] = [bbox_image.copy(), total_area_ratio]
            if len(bbox_his) > 0:
                bbox_center_image[i] = bbox_images.copy()
                bbox_histograms[i] = bbox_his.copy()

        return bbox_histograms,bbox_center_image

    #劃出histogram
    def draw_his(self,bbox_histograms, SAVE_PATH, bins):

        width_pixel_values = []
        height_pixel_values = []
        area_pixel_values = []

        width_ratio_values = []
        height_ratio_values = []
        area_ratio_values = []

        bbox_key = []
        try:
            #哪一個資料夾
            for k in bbox_histograms.keys():
                save_name = SAVE_PATH
                print(k)
                title_name = ['Width Values','Height Values','Area Values']
                ks = k.split(os.sep)
                for n in range(3,len(ks)):
                    save_name = save_name + os.sep + ks[n]
                    if not os.path.exists(save_name):
                        os.mkdir(save_name)

                f = open(save_name+r"\summary.txt","w+")
                print("bounding box 數量", file=f)

                #bbox_his是每個class的所有bounding box
                for i in sorted(bbox_histograms[k].keys()):

                    width_pixel_values.append([float(val.width_pixel) for val in sorted(bbox_histograms[k][i].values(),key = lambda x:x.width_pixel)])
                    height_pixel_values.append([float(val.height_pixel) for val in sorted(bbox_histograms[k][i].values(), key=lambda x:x.height_pixel)])
                    area_pixel_values.append([float(val.area_pixel) for val in sorted(bbox_histograms[k][i].values() , key=lambda x:x.area_pixel)])

                    width_ratio_values.append([float(val.width_ratio) for val in sorted(bbox_histograms[k][i].values(),key = lambda x:x.width_ratio)])
                    height_ratio_values.append([float(val.height_ratio) for val in sorted(bbox_histograms[k][i].values(), key=lambda x:x.height_ratio)])
                    area_ratio_values.append([float(val.area_ratio) for val in sorted(bbox_histograms[k][i].values() , key=lambda x:x.area_ratio)])

                    bbox_key.append(str(i))

                    print(f"     class {i} :{len(bbox_histograms[k][i])}",file=f)
                    print(f"             pixel: {min(area_pixel_values[-1]), np.percentile(area_pixel_values[-1], 25), np.mean(area_pixel_values[-1]), np.median(area_pixel_values[-1]), np.percentile(area_pixel_values[-1], 75), max(area_pixel_values[-1])}",file=f)
                    print(f"             ratio: {min(area_ratio_values[-1]), np.percentile(area_ratio_values[-1], 25), np.mean(area_ratio_values[-1]), np.median(area_ratio_values[-1]), np.percentile(area_ratio_values[-1], 75), max(area_ratio_values[-1])}",file=f)

                output_values = [width_pixel_values,height_pixel_values,area_pixel_values]
                output_ratio_values = [width_ratio_values,height_ratio_values,area_ratio_values]

                fig, axs = plt.subplots(3, 1, figsize=(8, 10))
                fig2, axs2 = plt.subplots(3, 1, figsize=(8, 10))
                fig3, axs3 = plt.subplots(3, 1, figsize=(8, 10))
                fig4, axs4 = plt.subplots(3, 1, figsize=(8, 10))

                #i為width,height,area
                #j為value
                #axs為count
                #axs2為frequency
                for i in range(3):
                    for j in range(len(bbox_histograms[k].keys())):
                        axs[i].hist(output_values[i][j], bins, range = (min(output_values[i][j]),max(output_values[i][j])), alpha = 0.5, label = bbox_key[j])
                        axs2[i].hist(output_values[i][j], bins, range = (min(output_values[i][j]),max(output_values[i][j])), density = True, alpha = 0.5, label = bbox_key[j])
                        axs3[i].hist(output_ratio_values[i][j], bins, range = (min(output_ratio_values[i][j]),max(output_ratio_values[i][j])), alpha = 0.5, label = bbox_key[j])
                        axs4[i].hist(output_ratio_values[i][j], bins, range = (min(output_ratio_values[i][j]),max(output_ratio_values[i][j])), density = True, alpha = 0.5, label = bbox_key[j])

                    axs[i].set(xlabel="Value", ylabel="Count", title=title_name[i])
                    axs[i].legend(loc = "upper right")
                    axs2[i].set(xlabel="Value", ylabel="Frequency", title=title_name[i])
                    axs2[i].legend(loc = "upper right")

                    axs3[i].set(xlabel="Value", ylabel="Count", title=title_name[i])
                    axs3[i].legend(loc = "upper right")
                    axs4[i].set(xlabel="Value", ylabel="Frequency", title=title_name[i])
                    axs4[i].legend(loc = "upper right")

                fig.tight_layout()
                fig2.tight_layout()
                fig3.tight_layout()
                fig4.tight_layout()

                fig.savefig(save_name + os.sep + "Pixel_Count.png")
                fig2.savefig(save_name + os.sep + "Pixel_Density.png")
                fig3.savefig(save_name + os.sep + "Ratio_Count.png")
                fig4.savefig(save_name + os.sep + "Ratio_Density.png")

                plt.close(fig)
                plt.close(fig2)
                plt.close(fig3)
                plt.close(fig4)

                width_pixel_values.clear()
                height_pixel_values.clear()
                area_pixel_values.clear()
                width_ratio_values.clear()
                height_ratio_values.clear()
                area_ratio_values.clear()
                bbox_key.clear()

                print(f"bins 數量: {bins}", file=f)

            f.close()

        except Exception as e:
            print(e)

    def class_number_satistic(self,bbox_histograms, SAVE_PATH):

        width_pixel_values = []
        height_pixel_values = []
        area_pixel_values = []

        width_ratio_values = []
        height_ratio_values = []
        area_ratio_values = []

        bbox_key = []
        try:
            #哪一個資料夾
            for k in bbox_histograms.keys():
                save_name = SAVE_PATH
                print(k)
                ks = k.split(os.sep)
                for n in range(3,len(ks)):
                    save_name = save_name + os.sep + ks[n]
                    if not os.path.exists(save_name):
                        os.mkdir(save_name)

                f = open(save_name+r"\summary.txt","w+")
                print("bounding box 數量", file=f)

                #bbox_his是每個class的所有bounding box
                for i in sorted(bbox_histograms[k].keys()):

                    width_pixel_values.append([float(val.width_pixel) for val in sorted(bbox_histograms[k][i].values(),key = lambda x:x.width_pixel)])
                    height_pixel_values.append([float(val.height_pixel) for val in sorted(bbox_histograms[k][i].values(), key=lambda x:x.height_pixel)])
                    area_pixel_values.append([float(val.area_pixel) for val in sorted(bbox_histograms[k][i].values() , key=lambda x:x.area_pixel)])

                    width_ratio_values.append([float(val.width_ratio) for val in sorted(bbox_histograms[k][i].values(),key = lambda x:x.width_ratio)])
                    height_ratio_values.append([float(val.height_ratio) for val in sorted(bbox_histograms[k][i].values(), key=lambda x:x.height_ratio)])
                    area_ratio_values.append([float(val.area_ratio) for val in sorted(bbox_histograms[k][i].values() , key=lambda x:x.area_ratio)])

                    bbox_key.append(str(i))

                    print(f"     class {i} :{len(bbox_histograms[k][i])}",file=f)
                    print(f"             pixel: {min(area_pixel_values[-1]), np.percentile(area_pixel_values[-1], 25), np.mean(area_pixel_values[-1]), np.median(area_pixel_values[-1]), np.percentile(area_pixel_values[-1], 75), max(area_pixel_values[-1])}",file=f)
                    print(f"             ratio: {min(area_ratio_values[-1]), np.percentile(area_ratio_values[-1], 25), np.mean(area_ratio_values[-1]), np.median(area_ratio_values[-1]), np.percentile(area_ratio_values[-1], 75), max(area_ratio_values[-1])}",file=f)

            f.close()

        except Exception as e:
            print(e)


if __name__ == "__main__":
    import argparse

    #用bbox計算
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--path", type=str, default=r"D:\data\translate_output\Fan_motherboard\motherboard\Motherboard2", help="your path")
    parser.add_argument("-sp","--savepath", type=str, default=r"D:\data\distribtuion", help="your save path")
    parser.add_argument("-sfk","--supported_file_keys", type=list, default= ['.jpg','.bmp','.png','.jpeg'], help="your supported file keys")
    parser.add_argument("-b","--bin_edges", type=int, default=20, help="your bin edges")

    args = parser.parse_args()
    b_his = BboxHistogram(args.supported_file_keys, args.path)
    resolutions = b_his.image_resolution_by_folder(b_his.PATH)
    if len(resolutions)>0:
        print("get reolutions")
        bbox_histograms, bbox_area_ratio_image = b_his.bbox_resolution(resolutions)
        print("get histogram")
        b_his.draw_his(bbox_histograms, args.savepath, args.bin_edges)
        print("finish")
    else:
        print("no bounding box")
