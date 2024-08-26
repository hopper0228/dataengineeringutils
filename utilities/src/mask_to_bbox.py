import os
import shutil
from glob import glob
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.convert_check import CheckMatch
from src.preprocess_tool import PreProcessTool


class MaskToBbox:
    def __init__(self, image_name = ['images','image'], label_name = ['labels','label'], mask_name = ['masks','mask']) -> None:
        self.PASCAL_COLOR_LIST = np.array([
            [0, 0, 0],
            [128, 0, 0],    #0酒紅
            [0, 128, 0],    #1綠色
            [128, 128, 0],  #2黃綠社
            [0, 0, 128],    #3深藍
            [128, 0, 128],  #4紅紫
            [0, 128, 128],  #5藍綠
            [128, 128, 128],#6灰色
            [64, 0, 0],     #7咖啡色
            [192, 0, 0],    #8亮紅
            [64, 128, 0],   #9亮綠色
            [192, 128, 0],  #10淺咖啡
            [64, 0, 128],   #11深紫
            [192, 0, 128],  #12桃紅
            [64, 128, 128], #13灰藍綠
            [192, 128, 128],#14灰粉
            [0, 64, 0],     #15深綠
            [128, 64, 0],   #16中淺咖啡
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128]    #19淺藍綠
        ], dtype=np.uint8)
        self.image_name = image_name
        self.label_name = label_name
        self.mask_name = mask_name

    def mask_to_bbox_by_folder(self, folderpath: Path, savepath: Path = '', savename: str = 'bbox') -> None:
        """
        if no savepath: images with bboxes folder will be the save path
        else: generate labels folder in folder path

        Arguments
        ---------
        folderpath: path
            input folder
        savepath: path
            where to save bboxes or masks, none by default
        savename: str
            the name of your directory, none by default

        Returns
        -------
        bool: True if successfully draw
        """
        pt = PreProcessTool()
        check = False

        cm = CheckMatch()
        match_list = cm.check_match(folderpath)
        for path, folder in match_list.items():
            if any(name.lower() in folder for name in self.mask_name):
                save_path = ''

                if savepath == '':
                    message = 'To bbox '
                else:
                    savepath = pt.get_save_path(path, savepath)
                    message = 'Draw bbox '

                    save_path = os.path.join(savepath, savename)

                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                mask_folder = [name for name in self.mask_name if name.lower() in folder]
                images = sorted(glob(os.path.join(path, folder[0], "*")))
                masks = sorted(glob(os.path.join(path, f'{mask_folder[0]}_palette', "*")))

                for x, y in tqdm(zip(images, masks), total = len(images), desc = message + path, ascii = True):
                    root, extension = os.path.splitext(y)
                    if extension == ".txt":
                        if os.path.exists(save_path):
                            shutil.rmtree(save_path)
                        break
                    name = x.split(os.sep)[-1].split(".")[0]

                    x = cv2.imread(x, cv2.IMREAD_COLOR)
                    y = cv2.imread(y)

                    bboxes, colors, yolo_format = self.mask_to_bbox(y)
                    if not os.path.exists(save_path):
                        if not os.path.exists(os.path.join(path,self.label_name[0])):
                            os.mkdir(os.path.join(path,self.label_name[0]))
                        f = open(f"{path}\{self.label_name[0]}\{name}.txt", "w+")

                    for col, bbox in enumerate(bboxes):
                        x = cv2.rectangle(x, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors[col], 8)
                        if not os.path.exists(save_path):
                            print(f'{yolo_format[col][0]} {yolo_format[col][1]} {yolo_format[col][2]} {yolo_format[col][3]} {yolo_format[col][4]}',file=f)

                    if os.path.exists(save_path):
                        cv2.imwrite(fr"{save_path}\{name}.png", x)
                    else:
                        f.close()
        return check

    def mask_to_bbox(
                self,
                mask: np.ndarray
        ) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int]], List[Tuple[int, float, float, float, float]]]:
        """convert mask to bbox

        Arguments
        ---------
        mask: np.ndarray
            input array

        Returns
        -------
        bboxes: List[Tuple[int, int, int, int]]
        colors: List[Tuple[int, int, int]]
            which stands for R, G, B, respectively
        yolo_format: List[Tuple[int, float, float, float, float]]]
        """
        unique_colors = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)

        bboxes = []
        colors = []
        yolo_format = []

        for color in unique_colors:
            if (color == [0,0,0]).all():
                continue
            color_mask = cv2.inRange(mask, color, color)

            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = color[[2, 1, 0]]
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                image_height, image_width, image_axis = mask.shape
                yolo_x = (x + w / 2) / image_width
                yolo_y = (y + h / 2) / image_height
                yolo_width = w / image_width
                yolo_height = h / image_height
                try:
                    yolo_format.append((np.where((self.PASCAL_COLOR_LIST == color).all(axis=1))[0][0]-1,yolo_x,yolo_y,yolo_width,yolo_height))
                    bboxes.append((x, y, x + w, y + h))
                    colors.append(tuple(int(c) for c in color[[2, 1, 0]]))
                except Exception as e:
                    #如果是沒有調過色的會出錯
                    print(e)
                    continue
        return bboxes, colors, yolo_format

if __name__ == "__main__":
    import argparse

    # convert mask to bbox directly
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--path", type=str, default=r"C:\Users\edmond_huang\Desktop\original_dataset", help="your path")
    parser.add_argument("-sp","--savepath", type=str, default=r"C:\Users\edmond_huang\Desktop\result", help="your save path")
    parser.add_argument("-sn","--savename", type=str, default="bbox", help="your save name")
    args = parser.parse_args()

    mtb = MaskToBbox()
    mtb.mask_to_bbox_by_folder(args.path, args.savepath, args.savename)
