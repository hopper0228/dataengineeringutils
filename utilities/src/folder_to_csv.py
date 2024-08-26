import csv
import os
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from src.convert_check import CheckMatch, PreCheckTool
from src.preprocess_tool import PreProcessTool


class FolderTOCsv:
    """
    CSV_COLUMN:List[str] = []
    FILE_SUFFIX:str = '.csv'

    @classmethod
    def foo(cls, file_name):
        print(file_name + cls.FILE_SUFFIX)
    """
    FILE_TYPE = '.csv' # you can change this to whatever you what
    CSV_COLUMN = ['index', 'image_path', 'class_id', 'pos_neg', 'bbox_path', 'mask_path']

    def get_csv_file_info(self, root_folder: Path, image_folder_name: str,
            label_folder_name: str, mask_folder_name: str, is_mask: bool) -> List[str]:

        info = []

        reference_path = os.getcwd()
        relative_path = os.path.relpath(root_folder, reference_path)

        image_folder_path = os.path.join(relative_path, image_folder_name)
        label_folder_path = os.path.join(relative_path, label_folder_name)
        mask_folder_path = os.path.join(relative_path, mask_folder_name)

        ppt = PreProcessTool()
        images = ppt.img_lb_list(image_folder_path, label_folder_path)
        images_mask = ppt.img_lb_list(image_folder_path,mask_folder_path)

        img_root_path = os.path.join(root_folder, image_folder_name)
        label_root_path = os.path.join(root_folder, label_folder_name)
        mask_root_path = os.path.join(root_folder, mask_folder_name)

        image_folder_name = os.path.join(root_folder.split(os.sep)[-1], image_folder_name)
        mask_folder_name = os.path.join(root_folder.split(os.sep)[-1], mask_folder_name)
        label_folder_name = os.path.join(root_folder.split(os.sep)[-1], label_folder_name)

        if is_mask is True:
            csv_name = relative_path.split(os.sep)[-1] + '_seg.csv'
        else:
            csv_name = relative_path.split(os.sep)[-1] + '_obj.csv'

        # create a csv file
        csv_file_path = os.path.join(os.path.dirname(relative_path), csv_name)

        info = [img_root_path, csv_name, csv_file_path]

        if is_mask is False:
            info.extend((images, label_root_path, label_folder_path, image_folder_name, label_folder_name))

        if is_mask is True:
            info.extend((images_mask, mask_root_path, image_folder_name, mask_folder_name, label_folder_name))

        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)

        return info

    def folder_to_csv(self, root_folder: Path) -> None:
        """generate root_folder.csv

        Arguments
        ---------
        root_folder: path
            folder path
        """
        cm = CheckMatch()
        match_list = cm.check_match(root_folder)
        for path in match_list.keys():
            if 'masks' in os.listdir(path):
                self.folder_csv_mask(path, match_list[path][0], match_list[path][1], match_list[path][2])

            self.folder_to_csv_bbox(path, match_list[path][0], match_list[path][1])

    def folder_to_csv_bbox(self, root_folder: Path,  image_folder_name: str, label_folder_name: str) -> None:
        """generate root_folder.csv

        Arguments
        ---------
        root_folder: path
            the path
        image_folder_name: str
            image folder's name
        label_folder_name:str
            label folder's name
        """
        pc = PreCheckTool()

        (img_root_path, csv_name, csv_file_path, images,
         label_root_path, label_folder_path, image_folder_name, label_folder_name) = self.get_csv_file_info(root_folder, image_folder_name, label_folder_name, "", False)

        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames = self.CSV_COLUMN)

            # write csv header
            csv_writer.writeheader()

            count = 0
            for image, label in tqdm(images.items(), desc = 'To csv', ascii = True):

                image_path = os.path.join(img_root_path, image)
                label_path = os.path.join(label_root_path, label)
                if not pc.check_image(image_path) or not pc.check_yolo_format(image_path,label_path):
                    continue
                # 生成文件路径
                unique_class = []
                with open(os.path.join(label_folder_path, label), mode='rb') as fin:
                    data = fin.read()
                try:
                    data = data.decode('utf-8')
                except Exception as e:
                    print(e)
                    break

                lines = data.split('\n')

                for line in lines:
                    line = line.strip() # remove /n and whitspace
                    if line == '':
                        continue
                    if line is None:
                        raise ValueError('line is empty')

                    class_id, x_center, y_center, width, height, confidence = (line.split() + [0.0])[:6]

                    if int(class_id) not in unique_class:
                        unique_class.append(int(class_id))

                unique_class.sort()

                class_id = np.array2string(np.array(unique_class))

                #class_id = f"[{','.join(map(str, unique_class))}]"

                pos_neg = 1
                image_path = '.' + os.sep + os.path.join(image_folder_name, image)
                bbox_path = '.' + os.sep + os.path.join(label_folder_name, label)
                mask_path = ""

                csv_writer.writerow({'index': count, 'image_path': image_path, 'class_id': class_id,
                                    'pos_neg': pos_neg, 'bbox_path': bbox_path, 'mask_path': mask_path})

                count += 1

        print(f"CSV file has been created at: {csv_file_path}")

    def folder_csv_mask(self, root_folder: Path, image_folder_name: str, label_folder_name: str, mask_folder_name: str) -> None:
        """generate root_folder.csv

        Arguments
        ---------
        root_folder: path
            the path of the root folder
        image_folder_name: str
            image folder's name
        label_folder_name: str
            label folder's name
        mask_folder_name: str
            mask folder's name
        """
        import cv2
        pc = PreCheckTool()

        (img_root_path, csv_name, csv_file_path,
        images_mask, mask_root_path, image_folder_name, mask_folder_name, label_folder_name) = self.get_csv_file_info(root_folder, image_folder_name, label_folder_name, mask_folder_name, True)

        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames = self.CSV_COLUMN)

            # write csv file header
            csv_writer.writeheader()

            count = 0
            for image, label in tqdm(images_mask.items(), desc = 'To csv', ascii = True):

                image_path = os.path.join(img_root_path, image)
                mask_path = os.path.join(mask_root_path, label)
                if not pc.check_image(image_path) or not pc.check_image(mask_path):
                    continue
                # 生成文件路径
                mask = cv2.imread(mask_path)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                unique_class = list(np.unique(mask))
                if 0 in unique_class:
                    unique_class.remove(0)


                pos_neg = 1
                if len(unique_class) == 0:
                    unique_class.append(0)
                unique_class.sort()

                class_id = np.array2string(np.array(unique_class))
                #class_id = f"[{','.join(map(str, unique_class))}]"

                bbox_name, extension = os.path.splitext(label)

                img_path = '.' + os.sep + os.path.join(image_folder_name, image)
                bbox_path = '.' + os.sep + os.path.join(label_folder_name, bbox_name + '.txt')
                mask_path = '.' + os.sep + os.path.join(mask_folder_name, label)

                csv_writer.writerow({'index': count, 'image_path': img_path, 'class_id': class_id,
                                    'pos_neg': pos_neg, 'bbox_path': bbox_path, 'mask_path': mask_path})

                count += 1

        print(f"CSV file has been created at: {csv_file_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--path", type=str, default=r"D:\data\temp\customer\CviLux", help="your path")
    args = parser.parse_args()

    ftc = FolderTOCsv()
    ftc.folder_to_csv(args.path)
