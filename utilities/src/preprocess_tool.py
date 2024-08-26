import copy
import os
import shutil
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image

from src.utils.annotation_utils.alg_annotation_utils import Annotation, AnnotationVisualizer


class DecodeProcess:
    def rle2mask(self, rle, imgshape):
        width = imgshape[0]
        height= imgshape[1]

        mask= np.zeros( width*height ).astype(np.uint8)

        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]

        current_position = 0
        for index, start in enumerate(starts):
            mask[int(start):int(start+lengths[index])] = 1
            current_position += lengths[index]

        return np.flipud( np.rot90( mask.reshape(height,width), k=1 ) )

    def train_decode(self, root_path):
        import pandas as pd
        PASCAL_COLOR_LIST = np.array([
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128]
            ], dtype = np.uint8)

        train = pd.read_csv(root_path + r'\train.csv')
        train = train[ train['EncodedPixels'].notnull() ]

        fn1 = ''
        train_path = os.path.join(root_path, "train")
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        train_image_path = os.path.join(train_path, "images")
        if not os.path.exists(train_image_path):
            os.makedirs(train_image_path)
        train_mask_path = os.path.join(train_path, "masks")
        if not os.path.exists(train_mask_path):
            os.makedirs(train_mask_path)

        #同一張圖片的不同class要跌再一起
        for i in range(0, len(train)):

            fn = train['ImageId'].iloc[i]

            class_id = train['ClassId'].iloc[i]

            img = cv2.imread(os.path.join(train_image_path,fn) )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = self.rle2mask( train['EncodedPixels'].iloc[i], img.shape  )

            color_mask = np.zeros_like(img)
            color_mask[mask == 1] = PASCAL_COLOR_LIST[class_id]
            bgr_image = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)

            img[mask==1,0] = 255

            if os.path.exists(os.path.join(train_mask_path, fn)):
                if fn1 == fn:

                    fn1 = train['ImageId'].iloc[i-1]

                    class_id1 = train['ClassId'].iloc[i-1]

                    img1 = cv2.imread(os.path.join(train_image_path,fn))
                    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                    mask1 = self.rle2mask( train['EncodedPixels'].iloc[i-1], img1.shape  )

                    color_mask1 = np.zeros_like(img1)
                    color_mask1[mask1 == 1] = PASCAL_COLOR_LIST[class_id1]
                    bgr_image1 = cv2.cvtColor(color_mask1, cv2.COLOR_RGB2BGR)

                    bgr_image = cv2.bitwise_or(bgr_image1, bgr_image)

            cv2.imwrite(os.path.join(train_mask_path, fn.replace('jpg','png')), bgr_image)

            fn1 = fn


class PreProcessTool:

    def unlabel_mask(self, root_path: Path, image_folder_name: str, label_folder_name: str) -> None:
        """
        Arguments
        ---------
        root_path: path
            root path
        image_folder_name: str
            name of the found image folder
        label_folder_name: str
            name of the found label folder
        """
        image_folder_path = os.path.join(root_path, image_folder_name)
        label_folder_path = os.path.join(root_path, label_folder_name)

        images = self.img_lb_list(image_folder_path,label_folder_path)

        for image, label in images.items():

            if label == "":
                image_temp = cv2.imread(os.path.join(image_folder_path,str(image)))
                image_temp = cv2.cvtColor(image_temp, cv2.COLOR_BGR2RGB)

                mask = np.zeros_like(image_temp)

                image_save_path = image.split(".")[0] + ".png"
                cv2.imwrite(os.path.join(label_folder_path, image_save_path), mask)

    def img_lb_list(self, image_folder_path: Path, label_folder_path: Path) -> Dict:
        """
        Arguments
        ---------
        image_folder_path: path
            path of found image folder
        label_folder_path: path
            path of found label folder

        Returns
        -------
        Dict:
            images
        """
        images = {}

        if not os.path.exists(label_folder_path):
            os.makedirs(label_folder_path)

        for image_file in sorted(os.listdir(image_folder_path)):
            images[image_file] = ""
            for label_file in sorted(os.listdir(label_folder_path)):
                img_root, extension = os.path.splitext(image_file)
                lb_root, extension = os.path.splitext(label_file)
                if img_root == lb_root:
                    images[image_file] = label_file
                    break
                else:
                    images[image_file] = ""

        return images

    def convert_dataset(self, root_path, save_path) -> None:
        """
        object detection: [images, labels]
        segment detection: [images, masks, masks_palette]

        Arguments
        ---------
        root_path: path
            folder path
        save_path: path
            save path
        """
        import shutil

        from convert_check import CheckMatch

        cm = CheckMatch()
        match_list = cm.check_match(root_path)
        for match_folder in match_list:
            temp_save_path = self.get_save_path(match_folder, save_path)
            for i in match_list[match_folder]:
                if not os.path.exists(os.path.join(temp_save_path, i)):
                    shutil.copytree(os.path.join(match_folder, i) , os.path.join(temp_save_path, i))

    def get_save_path(self, root_path, save_path) -> Path:
        """make a shortcut to root_path indside save_path

        Arguments
        ---------
        root_path: path
            folder path
        save_path: path
            save path

        Returns
        -------
        path:
            save_path
        """
        num = len(save_path.split(os.sep))
        for name in root_path.split(os.sep)[num:]:
            save_path = os.path.join(save_path,name)
            if not os.path.exists(save_path):
                if ' ' in save_path:
                    save_path = save_path.replace(' ', '_')
                os.makedirs(save_path, exist_ok=True)
        return save_path

    #draw_anno使用
    def is_rgb_image(self, root_path):
        """
        Args:
            root_path = 圖片路徑
        Returns:
            圖片中是否為rgb
        """
        image = cv2.imread(root_path)
        unique_color = np.unique(image)
        if any(x>20 for x in unique_color):
            return True
        return False

	#以下多為convert_check.py使用
    def is_rgb_folder(self, root_path: Path) -> bool:
        """
        Arguments
        ---------
        root_path: path
            image_folder's path

        Returns
        -------
        bool:
            check if all files in folder is an RGB image
        """
        for item in os.listdir(root_path):
            image = cv2.imread(os.path.join(root_path,item))
            unique_color = np.unique(image)
            if any(x>20 for x in unique_color):
                return True
        return False

    def is_rgb_image(self, root_path: Path) -> bool:
        """
        Arguments
        ---------
        root_path: path
            image's path

        Returns
        -------
        bool:
            check if image is an RGB image
        """
        image = cv2.imread(root_path)
        unique_color = np.unique(image)
        if any(x>20 for x in unique_color):
            return True
        return False

    def gray_to_color(self, root_path: Path) -> None:
        """ <<< change BGR into RGB

        Arguments
        ---------
        root_path: path
            mask folder's path
        """
        if not os.path.exists(root_path + '_palette'):
            os.makedirs(root_path + '_palette', exist_ok=True)
        for item in os.listdir(root_path):
            self.save_colorful_mask(os.path.join(root_path,item))

    def save_colorful_mask(self, mask_path: Path) -> None:
        """ change BGR into RGB

        Arguments
        ---------
        root_path: path
            mask's path
        """
        from PIL import Image
        PASCAL_COLORS = np.array([
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128]], dtype='uint8').flatten()

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(mask.astype(np.uint8), mode = 'L')
        img.putpalette(PASCAL_COLORS)
        if not os.path.exists(os.path.dirname(mask_path) + '_palette'):
            os.makedirs(os.path.dirname(mask_path) + '_palette',exist_ok=True)
        img.save(mask_path.replace(mask_path.split(os.sep)[-2],(mask_path.split(os.sep)[-2] + '_palette')))


    def color_to_gray(self, root_path: Path) -> None:
        """ change RGB into gray

        Arguments
        ---------
        root_path: path
            mask folder's path
        """
        contents = os.listdir(root_path)
        save_path = self.create_palette_folder(root_path)
        for item in contents:
            path = os.path.join(root_path,item)
            file_name = path.split(os.sep)[-1]
            self.save_gray_mask(os.path.join(save_path,file_name))
            root_path = save_path

    def create_palette_folder(self, root_path: Path) -> Path:
        """build masks, masks_palette folder

        Arguments
        ---------
        root_path: path
            dataset's path

        Returns
        -------
        path:
            masks path
        """
        if 'palette' not in root_path:
            os.rename(root_path, root_path + '_palette')
            os.makedirs(root_path,exist_ok=True)
            return root_path + '_palette'
        else:
            return root_path

    def save_gray_mask(self, mask_path: Path) -> None:
        """RGB to gray

        Arguments
        ---------
        mask_path: path
            mask's path
        """
        PASCAL_COLORS = np.array([
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128]], dtype='uint8')

        mask = cv2.imread(mask_path)
        new_mask = np.zeros_like(mask, dtype=np.uint8)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                color = mask[i, j]
                color = color[[2,1,0]]
                try:
                    index = np.where((PASCAL_COLORS == color).all(axis=1))[0][0]
                    if index > 0:
                        new_mask[i, j] = index
                    else:
                        new_mask[i, j] = 0
                except:
                    print(mask_path + ' no this color')
                    continue

        cv2.imwrite(mask_path.replace('_palette',''), new_mask)


class File:

    def __init__(self):
        self.type = ""
        self.num = 0
        self.file_num_dict = {}
        self.resolution = []
        self.folder_name = ""
        self.checkmatch = ""
        self.file_num = {}
        self.match_path = ""

    def clear(self):
        self.type = ""
        self.num = 0
        self.file_num_dict = {}
        self.resolution = []
        self.folder_name = ""

class Folder:
    def __init__(self):
        self.parent_name = ""
        self.folder_name = ""
        self.img_dict = {}          #圖檔的資訊=FILE CLASS
        self.file_dict = {}         #文檔的數量=txt
        self.repo_dict = {}         #此folder包含的folder ex:benjamin.repo_dict = {stylegan2-nv,stylegan2-pytorch}
        self._unknown_dict = {}     #其他
        self.checkmatch = ""

class FolderStruct:

    def __init__(self, image_name = ['images','image'], label_name = ['labels','label'], mask_name = ['masks','mask']) -> None:
        self.onefile = File()
        self.temp = []
        self.SUPPORTED_FILE_KEYS = ['.jpg','.bmp','.png','.jpeg','.JPG','.BMP','.PNG','.JPEG']
        self.f = None
        self.check_match_path = ''
        self.checkmatch = ""
        self.match_path = ""
        self.image_name = image_name
        self.label_name = label_name
        self.mask_name = mask_name

    def file_to_dict(self, folder_path: Path, parent_name: str, indent: int) -> Folder:
        """
        Arguments
        ---------
        folder_name: path
            folder's path
        parent_name:str
            folder's parent name

        Returns
        -------
        Folder:
            folder_struct
        """
        folder_dict = Folder()
        folder_dict.folder_name = folder_path.split(os.sep)[-1]
        folder_dict.parent_name = parent_name
        unique = []
        for num in self.onefile.resolution:
            if str(num) not in unique:
                unique.append(str(num))
        filetype = list(self.onefile.file_num_dict.keys())
        file_type = ''
        for ftype in filetype:
            if ftype in self.SUPPORTED_FILE_KEYS or ftype.lower() in '.txt' or ftype.lower() in '.tmp':
                file_type = ftype
        filecount = list(self.onefile.file_num_dict.values())
        for c in range(len(filetype)):
            if str(filetype[c]) in self.SUPPORTED_FILE_KEYS:
                folder_dict.img_dict[str(filetype[c]),"長寬"] = str(filecount[c]),unique
                if self.f:
                        print("  " * indent + "[" + str(filetype[c]) + "]" + ":" + str(filecount[c]), file = self.f)
                        if len(unique)>1:
                            print("  " * indent + "resolution : No fixed", file = self.f)
                        else:
                            print("  " * indent + "resolution : " + str(unique), file = self.f)
            else:
                folder_dict.file_dict[str(filetype[c])] = str(filecount[c])
                if self.f:
                        print("  " * indent + "[" + str(filetype[c]) + "]" + ":" + str(filecount[c]), file = self.f)

        if any(name.lower() in folder_path.lower() for name in self.image_name) and not any(name.lower() in folder_path.lower() for name in self.label_name) and not any(name.lower() in folder_path.lower() for name in self.mask_name):
            fc = 0
            for i in range(len(filecount)):
                fc = fc + filecount[i]
            self.onefile.file_num[fc] = folder_path
            self.onefile.checkmatch = folder_path
            self.check_match_path = os.path.dirname(folder_path)
        elif (any(name.lower() in folder_path.lower() for name in self.label_name) or any(name.lower() in folder_path.lower() for name in self.mask_name)) and len(self.onefile.file_num) > 0 and self.onefile.file_num_dict[file_type] in self.onefile.file_num:
            self.match_path = self.onefile.file_num[self.onefile.file_num_dict[file_type]]
            if self.onefile.file_num_dict[file_type] in self.onefile.file_num:
                self.onefile.checkmatch = self.match_path
                folder_dict.checkmatch = self.match_path
                self.onefile.file_num.pop(self.onefile.file_num_dict[file_type])
            else:
                self.onefile.checkmatch = "no match"

        self.onefile.clear()
        return folder_dict

    def folder_to_dict(self, folder_path, save_path = None, SUPPORTED_FILE_KEYS: List = [], indent: int = 0) -> Folder:
        """
        Arguments
        ---------
        folder_path: path
            folder_path
        save_path: path
            the path where you want to store the records
        save_name: str
            name of your record
        SUPPORTED_FILE_KEYS: List[str]
            file types which is being supported

        Returns
        -------
        Folder:
            folder_struct
        """
        if save_path != None and self.f == None:
            self.f = open(save_path,"w+")
            print(folder_path,file=self.f)

        if SUPPORTED_FILE_KEYS:
            self.SUPPORTED_FILE_KEYS = SUPPORTED_FILE_KEYS
        folder_dict = Folder()

        # 获取文件夹内的所有内容（包括文件和子文件夹）
        contents = os.listdir(folder_path)
        contents.sort(key = lambda x: (os.path.isdir(os.path.join(folder_path, x)), x))
        try:
            for item in contents:
                item_path = os.path.join(folder_path, item)
                if os.path.isdir(item_path):
                    # 如果是文件夹，递归调用folder_to_dict函数
                    if self.onefile.num > 0:
                        self.onefile.folder_name = folder_path
                        self.temp.append(copy.copy(self.onefile))
                        self.onefile.clear()
                    if self.f:
                        print("  " * indent + f"目錄: {item}", file = self.f)
                    folder_dict.folder_name = folder_path.split(os.sep)[-1]
                    folder_dict.parent_name = folder_path.split(os.sep)[-2]
                    folder_dict.repo_dict[item] = self.folder_to_dict(item_path, save_path, self.SUPPORTED_FILE_KEYS , indent+2)
                    if self.onefile.num!=0 and len(folder_dict.repo_dict[item].repo_dict) == 0:
                        folder_dict.repo_dict[item] = self.file_to_dict(item_path, folder_dict.folder_name, indent+2)
                        if os.path.exists(self.match_path):
                            self.onefile.checkmatch = ""

                    if self.match_path and folder_path == self.check_match_path:
                        folder_dict.checkmatch = "match"
                        self.match_path = ""

                else:
                    # 如果是文件，直接添加到字典
                    root, extension = os.path.splitext(item_path)
                    if extension in self.onefile.file_num_dict:
                        self.onefile.file_num_dict[extension] += 1
                    elif extension != '':
                        self.onefile.file_num_dict[extension] = 1

                    if extension in self.SUPPORTED_FILE_KEYS:
                        img = Image.open(item_path)
                        self.onefile.resolution.append(img.size)
                    else:
                        self.onefile.resolution.append(extension)

                    self.onefile.num += 1
        except Exception as e:
            self.onefile.num += 1
            print(e)

        if len(self.temp) > 0 and self.temp[-1].folder_name == folder_path:
            self.onefile = self.temp[-1]
            temp_file_dict = self.file_to_dict(folder_path,folder_path.split(os.sep)[-2], indent)
            folder_dict.file_dict = temp_file_dict.file_dict
            self.temp.pop()

        return folder_dict


class DrawAnnos:
    def __init__(self, image_name = ['images','image'], label_name = ['labels','label'], mask_name = ['masks','mask'], annos_name = 'Annosvis'):
        self.path_del_num = 3
        self.draw_path = []
        self.SAVE_PATH = ""
        self.project_name = ""
        self.image_name = image_name
        self.label_name = label_name
        self.mask_name = mask_name
        self.annos_name = annos_name


    def draw_by_struct(self, folder_structure: Dict) -> None:
        """
        Arguments
        ---------
        folder_structure: Dict
        """
        import re
        da = DrawAnnos()
        pt = PreProcessTool()
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        self.draw_path.append(os.path.join(self.draw_path[-1], folder_structure.folder_name))
        if 'match' in folder_structure.checkmatch:
            self.project_name = folder_structure.folder_name + ".ditprj"
        if os.path.isdir(self.draw_path[-1]):
            if chinese_pattern.search(self.draw_path[-2]):
                print(f'Found Mandarin in {self.draw_path[-2]}')
                self.draw_path.pop()
                return
            if os.path.exists(folder_structure.checkmatch):
                print(self.draw_path[-2])

                save_path = pt.get_save_path(self.draw_path[-3],self.SAVE_PATH)
                os.makedirs(os.path.join(save_path, "Hub"),exist_ok=True)
                save_path = os.path.join(save_path, "ori_and_check")
                os.makedirs(save_path,exist_ok=True)

                if self.project_name in os.listdir(self.draw_path[-2]):
                    #畫project的
                    da.draw_annos_with_ditprj(self.draw_path[-2], save_path)
                else:
                    #畫沒project的
                     da.draw_annos_without_ditprj(self.draw_path[-2], save_path)

        for i in folder_structure.repo_dict:
                if hasattr(folder_structure.repo_dict[i],"repo_dict"):
                    self.draw_by_struct(folder_structure.repo_dict[i])

        self.draw_path.pop()

    def draw_by_folder(self, root_path: Path, save_path: Path) -> None:
        """draw labed image

        Arguments
        ---------
        root_path : path
            dataset's path
        save_path : path
            where should the results be saved
        """
        self.draw_path = [os.path.dirname(root_path)]
        self.SAVE_PATH = save_path
        if not os.path.isdir(save_path) and save_path:
            os.makedirs(save_path, exist_ok=True)

        if os.path.exists(root_path) and os.path.isdir(root_path):
            cp = FolderStruct(self.image_name,self.label_name,self.mask_name)
            folder_structure = cp.folder_to_dict(root_path)
            self.draw_by_struct(folder_structure)
        else:
            print(f"{root_path} does not exist or isn't a directory.")


    def draw_annos_with_ditprj(self, PATH: Path, SAVE_PATH: Path) -> None:
        """
        Arguments
        ---------
        PATH: path
        SAVE_PATH: path
            where should the results be saved
        """
        from tqdm import tqdm

        from src.utils.ditproj_datasets import AISDataConverter, DITProject

        project_name = PATH.split(os.sep)
        CONFIG = project_name[-1] + ".ditprj"
        proj = DITProject(PATH, CONFIG)
        if not proj.label_type == "NULL":
            meta_data = proj.get_meta_data()
        else:
            return

        SAVE_PATH = os.path.join(SAVE_PATH, project_name[-1])
        # SAVE_PATH = PreProcessTool().get_save_path(PATH,SAVE_PATH)
        # if os.path.isdir(SAVE_PATH):
        #     shutil.rmtree(SAVE_PATH)
        # os.makedirs(SAVE_PATH, exist_ok=True)

        if meta_data[0][4]:
            OUT_PATH = os.path.join(SAVE_PATH, self.label_name[0])
        else:
            OUT_PATH = os.path.join(SAVE_PATH, self.mask_name[0])
            os.makedirs(OUT_PATH + "_palette",exist_ok=True)

        os.makedirs(OUT_PATH,exist_ok=True)
        if os.path.exists(os.path.join(SAVE_PATH, self.image_name[0])):
            shutil.rmtree(os.path.join(SAVE_PATH, self.image_name[0]))
        shutil.copytree(os.path.join(PATH, "Image"), os.path.join(SAVE_PATH, self.image_name[0]))

        annos = AISDataConverter.convert_dataset(meta_data, labels_map=proj.config['Trainer']['ListHulkParameter'][-1]['common_param']['labels_map'], dataset_root = PATH, output_path = OUT_PATH, is_encrypted = True)

        f = open(os.path.join(SAVE_PATH, "imgclass.txt"), "w+")

        SAVE_PATH = os.path.join(SAVE_PATH, self.annos_name)
        if not os.path.isdir(SAVE_PATH):
            os.makedirs(SAVE_PATH,exist_ok=True)

        classnumber = 0
        unique_class = {}

        for i in tqdm(range(len(meta_data)), desc = 'Drawing', ascii = True):
            IMG_PATH = os.sep.join([PATH, "Image", meta_data[i][1]])
            TXT_PATH = ''
            MASK_PATH = ''
            if meta_data[0][4]:
                TXT_PATH = os.sep.join([OUT_PATH, str(meta_data[i][1]).split(".")[0] + ".txt"])
            else:
                MASK_PATH = os.sep.join([OUT_PATH + f'_palette', str(meta_data[i][1]).split(".")[0] + ".png"])

            anno = Annotation(image_path = IMG_PATH,
                            bbox_path = TXT_PATH,
                            mask_path = MASK_PATH,
                            save_path = SAVE_PATH,
                            )
                #檢查
            out_imgs = AnnotationVisualizer.draw_annotations(anno, mask_box_split = False) #['original_image', 'annotated_image']
            if str(annos[i].class_id) not in unique_class and annos[i].class_id:
                print(meta_data[i][1] + ":" + str(annos[i].class_id), file=f)
                unique_class[str(annos[i].class_id)] = meta_data[i][1]
                classid = max(annos[i].class_id)
                classnumber = max(classnumber , classid)

        if meta_data[0][4]:
            print("total_class: " + str(classnumber+1), file=f)
        else:
            print("total_class: " + str(classnumber+2), file=f)
        print("path" + ":"+ PATH, file=f)
        f.close()

    def draw_annos_without_ditprj(self, PATH: Path, SAVE_PATH: Path) -> None:
        """
        Arguments
        ---------
        PATH: path
        SAVE_PATH: path
            where should the results be saved
        """
        from tqdm import tqdm

        from src.convert_check import CheckMatch
        pt = PreProcessTool()

        SAVE_PATH = os.path.join(SAVE_PATH, PATH.split(os.sep)[-1])
        #SAVE_PATH = pt.get_save_path(PATH,SAVE_PATH)
        # if os.path.isdir(SAVE_PATH):
        #     shutil.rmtree(SAVE_PATH)
        # os.makedirs(SAVE_PATH, exist_ok=True)

        match_list = CheckMatch().check_match(PATH)
        if match_list == []:
            return
        for item in match_list[PATH]:
            if os.path.exists(os.path.join(SAVE_PATH, item)):
                shutil.rmtree(os.path.join(SAVE_PATH, item))
            shutil.copytree(os.path.join(PATH, item), os.path.join(SAVE_PATH, item))

        img_name = os.listdir(os.path.join(PATH, match_list[PATH][0]))
        label_name = os.listdir(os.path.join(PATH, match_list[PATH][1]))

        f = open(os.path.join(SAVE_PATH, "imgclass.txt"), "w+")

        SAVE_PATH = os.path.join(SAVE_PATH, self.annos_name)
        if not os.path.isdir(SAVE_PATH):
            os.makedirs(SAVE_PATH,exist_ok=True)

        img_count = 0
        anno_count = 0
        classnumber = []
        unique_class = set()

        for i in tqdm(range(len(img_name)), desc = 'Drawing', ascii = True):
            checkimg = img_name[i].split(".")
            if checkimg[-1] == "txt":
                img_count += 1
            IMG_PATH = os.sep.join([PATH, match_list[PATH][0], img_name[i+img_count]])
            if checkimg[0] in label_name[i]:
                if '.txt' in label_name[i]:
                    TXT_PATH = os.sep.join([PATH, match_list[PATH][1], label_name[i+anno_count]])
                    MASK_PATH = ''
                    uc = set()
                    with open(TXT_PATH, 'r') as file:
                        lines = file.readlines()
                        for line in lines:
                            parts = line.strip().split()
                            if int(parts[0]) not in uc:
                                uc.add(int(parts[0]))

                else:
                    MASK_PATH = os.sep.join([PATH, match_list[PATH][1], label_name[i+anno_count]])
                    TXT_PATH = ''
                    mask = cv2.imread(MASK_PATH)
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    uc = set(np.unique(mask))
            else:
                anno_count += 1

            if len(MASK_PATH)>0 and not pt.is_rgb_image(MASK_PATH):
                pt.save_colorful_mask(MASK_PATH)
                MASK_PATH = MASK_PATH.replace(MASK_PATH.split(os.sep)[-2],(MASK_PATH.split(os.sep)[-2] + '_palette'))

            anno = Annotation(image_path = IMG_PATH,
                            bbox_path = TXT_PATH,
                            mask_path = MASK_PATH,
                            save_path = SAVE_PATH,
                            )
            out_imgs = AnnotationVisualizer.draw_annotations(anno, mask_box_split = False) #['original_image', 'annotated_image']

            if str(uc) not in unique_class and uc:
                unique_class.add(str(uc))
                for cn in uc:
                    if cn not in classnumber:
                        classnumber.append(cn)
                print(img_name[i] + ":" + str(uc), file=f)
        if not classnumber:
            classnumber.append(0)
        print("total_class:" + str(classnumber[-1]+1), file=f)
        print("path:" + PATH, file=f)
        f.close()

    #以下無用
    def create_color_mask_from_folder(self, mask_path: Path) -> None:
        """
        Argumetns
        ---------
        mask_path: path
        """
        train_path = os.path.join(mask_path, 'train')
        test_path = os.path.join(mask_path, 'test')
        save_train_path = os.path.join(train_path, 'masks')
        save_test_path = os.path.join(test_path, 'masks')
        if not os.path.exists(save_train_path):
            os.makedirs(save_train_path, exist_ok=True)
            os.makedirs(save_test_path, exist_ok=True)

        self.create_color_mask_from_mask(os.path.join(train_path,'masks'),save_train_path)
        self.create_color_mask_from_mask(os.path.join(test_path,'masks'),save_test_path)

    def create_color_mask_from_mask(self, mask_paths: List[Path], mask_save_path: Path) -> None:
        """
        mask_paths: List[path]
            to store masks' path
        mask_save_path: path
            where should the results be saved
        """
        import cv2
        import numpy as np
        color_map = np.array([
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

        add = 0
        class_index = {}
        unique_class = []
        for mask_path in os.listdir(mask_paths):
            mask_ori = cv2.imread(os.path.join(mask_paths, mask_path))
            mask = cv2.cvtColor(mask_ori, cv2.COLOR_BGR2GRAY)


            class_id = mask_path.split(os.sep)[-1].split('_')[1]

            if class_id not in unique_class:
                unique_class.append(class_id)
            unique_class.sort()
            for idx,class_idx in enumerate(unique_class):
                if class_idx == "00" or class_idx == "000":
                    class_index[class_idx] = 0
                    add += 1
                else:
                    class_index[class_idx] = idx - add + 1

            # generate a new color image that matches the dimentions of an existed image
            color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

            # traverse all pixel in the mask and fill in colors based on their categories
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    class_idx = mask[i, j]
                    if class_idx == 0:
                        color_mask[i, j] = color_map[class_idx % len(color_map)]
                    # 根据类别填充颜色
                    else:
                        color_mask[i, j] = color_map[class_index[class_id]]

            cv2.imwrite(os.path.join(mask_save_path,mask_path),color_mask)
        #return color_mask

    def draw_one_anno(self, image_path, label_path: Path, save_path: Path) -> None:
        """
        Arguments
        ---------
        image_path: path
            the path of the images
        label_path: path
            the path of the label
        save_path: path
            where should the results be saved
        """
        IMG_PATH = image_path
        SAVE_PATH = save_path
        if label_path.split('.')[-1] == "txt":
            TXT_PATH = label_path
            MASK_PATH = ''
        else:
            TXT_PATH = ''
            if os.path.exists(label_path.replace('masks','masks_palette')):
                MASK_PATH = label_path.replace('masks','masks_palette')
            else:
                MASK_PATH = label_path
        anno = Annotation(image_path = IMG_PATH,
                            bbox_path = TXT_PATH,
                            mask_path = MASK_PATH,
                            save_path = SAVE_PATH,
                            )
        out_imgs , uc = AnnotationVisualizer.draw_annotations(anno, mask_box_split = False) #['original_image', 'annotated_image']


class SplitDataset:
    def __init__(self):
        self.save_path = ""

    #CHV_dataset
    def split_CHV_dataset(self, root_folder: Path, image_name: str, label_name: str, txt_path: Path) -> None:
        father_path = os.path.dirname(root_folder)

        for file_name in os.listdir(txt_path):

            folder_name = os.path.join(root_folder, file_name.split('.')[0])
            os.makedirs(folder_name, exist_ok=True)
            image_folder = os.path.join(folder_name, image_name)
            os.makedirs(image_folder, exist_ok=True)
            label_folder = os.path.join(folder_name, "labels")
            os.makedirs(label_folder, exist_ok=True)

            # read image categories list
            with open(os.path.join(txt_path,file_name), 'r') as file:
                image_names = [line.strip() for line in file]

            for img_name in image_names:
                shutil.copy(os.path.join(father_path,img_name), image_folder)
                shutil.copy(os.path.join(father_path,img_name.replace(img_name.split('.')[-1],"txt").replace(image_name,label_name)), label_folder)


    #CVC-ClinicDB special structure
    def add_label(self, root_folder):
        train_image = os.path.join(root_folder, "train/images")
        test_image = os.path.join(root_folder, "test/images")
        train_label = os.path.join(root_folder, "train/labels")
        test_label = os.path.join(root_folder, "test/labels")
        os.makedirs(train_image, exist_ok=True)
        os.makedirs(test_image, exist_ok=True)
        os.makedirs(train_label, exist_ok=True)
        os.makedirs(test_label, exist_ok=True)
        train_root = os.path.join(root_folder,"train")
        test_root = os.path.join(root_folder,"test")
        label_path = os.path.join(root_folder,"Labels")
        for img in os.listdir(train_root):
            if img.split('.')[-1] in ['jpg','bmp','png','jpeg', 'tif']:
                shutil.copy(os.path.join(train_root, img), train_image)
                shutil.copy(os.path.join(label_path, img), train_label)
        for img in os.listdir(test_root):
            if img.split('.')[-1] in ['jpg','.bmp','png','jpeg', 'tif']:
                shutil.copy(os.path.join(test_root, img), test_image)
                shutil.copy(os.path.join(label_path, img), test_label)


    # the AITEX-Fabric folder structure is unique,
    # you need to handle root_folder separately for images and NODefect
    def split_AITEX_Fabric(self, root_folder: Path, image_name: str, label_name: str) -> None:
        da = DrawAnnos()
        other_image_name = "NODefect_images"
        self.split_data_with_label(root_folder, image_name, label_name)
        self.split_data_with_label(root_folder, other_image_name, label_name)
        da.create_color_mask_from_folder(root_folder)

    def split_data_with_label(self, root_folder: Path, image_name: str, label_name: str) -> None:
        from sklearn.model_selection import train_test_split

        image_folder = os.path.join(root_folder, image_name)
        label_folder = os.path.join(root_folder, label_name)

        image_folder = self.combine_images(image_folder)
        self.save_path = ""

        dataset = {}

        for image_file in sorted(os.listdir(image_folder)):
            for label_file in sorted(os.listdir(label_folder)):
                if image_file.split('.')[0] in label_file:
                    dataset[os.path.join(image_folder, image_file)] = os.path.join(label_folder, label_file)

        class_counts = {}
        for label in dataset.values():
            label = label.split(os.sep)[-1].split('_')[1]
            class_counts[label] = class_counts.get(label, 0) + 1

        train_root = os.path.join(root_folder, "train")
        test_root = os.path.join(root_folder, "test")
        os.makedirs(train_root, exist_ok=True)
        os.makedirs(test_root, exist_ok=True)

        # biuild train and test dataset
        train_folder = os.path.join(train_root, "images")
        test_folder = os.path.join(test_root, "images")
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)
        train_lb_folder = os.path.join(train_root, "masks")
        test_lb_folder = os.path.join(test_root, "masks")
        os.makedirs(train_lb_folder, exist_ok=True)
        os.makedirs(test_lb_folder, exist_ok=True)

        min_samples_per_class = 2

        # for each class, split the samples into training and testing sets, ensuring that each set contains at least two samples
        for label, count in class_counts.items():
            # check if samples are enough
            if count < min_samples_per_class:
                continue

            label_images = [image for image, lbl in dataset.items() if lbl.split(os.sep)[-1].split('_')[1] == label]
            # split datasets into train and test dataset, and ensure every set have both of them
            train_images, test_images = train_test_split(label_images, test_size=0.2, random_state=42)

            # copy the training set to the corresponding folder
            for image_path in train_images:
                shutil.copy(image_path, train_folder)
                shutil.copy(dataset[image_path], train_lb_folder)

            # copy the Testing set to the corresponding folder
            for image_path in test_images:
                shutil.copy(image_path, test_folder)
                shutil.copy(dataset[image_path], test_lb_folder)

        shutil.rmtree(image_folder)

    def combine_images(self, root_folder: Path) -> Path:
        if self.save_path == "":
            self.save_path = os.path.join(os.path.dirname(root_folder), "all_images")
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
        for file in os.listdir(root_folder):
            folder_path = os.path.join(root_folder, file)
            file_name, extension = os.path.splitext(file)
            if extension in ['.jpg','.bmp','.png','.jpeg']:
                shutil.copy(folder_path, self.save_path)
            elif extension == '':
                self.combine_images(folder_path)
        return self.save_path


if __name__ == "__main__":
    import argparse

    #要有image跟label
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default=r"C:\Users\edmond_huang\Desktop\original_dataset", help="your path")
    parser.add_argument("-sp", "--save_path", type=str, default=r"C:\Users\edmond_huang\Desktop\result", help="your save path")
    parser.add_argument("-m","--mode", type=str, default="draw", help="your image folder name")

    args = parser.parse_args()

    if args.mode == 'struct':
        fs = FolderStruct()
        folder_struct = fs.folder_to_dict(args.path)
    if args.mode == 'draw':
        '''
        path:       origin dataset
        save path:  output folder
        '''
        df = DrawAnnos()
        df.draw_by_folder(args.path,args.save_path)
    if args.mode == "convert":
        '''
        path:       output folder
        save path:  convert folder
        '''
        pt = PreProcessTool()
        pt.convert_dataset(args.path, args.save_path)

    print('end')
