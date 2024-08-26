import os
import sys

sys.path.append(r'../annotation_utils')
from enum import IntEnum, unique
from typing import List, Optional

import cv2
import numpy as np

from .alg_io_utils import load_image, load_mask, parse_file_extension


#### Annotation ####
# Mask: are grayscaled images, values of pixels represent their class_id.
# BBox: default yolo format
class Bbox:
    @unique
    class BboxType(IntEnum):
        XY_CENTER_WH = 0 # (x_center, y_center, w, h)
        XY_TL_WH = 1 # (x_top_left, y_top_left, w, h)
        XY_CORNER = 2 # (x_top_left, y_top_left, x_bottom_right, y_bottom_right)

    def __init__(self, xmin:int=-1, ymin:int=-1, xmax:int=-1, ymax:int=-1, class_id:int=0, confidence:float=0):
            self.curr_type = Bbox.BboxType.XY_CORNER
            self.xmin = int(xmin)
            self.ymin = int(ymin)
            self.xmax = int(xmax)
            self.ymax = int(ymax)
            self.class_id = int(class_id)
            self.confidence = float(confidence)

            # Not
            self._img_width = 0
            self._img_height = 0

    def set_image_width_hegith(self, image_width:int, image_height:int):
        assert (image_width * image_height ) >  0, 'Wrong input.'
        assert self.xmax < image_width, 'Wrong input: image width < xmax'
        assert self.ymax < image_height, 'Wrong input: image height < ymax'

        self._img_width = image_width
        self._img_height = image_height

    def __str__(self):
        return "{} {} {} {} {}".format(self.class_id, self.xmin, self.ymin, self.xmax, self.ymax)

    def return_lt_rb(self):
        return (self.xmin, self.ymin), (self.xmax, self.ymax)

    def return_cnt_wh(self, image_width:int = None, image_height:int = None, normalize = True)->List[int]:
        if not bool(image_width) and bool(image_height):
            self.set_image_width_hegith(image_width, image_height)

        width = self.xmax - self.xmin
        height = self.ymax - self.ymin
        x_center = self.xmin + int(width / 2)
        y_center = self.ymin + int(height / 2)

        if normalize:
            width = float(width) / self._img_width
            height = float(height) / self._img_height
            x_center = float(x_center) / self._img_width
            y_center = float(y_center) / self._img_height

        return [self.class_id, x_center, y_center, width, height, self.confidence]

    def parse_bbox_from_line(self, line:str, type:BboxType, is_normalized:bool, image_width:int, image_height:int):
        if line is None:
            raise ValueError('line is empty')

        class_id = -1
        xmin, ymin, xmax, ymax = -1, -1, -1, -1
        confidence = 0.0
        self.set_image_width_hegith(image_width, image_height)

        if type == self.BboxType.XY_CENTER_WH:
            class_id, x_center, y_center, width, height, confidence = (line.split() + [0.0])[:6]

            x_center = float(x_center)
            y_center = float(y_center)
            width = float(width)
            height = float(height)
            class_id = int(class_id)
            confidence = float(confidence)

            xmin = x_center - width/2
            xmax = x_center + width/2
            ymin = y_center - height/2
            ymax = y_center + height/2

        elif type == self.BboxType.XY_TL_WH:
            class_id, x_tl, y_tl, width, height, confidence = (line.split() + [0.0])[:6]

            xmin = float(x_tl)
            ymin = float(y_tl)
            width = float(width)
            height = float(height)
            class_id = int(class_id)
            confidence = float(confidence)

            xmax = xmin + width
            ymax = ymin + height

        elif type == self.BboxType.XY_CORNER:
            class_id, x_tl, y_tl, x_br, y_br, confidence = (line.split() + [0.0])[:6]

            xmin = float(x_tl)
            ymin = float(y_tl)
            xmax = float(x_br)
            ymax = float(y_br)
            class_id = int(class_id)
            confidence = float(confidence)

        if is_normalized:
            xmin *= image_width
            ymin *= image_height
            xmax *= image_width
            ymax *= image_height

        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)

        # Check boundary
        if xmin < 0:
            print('xmin {} < 0. calibrate xmin to 0'.format(xmin))
            xmin = 0
        if ymin < 0:
            print('ymin {} < 0. calibrate ymin to 0'.format(ymin))
            ymin = 0
        if xmax >= image_width:
            print('xmax {} >= image_width {}. calibrate xmax to {}'.format(xmax, image_width, image_width-1))
            xmax = image_width -1
        if ymax >= image_height:
            print('ymax {} >= image_height {}. calibrate ymax to {}'.format(ymax, image_height, image_height-1))
            ymax = image_height -1

        if (xmin > xmax or ymin > ymax) or (class_id < 0) or (confidence < 0.0 or confidence > 1.0):
            raise ValueError('Wrong Annotation. class_id:{} xmin:{} ymin:{} xmax:{} ymax:{} confidecne:{}'.format(
                class_id, xmin, ymin, xmax, ymax, confidence))

        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.class_id = class_id
        self.confidence = confidence

        return self

class MappingTable:

    NOT_FOUND_ID = 5487

    def __init__(self) -> None:
        self.table = []

    def __call__(self,
                 class_id: int,
                 pos_neg: int,
                 ) -> int:
        unique_id = self._compare(class_id, pos_neg)
        return unique_id

    def _compare(self,
                 class_id: int,
                 pos_neg: int,
                 ) -> int:
        result = next((data['unique_id'] for data in self.table if
                       (data['class_id'] == class_id and data['pos_neg'] == pos_neg)), self.NOT_FOUND_ID)
        return result

    def register(self,
                 class_id: int,
                 pos_neg: int
                 ) -> bool:
        # check value
        class_id = max(class_id, -1)

        # pos_neg = -1, 0, 1
        pos_neg = min(max(pos_neg, -1), 1)

        if class_id == -1 and pos_neg == -1:
            return False
        if self._compare(class_id, pos_neg) != self.NOT_FOUND_ID:
            return False

        self.table.append(
            {
                'class_id' : class_id,
                'pos_neg'  : pos_neg,
                'unique_id': -1
            }
        )
        return True

    def generate(self):
        self.table.sort(key=lambda x: (x['class_id'], x['pos_neg']))
        for i in range(len(self.table)):
            self.table[i]['unique_id'] = i

    def size(self):
        return len(self.table)

    def reset(self):
        self.table = []

class Annotation:
    def __init__(self, class_id:int=-1, pos_neg:int=-1, image_path:str=None, bbox_path:str=None, mask_path:str=None, save_path:str=None, annos:list=None) -> None:
        self.class_id = class_id if class_id >= 0 else -1
        if pos_neg <= -1:
            self.pos_neg = -1
        elif pos_neg >= 1:
            self.pos_neg = 1
        else:
            self.pos_neg = 0
        self.image_path = image_path if (image_path and image_path != '') else None
        self.bbox_path = bbox_path if (bbox_path and bbox_path != '') else None
        self.mask_path = mask_path if (mask_path and mask_path != '') else None
        self.mapping_table = None

        self.annos = annos
        self.save_path = save_path if (save_path and save_path !='') else None

    def is_empty(self)->bool:
        return not bool(self.image_path)

    def bbox_exists(self)->bool:
        return (self.bbox_path is not None) and (os.path.isfile(self.bbox_path))

    def mask_exists(self)->bool:
        return (self.mask_path is not None) and (os.path.isfile(self.mask_path))

    def read_image(self, grayscale: bool = False) -> Optional[np.ndarray]:
        image = None
        if self.image_path:
            image = load_image(self.image_path, grayscale)
        return image

    def read_class_id(self)->int:
        class_id = None
        if self.mapping_table:
            class_id = self.mapping_table(self.class_id, self.pos_neg)
        return class_id

    def read_bbox(self, type:Bbox.BboxType=Bbox.BboxType.XY_CENTER_WH, is_normalized:bool=True, image_width:int=512, image_height:int=512)->List[Bbox]:
        '''
        Supported 3 types of bbox annotation.
        Parse bboxs to unified format (x_center, y_center, width, height)
        '''
        bboxes = []
        if self.bbox_path is None or self.bbox_path == '':
            raise ValueError('Filepath is empty')

        if not os.path.isfile(self.bbox_path):
            raise ValueError('Filepath {} not existed'.format(self.bbox_path))

        file_extension = parse_file_extension(self.bbox_path)
        if file_extension != '.txt':
            print('File extension is not txt: {}'.format(self.bbox_path))
            return bboxes
            #raise ValueError('File extension is not txt: {}'.format(self.bbox_path))

        if image_width <= 0 or image_height <= 0:
            raise ValueError('image_width {} or image_height {} less than 0'.format(image_width, image_height))

        with open(file=self.bbox_path, mode='rb') as fin:
            data = fin.read()

        data = data.decode('utf-8')

        lines = data.split('\n')
        for line in lines:
            line = line.strip() # remove /n and whitspace
            if line == '':
                continue
            bbox = Bbox()
            bbox.parse_bbox_from_line(line, type, is_normalized, image_width, image_height)
            bboxes.append(bbox)

        return bboxes

    def read_mask(self) -> Optional[np.ndarray]:
        mask = None
        rgb_mask = None
        if self.mask_path:
            mask, rgb_mask = load_mask(self.mask_path)
        return mask, rgb_mask

    def set_mapping_table(self, mapping_table)->None:
        self.mapping_table = mapping_table

class AnnotationVisualizer:
    PASCAL_COLOR_LIST = np.array([
    [0, 0, 0],      #黑色
    [128, 0, 0],    #酒紅
    [0, 128, 0],    #綠色
    [128, 128, 0],  #黃綠社
    [0, 0, 128],    #深藍
    [128, 0, 128],  #紅紫
    [0, 128, 128],  #藍綠
    [128, 128, 128],#灰色
    [64, 0, 0],     #咖啡色
    [192, 0, 0],    #亮紅
    [64, 128, 0],   #亮綠色
    [192, 128, 0],  #淺咖啡
    [64, 0, 128],   #深紫
    [192, 0, 128],  #桃紅
    [64, 128, 128], #灰藍綠
    [192, 128, 128],#灰粉
    [0, 64, 0],     #深綠
    [128, 64, 0],   #中淺咖啡
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128]    #淺藍綠
    ], dtype = np.uint8)

    @classmethod
    def _get_pascal_color(cls, class_id:int, bgr_order = True)->np.array:
        class_id = 0 if class_id < 0 else int(class_id)

        col = cls.PASCAL_COLOR_LIST[class_id % len(cls.PASCAL_COLOR_LIST)]
        return col[::-1] if bgr_order else col

    @classmethod
    def draw_bboxes(cls, bgr_img:np.array, bboxes:List[Bbox])->np.ndarray:
        if not bool(bboxes):
            return bgr_img

        for box in bboxes:
            lt = (box.xmin, box.ymin)
            rb = (box.xmax, box.ymax)
            if box.class_id == 0:
                box.class_id = len(cls.PASCAL_COLOR_LIST) - 1
            col = cls._get_pascal_color(box.class_id).tolist()
            _ = cv2.rectangle(bgr_img, lt, rb, col, 4)

        return bgr_img

    @classmethod
    def draw_mask(cls, bgr_img:np.array, mask:np.array, rgb_mask:np.array)->np.ndarray:
        if mask.max() == 0:
            return bgr_img

        temp_mask = np.zeros_like(bgr_img, dtype = np.uint8)

        class_ids = np.unique(mask)
        for class_id in class_ids:
            if class_id == 0:
                continue
            temp_mask[mask == class_id] = cls._get_pascal_color(class_id)

        bgr_img2 = cv2.addWeighted(bgr_img, 1, rgb_mask, 1, 0)

        return bgr_img2

    @classmethod
    def draw_annotations(cls, anno:Annotation, img_gray_scale:bool = False, mask_box_split = False, show_imgs:bool = True)->List[np.array]:
        unique_classes = set()
        img_list = []
        if anno.is_empty():
            return img_list

        img_ori = anno.read_image(grayscale = img_gray_scale)
        if img_gray_scale:
            img_ori = np.stack((img_ori,)*3, axis = -1)
        img_list.append(img_ori)
        try:
            img_0 = img_ori.copy()
        except:
            return img_list,unique_classes


        H, W = img_ori.shape[:2]
        bboxes = anno.read_bbox(image_height = H, image_width = W) if anno.bbox_exists() else None
        mask, rgb_mask = anno.read_mask() if anno.mask_exists else None

        if anno.bbox_exists():
            img_0 = cls.draw_bboxes(img_0, bboxes)
            for i in bboxes:
                if i.class_id not in unique_classes:
                    unique_classes.add(i.class_id)
            img_list.append(img_0)

        if mask is not None:
            if not mask_box_split:
                img_0 = cls.draw_mask(img_0, mask, rgb_mask)
                if not bool(bboxes):
                    img_list.append(img_0)
                else:
                    img_list[1] = img_0
            else:
                img_1 = img_ori.copy()
                img_1 = cls.draw_mask(img_1, mask, rgb_mask)
                img_list.append(img_1)

            temp_mask = np.zeros_like(img_0, dtype = np.uint8)
            class_ids = np.unique(mask)
            for class_id in class_ids:
                if class_id == 0:
                    continue
                temp_mask[mask == class_id] = cls._get_pascal_color(class_id)

            unique_classes = set(np.unique(mask))

        if show_imgs:
            try:
                for (i, curr_img) in enumerate(img_list):
                    if(i!=0):
                        cv2.imwrite(anno.save_path + "\\" + anno.image_path.split("\\")[-1], curr_img)
                cv2.waitKey(0)
            except Exception as e:
                Warning(e)
            cv2.destroyAllWindows()

        return img_list, unique_classes
