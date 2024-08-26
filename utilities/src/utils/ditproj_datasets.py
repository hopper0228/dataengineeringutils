import json
import os
import sys
from enum import Enum

sys.path.append(r'../')
sys.path.append(r'../utils')
import warnings
from pathlib import Path
from typing import Dict, List, Set

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from aisv_label_processor.app.python.aisv_label_processor_pywrapper import AisvLabelProcessor

from .annotation_utils.alg_annotation_utils import Annotation, Bbox


class ImageTask(Enum):
        CLASSIFICATION = 'Classification'
        SEGMENTATION= 'Segmentation'
        OBJECT_DETECTION= 'Object_Detection'

def parse_meta_files(dataset_root: str, meta_filenames: List[str]) -> List[List[str]]:
    """
    parse meta files (.csv).
    Args:
        dataset_root: path to dataset.
        meta_filenames: meta filename (.csv).

    Returns:
        meta data. format: [[row_id, image_path, class_id, pos_neg, bbox_path, mask_path]...]

    """
    output = []
    for meta_filename in meta_filenames:
        csv_path = os.path.join(dataset_root, meta_filename)

        with open(csv_path, newline='') as f:
            rows = f.readlines()

        # filter header row
        if 'index' in rows[0]:
            rows = rows[1:]

        # filter empty row and empty line
        for i, row in enumerate(rows):
            row = row.strip()
            if not all(c in ', ' for c in row):
                data = row.split(',')
                if len(data) == 6:
                    data[1] = Path(data[1]).as_posix()
                    data[4] = Path(data[4]).as_posix()
                    data[5] = Path(data[5]).as_posix()
                    output.append(data)
                else:
                    warnings.warn(f'{row} is in wrong format')
    return output

class DITProject:
    """
    Data structure of DIT project
    """
    config = None
    dataset_root: str
    image_root: str
    label_root: str
    label_type: ImageTask
    image_num: int
    project_name: str
    class_names: List[str]
    image_items_dict: Dict[str, List[Dict]]

    def __init__(self, dataset_root: str, config_name: str = None):
        self.dataset_root = dataset_root

        if config_name is None:
            # find .ditprj config file
            top_filenames = os.listdir(dataset_root)

            for filename in top_filenames:
                if filename.endswith('.ditprj'):
                    self.config_path = os.path.join(dataset_root, filename)
                    break
        else:
            self.config_path = os.path.join(dataset_root, config_name)
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
            self.config = self.config['InfoList']

        self.image_root = os.path.join(dataset_root, self.config['Labeller']['ImagePath'])
        self.label_root = os.path.join(dataset_root, self.config['Labeller']['MaskPath'])

        label_type = self.config['Labeller']['LabellerData']['LabelType']
        if label_type == 'Classification':
            self.label_type = ImageTask.CLASSIFICATION
        elif label_type == 'Segmentation':
            self.label_type = ImageTask.SEGMENTATION
        elif label_type == 'Object_Detection':
            self.label_type = ImageTask.OBJECT_DETECTION
        else:
            print(f'{label_type} is not allowed. Expected: Classification, Segmentation or Object_Detection.')
            self.label_type = "NULL"
            pass
            #raise ValueError(f'{label_type} is not allowed. Expected: Classification, Segmentation or Object_Detection.')

        self.image_num = self.config['Labeller']['ImageNum']
        self.project_name = self.config['ProjectManager']['ProjectName']
        self.class_names = [class_item['ClassName'] for class_item in
                            self.config['Labeller']['LabellerData']['ClassItem']]

        self.image_items_dict = self.config['Labeller']['LabellerData']['DictImageItem']

    def _get_data_name_list(self, key: str) -> Set[str]:
        data_name_list = self.config['Trainer']['ListHulkParameter'][-1]['common_param'][key]
        return {Path(p).as_posix() for p in data_name_list}

    def get_meta_data(self, mode: str = 'all') -> List[List[str]]:
        """
        Get meta data. format: [[row_id, image_path, class_id, pos_neg, bbox_path, mask_path]...]
        Returns:
            meta_data: List[List[str]]
        """
        if mode not in ('all', 'train', 'val'):
            raise ValueError(f'mode={mode} is not allowed. Expected: all, train, and val.')

        if self.label_type == ImageTask.CLASSIFICATION:
            meta_data = parse_meta_files(self.label_root, ['LabellerInfo.csv'])
            # filter meta_data
            if mode == 'train':
                criteria = self._get_data_name_list('train_data_name_list')
            if mode == 'val':
                criteria = self._get_data_name_list('val_data_name_list')
            if mode == 'all':
                criteria = self._get_data_name_list('train_data_name_list')
                criteria.update(self._get_data_name_list('val_data_name_list'))
            meta_data = list(filter(lambda x: x[1] in criteria, meta_data))
        else:
            meta_data = []
            row_id = 0
            for key in self.image_items_dict:
                for image_item in self.image_items_dict[key]:
                    image_path = Path(image_item['ImageName']).as_posix()

                    # filter meta_data
                    if mode == 'train' and image_path not in self.train_data_name_list:
                        continue
                    if mode == 'val' and image_path not in self.val_data_name_list:
                        continue

                    class_id = -1
                    pos_neg = -1

                    base_name = image_path.rpartition('.')[0]

                    if self.label_type == ImageTask.OBJECT_DETECTION:
                        bbox_path = base_name + '.txt'
                    else:
                        bbox_path = ''

                    if self.label_type == ImageTask.SEGMENTATION:
                        # DIT project store mask as file with .tmp extension
                        mask_path = base_name + '.tmp'
                    else:
                        mask_path = ''
                    meta_data.append([row_id, image_path, class_id, pos_neg, bbox_path, mask_path])

                    row_id += 1

        return meta_data

#### Convert all files
class ImageSample:
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

    def __init__(self, image, image_name, bboxes = None, mask = None):
        self.image = image
        self.image_name = image_name
        self.class_id = None
        self.mask = mask
        self.bboxes = bboxes

    @classmethod
    def save_colorful_mask(cls, mask: np.ndarray, name: str)->None:
        '''
        mask should be gray-scaled
        '''
        img = Image.fromarray(mask.astype(np.uint8), mode = 'L')
        img.save(name)
        img.putpalette(cls.PASCAL_COLORS)
        cls.class_ids = np.unique(img)
        palette_folder = name.split(os.sep)[-2]
        img.save(name.replace(palette_folder, palette_folder + '_palette'))


    @classmethod
    def save_bbox_cnt_wh(cls, bboxes:List[Bbox], name: str)->None:
        with open(name, 'a') as file:
            for bbox in bboxes:
                cls_id, x_cnt, y_cnt, w, h, _ = bbox.return_cnt_wh()
                line = f"{cls_id} {x_cnt} {y_cnt} {w} {h}\n"

                file.write(line)

    def write_out_annotatnion(self, output_path:str, name:str = None, bbox_suffix = '.txt', mask_suffix:str = '.png'):
        name = self.image_name if not name else name

        written = False
        if self.bboxes is not None and bool(self.bboxes):
            path = os.path.join(output_path, name + bbox_suffix)
            self.save_bbox_cnt_wh(self.bboxes, path)

            written = written or True

        if self.mask is not None:
            path = os.path.join(output_path, name + mask_suffix)
            self.save_colorful_mask(self.mask, path)

            written = written or True

        return written

class AISDataConverter:
    CLASS_ID_NULL_TAG = -1
    POS_NEG_NULL_TAG = -1
    ANNOTATION_PATH_NULL_TAG = ''

    @classmethod
    def read_bbox(cls, bbox_path, image_width:int, image_height:int, is_encrypted: bool = False):
        """
        create bbox from file.
        Args:
            bbox_path: path to bbox file.
            fixed_class_id: replace class_id by fixed_class_id.
            is_encrypted: bbox_path is encrypted or not. Default: False.

        Returns:

        """

        bboxes = []
        if is_encrypted:
            bbox_info = AisvLabelProcessor.decrypt_string_data(bbox_path)
        else:
            with open(file=bbox_path, mode='r') as fin:
                bbox_info = fin.read()

        lines = bbox_info.split('\n')
        for line in lines:
            line = line.strip()  # remove /n and whitespace
            if line == '':
                continue
            box = Bbox()
            box.parse_bbox_from_line(line, Bbox.BboxType.XY_CENTER_WH, is_normalized = True, image_width = image_width, image_height = image_height)
            bboxes.append(box)

        return bboxes

    @classmethod
    def convert_dataset(
            cls,
            meta_data: List[List[str]],
            labels_map: Dict[str, int] = {},
            dataset_root: str = None,
            image_root: str = None,
            label_root: str = None,
            fixed_class_id: int = None,
            is_encrypted: bool = True,
            output_path = r''
    ):
        """
        create dataset and convert dataset into AISDataHub format.
        Args:
            meta_data: meta data. format: [[row_id, image_path, class_id, pos_neg, bbox_path, mask_path]...]
            labels_map: labels table. key:value, key: <class_id>_<pos_neg>; value: specific label value.
            dataset_root: path to dataset root
            image_root: path to image root.
            label_root: path to label root.
            has_class_id: meta_file has classification task.
            has_mask: meta_file has segmentation task.
            has_bbox: meta_file has object detection task.
            fixed_class_id: replace class_id by fixed_class_id.
            is_encrypted: bbox_path is encrypted or not. Default: False.

        Returns:

        """
        if dataset_root is None and (image_root is None or label_root is None):
            raise ValueError('One of dataset_root or (image_root, label_root) should be filled.')

        if image_root is None and label_root is None:
            image_root = os.path.join(dataset_root, 'Image')
            label_root = os.path.join(dataset_root, 'Label')

        annotations = []
        f = open(dataset_root + r"\imgclass.txt","w+")

        with tqdm(total=len(meta_data), desc='Data Processing', ascii = True) as pbar:
            for row_id, image_path, class_id, pos_neg, bbox_path, mask_path in meta_data:
                unique_classes = set()

                if image_path.startswith('.'):
                    # remove ./ path in public dataset
                    image_path = os.path.join('', *image_path.split('/')[1:])
                image_path = os.path.join(image_root, image_path)

                image = cv2.imread(image_path)
                if image is None:
                    break
                W, H = image.shape[:2]
                img_name = os.path.splitext(image_path.split(os.path.sep)[-1])[0]

                annotation = ImageSample(image=image, image_name=img_name)
                if class_id != cls.CLASS_ID_NULL_TAG:
                    class_label = str(class_id) + ('_' + str(pos_neg) if pos_neg != cls.POS_NEG_NULL_TAG else '')
                    if any(labels_map):
                        if class_label in labels_map:
                            classification_class_id = labels_map[class_label]
                        else:
                            warnings.warn(
                                f'<class_id>_<pos_neg>={class_label} is not assigned before. Use <class_id>={class_id} as label.')
                            classification_class_id = int(class_id)
                    annotation.class_id = classification_class_id

                if bbox_path != cls.ANNOTATION_PATH_NULL_TAG:
                    if bbox_path.startswith('.'):
                        # remove ./ path in public dataset
                        bbox_path = os.path.join('', *bbox_path.split('/')[1:])
                    bbox_path = os.path.join(label_root, bbox_path)
                    bboxes = cls.read_bbox(bbox_path, is_encrypted=is_encrypted, image_width = W, image_height = H)
                    annotation.bboxes = bboxes
                    if bboxes:
                        for i in range(len(bboxes)):
                            unique_classes.add(bboxes[i].class_id)
                        print(annotation.image_name + ":" + str(unique_classes) + "\n", file=f)

                if mask_path != cls.ANNOTATION_PATH_NULL_TAG:
                    if mask_path.startswith('.'):
                        # remove ./ path in public dataset
                        mask_path = os.path.join('', *mask_path.split('/')[1:])
                    mask_path = os.path.join(label_root, mask_path)
                    if is_encrypted:
                        mask = AisvLabelProcessor.decrypt_mask_data(mask_path)
                    else:
                        mask = cv2.imread(mask_path)
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

                    if fixed_class_id is not None:
                        mask[np.where(mask != 0)] = fixed_class_id + 1 if fixed_class_id == 0 else fixed_class_id

                    annotation.mask = mask

                    unique_classes = set(np.unique(annotation.mask))
                    if 0 in unique_classes:
                        unique_classes.remove(0)
                    unique_classes = {x-1 for x in unique_classes}
                    print(annotation.image_name + ":" + str(unique_classes) + "\n", file=f)

                annotation.class_id = unique_classes
                annotations.append(annotation)
                pbar.update()
                if os.path.exists(output_path):
                    annotation.write_out_annotatnion(output_path, name = img_name)

            f.close()


            return annotations
