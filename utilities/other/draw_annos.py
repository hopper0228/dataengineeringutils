import os
import shutil

from tqdm import tqdm

#### Visualization Region
from utils.annotation_utils.alg_annotation_utils import Annotation, AnnotationVisualizer
#### Transfer project files and annotations
from utils.ditproj_datasets import AISDataConverter, DITProject

#from ..preprocess_tool import PreProcessTool

class DrawAnnos:
    def __init__(self) -> None:
        self.path_del_num = 3

    def draw_annos_with_ditprj(self,PATH:str, SAVE_PATH:str):

        project_name = PATH.split("\\")
        CONFIG = PATH + "\\" + project_name[len(project_name)-1] + ".ditprj"
        proj = DITProject(PATH, CONFIG)
        if not proj.label_type == "NULL":
            meta_data = proj.get_meta_data()
        else:
            return

        OUT_PATH = PATH + "\Annosvis_palette"
        if not os.path.isdir(OUT_PATH):
            os.mkdir(OUT_PATH)
        if not os.path.isdir(PATH + "\Annosvis"):
            os.mkdir(PATH + "\Annosvis")

        annos = AISDataConverter.convert_dataset(meta_data, labels_map=proj.config['Trainer']['ListHulkParameter'][-1]['common_param']['labels_map'], dataset_root = PATH, output_path = OUT_PATH, is_encrypted = True)

        for i in range(self.path_del_num, len(project_name)):
            SAVE_PATH = SAVE_PATH + "\\" + str(project_name[i])
            if not os.path.isdir(SAVE_PATH):
                os.mkdir(SAVE_PATH)

        f = open(SAVE_PATH + "\\imgclass.txt", "w+")

        if os.path.exists(SAVE_PATH + "\\images"):
            shutil.rmtree(SAVE_PATH + "\\images")
            if meta_data[0][4]:
                shutil.rmtree(SAVE_PATH + "\\labels")
            else:
                shutil.rmtree(SAVE_PATH + "\\masks")
                shutil.rmtree(SAVE_PATH + "\\masks_palette")
        shutil.copytree(PATH + "\\Image", SAVE_PATH + "\\images")
        if meta_data[0][4]:
            shutil.copytree(PATH + "\\Annosvis", SAVE_PATH + "\\labels")
        else:
            shutil.copytree(PATH + "\\Annosvis", SAVE_PATH + "\\masks")
            shutil.copytree(PATH + "\\Annosvis_palette", SAVE_PATH + "\\masks_palette")

        SAVE_PATH = SAVE_PATH + "\Annosvis"
        if not os.path.isdir(SAVE_PATH):
            os.mkdir(SAVE_PATH)

        classnumber = 0
        unique_class = {}

        for i in tqdm(range(len(meta_data)), desc='drawing'):
            IMG_PATH = PATH + "\\Image\\" + meta_data[i][1]
            TXT_PATH = ''
            MASK_PATH = ''
            if meta_data[0][4]:
                TXT_PATH = PATH + "\\Annosvis\\" + str(meta_data[i][1]).split(".")[0] + ".txt"
            else:
                MASK_PATH = PATH + "\\Annosvis_palette\\" + str(meta_data[i][1]).split(".")[0] + ".png"

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

        print("total_class: " + str(classnumber), file=f)
        print("image number: " + str(len(meta_data)), file=f)
        print("path" + ":"+ PATH, file=f)
        f.close()

    def draw_annos_without_ditprj(self, PATH:str, SAVE_PATH:str):
        #pt = PreProcessTool()
        project_name = PATH.split("\\")

        for i in range(self.path_del_num, len(project_name)):
            SAVE_PATH = SAVE_PATH + "\\" + str(project_name[i])
            if not os.path.isdir(SAVE_PATH):
                os.mkdir(SAVE_PATH)

        temp_folder_name = os.listdir(PATH)
        folder_name = []
        for i in temp_folder_name:
            if os.path.isdir(PATH + "\\" + i):
                if "image" in i.lower() or "label" in i.lower() or "mask" in i.lower():
                    folder_name.append(i)
                    if os.path.exists(SAVE_PATH + "\\" + i):
                        shutil.rmtree(SAVE_PATH + "\\" + i)
                    shutil.copytree(PATH + "\\" + i, SAVE_PATH + "\\" + i)

        img_name = os.listdir(PATH + "\\" + folder_name[0])
        label_name = os.listdir(PATH + "\\" + folder_name[1])

        f = open(SAVE_PATH + "\\imgclass.txt", "w+")

        SAVE_PATH = SAVE_PATH + "\Annosvis"
        if not os.path.isdir(SAVE_PATH):
            os.mkdir(SAVE_PATH)

        img_count = 0
        anno_count = 0
        max_class = []
        unique_class = set()

        for i in range(len(img_name)):
            checkimg = img_name[i].split(".")
            if checkimg[-1] == "txt":
                img_count += 1
            IMG_PATH = PATH + "\\" + folder_name[0] + "\\" + img_name[i+img_count]
            if checkimg[0] in label_name[i]:
                if '.txt' in label_name[i]:
                    TXT_PATH = PATH + "\\" + folder_name[1] + "\\" + label_name[i+anno_count]
                    MASK_PATH = ''
                else:
                    MASK_PATH = PATH + "\\" + folder_name[1] + "\\" + label_name[i+anno_count]
                    TXT_PATH = ''
            else:
                anno_count += 1

            # if len(MASK_PATH)>0 and not pt.is_rgb_image(MASK_PATH):
            #     pt.save_colorful_mask(MASK_PATH)

            anno = Annotation(image_path = IMG_PATH,
                            bbox_path = TXT_PATH,
                            mask_path = MASK_PATH,
                            save_path = SAVE_PATH,
                            )
            out_imgs , uc = AnnotationVisualizer.draw_annotations(anno, mask_box_split = False) #['original_image', 'annotated_image']
            if str(uc) not in unique_class and uc:
                unique_class.add(str(uc))
                for cn in uc:
                    if cn not in max_class:
                        max_class.append(cn)
                print(img_name[i] + ":" + str(uc), file=f)

        print("total_class:" + str(len(max_class)), file=f)
        print("path" + ":"+ PATH, file=f)
        f.close()


    def create_color_mask_from_folder(self,mask_path):
        train_path = os.path.join(mask_path, 'train')
        test_path = os.path.join(mask_path, 'test')
        save_train_path = os.path.join(train_path, 'masks')
        save_test_path = os.path.join(test_path, 'masks')
        if not os.path.exists(save_train_path):
            os.mkdir(save_train_path)
            os.mkdir(save_test_path)

        self.create_color_mask_from_mask(os.path.join(train_path,'masks'),save_train_path)
        self.create_color_mask_from_mask(os.path.join(test_path,'masks'),save_test_path)

    def create_color_mask_from_mask(self,mask_paths, mask_save_path):
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

            # 创建一个空的彩色掩码图像，与原始 mask 具有相同的形状
            color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

            # 遍历 mask 中的每个像素，根据类别填充颜色
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

    def draw_one_anno(self, image_path, label_path, save_path):
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

    def draw_tiny(self, PATH:str, SAVE_PATH:str):
            project_name = PATH.split("\\")

            for i in range(self.path_del_num, len(project_name)):
                SAVE_PATH = SAVE_PATH + "\\" + str(project_name[i])
                if not os.path.isdir(SAVE_PATH):
                    os.mkdir(SAVE_PATH)

            temp_folder_name = os.listdir(PATH)
            folder_name = []
            for i in temp_folder_name:
                if os.path.isdir(PATH + "\\" + i):
                    if "label" in i.lower() or "mask" in i.lower():
                        folder_name.append(i)
                        if os.path.exists(SAVE_PATH + "\\" + i):
                            shutil.rmtree(SAVE_PATH + "\\" + i)
                        shutil.copytree(PATH + "\\" + i, SAVE_PATH + "\\" + i)
                else:
                    folder_name.append(i)
                    if os.path.exists(SAVE_PATH+"\\"+i):
                        os.remove(SAVE_PATH+"\\"+i)
                    shutil.copy(PATH + "\\" + i, SAVE_PATH)

            img_name = PATH + "\\" + folder_name[0]
            label_name = os.listdir(PATH + "\\" + folder_name[1])

            f = open(SAVE_PATH + "\\imgclass.txt", "w+")

            SAVE_PATH = SAVE_PATH + "\Annosvis"
            if not os.path.isdir(SAVE_PATH):
                os.mkdir(SAVE_PATH)

            #img_save_name = SAVE_PATH + "\\" + folder_name[0]

            anno_count = 0
            max_class = []
            unique_class = set()

            for i in range(len(label_name)-1):
                #if i == 0:
                #    IMG_PATH = img_name
                #else:
                #    IMG_PATH = img_save_name
                IMG_PATH = img_name
                TXT_PATH = ''
                MASK_PATH = ''
                if "label" in folder_name[1].lower():
                    checktxt = label_name[i].split(".")
                    if checktxt[len(checktxt)-1] != "txt":
                        anno_count += 1
                    TXT_PATH = PATH + "\\" + folder_name[1] + "\\" + label_name[i+anno_count]
                else:
                    checkmask = label_name[i].split(".")
                    if checkmask[len(checkmask)-1] == "txt":
                        anno_count += 1
                    MASK_PATH = PATH + "\\" + folder_name[1] + "\\" + label_name[i+anno_count]

                anno = Annotation(image_path = IMG_PATH,
                                bbox_path = TXT_PATH,
                                mask_path = MASK_PATH,
                                save_path = SAVE_PATH,
                                )
                out_imgs , uc = AnnotationVisualizer.draw_annotations(anno, mask_box_split = False) #['original_image', 'annotated_image']
                if str(uc) not in unique_class and uc:
                    unique_class.add(str(uc))
                    for cn in uc:
                        if cn not in max_class:
                            max_class.append(cn)
                    print(str(uc), file=f)

            print("total_class:" + str(len(max_class)), file=f)
            print("path" + ":"+ PATH, file=f)
            f.close()


if __name__ == "__main__":
    da = DrawAnnos()
    da.draw_annos_with_ditprj(r'D:\data\translate_all\4F_Presence_Append',r'D:\data\translate_output')
