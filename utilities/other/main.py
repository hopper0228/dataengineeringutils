import os

#### Transfer project files and annotations
from utils.ditproj_datasets import AISDataConverter, DITProject, ImageSample

PATH = r"D:\data\three\customer\Teaegg\AISVision_project\TeaEgg_Seg"

OUT_PATH = PATH + "\Annosvis"
if not os.path.isdir(OUT_PATH):
    os.mkdir(OUT_PATH)
project_name = PATH.split("\\")
CONFIG = PATH + "\\" + project_name[len(project_name)-1] + ".ditprj"
proj = DITProject(PATH, CONFIG)
meta_data = proj.get_meta_data()

annos = AISDataConverter.convert_dataset(meta_data, dataset_root = PATH, output_path = OUT_PATH, is_encrypted = True)

#### Visualization Region
from utils.annotation_utils.alg_annotation_utils import Annotation, AnnotationVisualizer

SAVE_PATH = "D:\\data\\three\maskorbbox\\" + str(project_name[4] + "-" + project_name[len(project_name)-1])
if not os.path.isdir(SAVE_PATH):
    os.mkdir(SAVE_PATH)

f = open(SAVE_PATH + "\\imgclasses.txt", "w+")
classnumber = 0
unique_class = {}

for i in range(len(meta_data)):
    IMG_PATH = PATH + "\\Image\\" + meta_data[i][1]
    TXT_PATH = ''
    MASK_PATH = ''
    if meta_data[0][4]:
        TXT_PATH = PATH + "\\Annosvis\\" + str(meta_data[i][1]).split(".")[0] + ".txt"
    else:
        MASK_PATH = PATH + "\\Annosvis\\" + str(meta_data[i][1]).split(".")[0] + ".png"

    anno = Annotation(image_path = IMG_PATH,
                    bbox_path = TXT_PATH,
                    mask_path = MASK_PATH,
                    save_path = SAVE_PATH,
                    )
    if str(annos[i].class_id) not in unique_class and annos[i].class_id:
        out_imgs = AnnotationVisualizer.draw_annotations(anno, mask_box_split = False) #['original_image', 'annotated_image']
        print(meta_data[i][1] + ":" + str(annos[i].class_id), file=f)
        unique_class[str(annos[i].class_id)] = meta_data[i][1]
        classid = max(annos[i].class_id)
        classnumber = max(classnumber , classid)

print("total_class:" + str(classnumber), file=f)
print("path" + ":"+ PATH, file=f)
f.close()
