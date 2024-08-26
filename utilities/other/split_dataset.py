import os
import shutil

from sklearn.model_selection import train_test_split


class SplitDataset:
    def __init__(self) -> None:
        self.save_path = ""

    #CHV_dataset
    def split_CHV_dataset(self, root_folder, image_name, label_name, txt_path):
        father_path = os.path.dirname(root_folder)

        for file_name in os.listdir(txt_path):

            folder_name = os.path.join(root_folder, file_name.split('.')[0])
            os.makedirs(folder_name, exist_ok=True)
            image_folder = os.path.join(folder_name, image_name)
            os.makedirs(image_folder, exist_ok=True)
            label_folder = os.path.join(folder_name, "labels")
            os.makedirs(label_folder, exist_ok=True)

            # 读取图像名称列表
            with open(os.path.join(txt_path,file_name), 'r') as file:
                image_names = [line.strip() for line in file]

            for img_name in image_names:
                shutil.copy(os.path.join(father_path,img_name), image_folder)
                shutil.copy(os.path.join(father_path,img_name.replace(img_name.split('.')[-1],"txt").replace(image_name,label_name)), label_folder)


    #CVC-ClinicDB 結構特殊
    def add_label(self,root_folder):
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


    #AITEX-Fabric 資料夾結構較特殊，root_folder要用images跟NODefect各做一次
    def split_AITEX_Fabric(self, root_folder, image_name, other_image_name, label_name):
        from draw_annos import DrawAnnos
        da = DrawAnnos()
        self.split_data_with_label(root_folder, image_name, label_name)
        self.split_data_with_label(root_folder, other_image_name, label_name)
        da.create_color_mask_from_folder(root_folder)

    def split_data_with_label(self, root_folder, image_name, label_name):

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
        # 創建訓練集和測試集的資料夾
        train_folder = os.path.join(train_root, "images")
        test_folder = os.path.join(test_root, "images")
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)
        train_lb_folder = os.path.join(train_root, "masks")
        test_lb_folder = os.path.join(test_root, "masks")
        os.makedirs(train_lb_folder, exist_ok=True)
        os.makedirs(test_lb_folder, exist_ok=True)

        min_samples_per_class = 2

        # 對於每個類別，將樣本分成訓練集和測試集，確保每個集合中都有至少兩個樣本
        for label, count in class_counts.items():
            # 檢查類別是否有足夠的樣本數量
            if count < min_samples_per_class:
                continue

            label_images = [image for image, lbl in dataset.items() if lbl.split(os.sep)[-1].split('_')[1] == label]
            # 將資料集切分成訓練集和測試集，確保每個集合中都包含所有的類別
            train_images, test_images = train_test_split(label_images, test_size=0.2, random_state=42)

            # 將訓練集複製到相應的資料夾
            for image_path in train_images:
                shutil.copy(image_path, train_folder)
                shutil.copy(dataset[image_path], train_lb_folder)

            # 將測試集複製到相應的資料夾
            for image_path in test_images:
                shutil.copy(image_path, test_folder)
                shutil.copy(dataset[image_path], test_lb_folder)

        shutil.rmtree(image_folder)

    def combine_images(self, root_folder):
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


    #目前沒用
    def split_data_without_label(self, root_folder, image_name, label_name):

        image_folder = os.path.join(root_folder, image_name)
        label_folder = os.path.join(root_folder, label_name)

        dataset = {}

        for image_file in sorted(os.listdir(image_folder)):
            for label_file in sorted(os.listdir(label_folder)):
                if image_file.split('.')[0] in label_file:
                    dataset[os.path.join(image_folder, image_file)] = os.path.join(label_folder, label_file)

        # 創建訓練集和測試集的資料夾
        train_root = os.path.join(root_folder, "train")
        test_root = os.path.join(root_folder, "test")
        os.makedirs(train_root, exist_ok=True)
        os.makedirs(test_root, exist_ok=True)
        train_folder = os.path.join(train_root, "images")
        test_folder = os.path.join(test_root, "images")
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)
        train_lb_folder = os.path.join(train_root, "labels")
        test_lb_folder = os.path.join(test_root, "labels")
        os.makedirs(train_lb_folder, exist_ok=True)
        os.makedirs(test_lb_folder, exist_ok=True)

        # 將資料集切分成訓練集和測試集，確保每個集合中都包含所有的類別
        train_images, test_images = train_test_split(dataset.keys(), test_size=0.2, random_state=42)

        # 將訓練集複製到相應的資料夾
        for image_path in train_images:
            shutil.copy(image_path, train_folder)
            shutil.copy(dataset[image_path], train_lb_folder)

        # 將測試集複製到相應的資料夾
        for image_path in test_images:
            shutil.copy(image_path, test_folder)
            shutil.copy(dataset[image_path], test_lb_folder)


if __name__ == "__main__":
    import argparse

    #要有image跟label
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default=r'C:\Users\hopper_chu\Documents\Python_Scripts\DAGM\assdatahub\segement_detection\AITEX-Fabric', help="your path")
    parser.add_argument("-m","--image_name", type=str, default="Defect_images", help="your image folder name")
    parser.add_argument("-om","--other_image_name", type=str, default="NODefect_images", help="your image folder name")
    parser.add_argument("-l","--label_name", type=str, default= "Mask_images", help="your label folder name")
    parser.add_argument("-tp","--txt_path", type=str, default=r"C:\Users\hopper_chu\Documents\Python_Scripts\DAGM\assdatahub\object_detection\CHV_dataset\CHV_dataset\data split")
    args = parser.parse_args()

    sd = SplitDataset()
    #sd.add_label(args.path)
    #sd.split_CHV_dataset(args.path, args.image_name, args.label_name, args.txt_path)
    sd.split_AITEX_Fabric(args.path, args.image_name, args.other_image_name, args.label_name)
