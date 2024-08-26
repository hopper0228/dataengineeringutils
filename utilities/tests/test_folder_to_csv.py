import os

from src.folder_to_csv import FolderTOCsv


class TestFolderToCsv:
    def test_folder_to_csv_bbox(self):

        root_folder = r"C:\Users\edmond_huang\Desktop\original_dataset\object\ioport\OriginalImage"
        image_folder_name = "JPEGImages"
        label_folder_name = "Labels"

        (img_root_path, csv_name, csv_path, images,
         label_root_path, label_folder_path) = FolderTOCsv().get_csv_file_info(root_folder, image_folder_name, label_folder_name, "", False)

        FolderTOCsv().folder_to_csv_bbox(root_folder, image_folder_name, label_folder_name)

        assert os.path.exists(csv_path) == True
        assert os.path.basename(csv_path) == csv_name

    def test_folder_csv_mask(self):
        root_folder = r"C:\Users\edmond_huang\Desktop\original_dataset\segment\CviLux\CviLux_512_crop_modify"
        image_folder_name = "Image"
        label_folder_name = "Label"
        mask_folder_name = "Annosvis"

        (img_root_path, csv_name, csv_path,
        images_mask, mask_root_path) = FolderTOCsv().get_csv_file_info(root_folder, image_folder_name, label_folder_name, mask_folder_name, True)

        FolderTOCsv().folder_csv_mask(root_folder, image_folder_name, label_folder_name, mask_folder_name)

        assert os.path.exists(csv_path) == True
        assert os.path.basename(csv_path) == csv_name
