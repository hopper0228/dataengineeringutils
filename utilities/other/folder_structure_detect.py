import copy
import os

from PIL import Image


class File:

    def __init__(self) -> None:
        self.type = ""
        self.num = 0
        self.file_num_dict = {}
        self.resolution = []
        self.folder_name = ""

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

    def print_structure(self, indent=0):
        for item, value in self.repo_dict.items():
            if isinstance(value, Folder):
                print("  " * indent + f"目錄: {item}")
                value.print_structure(indent + 2)
            else:
                print("  " * indent + f"檔案: {item}")

class FolderStruct:

    def __init__(self) -> None:
        self.onefile = File()
        self.temp = []
        self.SUPPORTED_FILE_KEYS = ['.jpg','.bmp','.png','.jpeg','.JPG','.BMP','.PNG','.JPEG']
        self.f = None

    def file_to_dict(self,folder_name, parent_name, indent):
        folder_dict = Folder()
        folder_dict.folder_name = folder_name
        folder_dict.parent_name = parent_name
        unique = []
        for num in self.onefile.resolution:
            if str(num) not in unique:
                unique.append(str(num))
        filetype = list(self.onefile.file_num_dict.keys())
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

        self.onefile.clear()
        return folder_dict

    def folder_to_dict(self,folder_path, save_path=None, save_name=None, SUPPORTED_FILE_KEYS=[], indent=0):
        if save_path != None and self.f == None:
            sp = save_path + os.sep + save_name
            self.f = open(sp + ".txt","w+")
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
                    if save_path == None:
                        pass
                        #print("  " * indent + "目錄: " + item)
                    if self.f:
                        print("  " * indent + f"目錄: {item}", file = self.f)
                    folder_dict.folder_name = folder_path.split(os.sep)[-1]
                    folder_dict.parent_name = folder_path.split(os.sep)[-2]
                    folder_dict.repo_dict[item] = self.folder_to_dict(item_path, save_path, save_name, self.SUPPORTED_FILE_KEYS , indent+2)
                    if self.onefile.num!=0 and len(folder_dict.repo_dict[item].repo_dict) == 0:
                        folder_dict.repo_dict[item] = self.file_to_dict(item, folder_dict.folder_name, indent+2)
                else:
                    # 如果是文件，直接添加到字典
                    root, extension = os.path.splitext(item_path)
                    if extension in self.onefile.file_num_dict:
                        self.onefile.file_num_dict[extension] += 1
                    else:
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
            folder_dict.file_dict = self.file_to_dict(folder_path.split(os.sep)[-1],folder_path.split(os.sep)[-2], indent)
            self.temp.pop()

        return folder_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--path", type=str, default=r"D:\data\20231122_FAEData\FromFAE", help="your path")
    parser.add_argument("-sp","--savepath", type=str, default=r"D:\data\result", help="your save path")
    parser.add_argument("-sn","--savename", type=str, default="", help="your save path")
    parser.add_argument("-sfk","--supported_file_keys", type=list, default= ['.jpg','.bmp','.png','.jpeg','.JPG','.BMP','.PNG','.JPEG'], help="your supported file keys")
    args = parser.parse_args()

    fs = FolderStruct()
    if os.path.exists(args.path) and os.path.isdir(args.path):
        save_name = f"{str(len(os.listdir(args.savepath)) + 1)}-{args.path.split(os.sep)[-1]}-folderstruct"
        folder_structure = fs.folder_to_dict(args.path, args.savepath, save_name, args.supported_file_keys)
        fs.f.close()
        print(folder_structure.print_structure())
    else:
        print("指定的文件夹路径不存在或不是文件夹。")
