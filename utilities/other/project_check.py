import os

from PIL import Image


class File:

    def __init__(self) -> None:
        self.type = ""
        self.num = 0
        self.file_num_dict = {}
        self.resolution = []
        self.checkmatch = ""
        self.file_num = {}
        self.match_path = ""

    def clear(self):
        self.type = ""
        self.num = 0
        self.file_num_dict = {}
        self.resolution = []

class Folder:
    def __init__(self):
        self.name = ""
        self.img_dict = {}          #圖檔的資訊=FILE CLASS
        self.file_dict = {}         #文檔的數量=txt
        self.repo_dict = {}         #此folder包含的folder ex:benjamin.repo_dict = {stylegan2-nv,stylegan2-pytorch}
        self._unknown_dict = {}     #其他
        self.checkmatch = ""

    def print_structure(self, indent=0):
        for item, value in self.repo_dict.items():
            if isinstance(value, Folder):
                print("  " * indent + f"目錄: {item}")
                value.print_structure(indent + 2)
            else:
                print("  " * indent + f"檔案: {item}")


class CheckProject:

    def __init__(self,Keys,file) -> None:
        self.onefile = File()
        if Keys:
            self.SUPPORTED_FILE_KEYS = Keys
        else:
            self.SUPPORTED_FILE_KEYS = ['.jpg','.bmp','.png','.jpeg']
        if file:
            self.f = file
        else:
            self.f = None
        self.print_path = []
        self.have_print = []
        self.check_match_path = ''
        self.checkmatch = ""
        self.match_path = ""

    def file_to_dict(self,item,indent):
        folder_dict = Folder()
        unique = []
        for num in self.onefile.resolution:
            if str(num) not in unique:
                unique.append(str(num))
        filetype = list(self.onefile.file_num_dict.keys())
        for ftype in filetype:
            if ftype in self.SUPPORTED_FILE_KEYS or ftype.lower() in '.txt' or ftype.lower() in '.tmp':
                file_type = ftype
        filecount = list(self.onefile.file_num_dict.values())
        for c in range(len(filetype)):
            if str(filetype[c]) in self.SUPPORTED_FILE_KEYS:
                folder_dict.img_dict[str(filetype[c]),"長寬"] = str(filecount[c]),unique
            else:
                folder_dict.file_dict[str(filetype[c])] = str(filecount[c])

        if "image" in item.lower():
            fc = 0
            for i in range(len(filecount)):
                fc = fc + filecount[i]
            self.onefile.file_num[fc] = item
            self.onefile.checkmatch = item
        elif "label" in item.lower() and len(self.onefile.file_num) > 0 or "mask" in item.lower() and len(self.onefile.file_num) > 0:
            self.match_path = self.onefile.file_num[self.onefile.file_num_dict[file_type]]
            if self.onefile.file_num_dict[file_type] in self.onefile.file_num:
                self.onefile.file_num.pop(self.onefile.file_num_dict[file_type])
                self.onefile.checkmatch = "match"
            else:
                self.onefile.checkmatch = "no match"
        #else:
        #    self.onefile.checkmatch = "no use"

        self.onefile.clear()
        return folder_dict

    def folder_start(self,folder_path):
        print(folder_path, file=self.f)
        folder_dict = self.folder_to_dict(folder_path)
        return folder_dict

    def folder_to_dict(self,folder_path, indent=2):
        folder_dict = Folder()
        folder_dict.name = folder_path.split(os.sep)[-1]
        contents = os.listdir(folder_path)
        try:
            for item in contents:
                item_path = os.path.join(folder_path, item)
                #如果是資料夾
                if os.path.isdir(item_path):
                    if self.onefile.num > 0:
                        for file in self.onefile.file_num_dict.keys():
                            folder_dict.repo_dict[file] = self.onefile.file_num_dict[file]
                        self.onefile.clear()

                    if self.f:
                        self.print_path.append("  " * indent + f"目錄: {item}")

                    if 'image' in item.lower():
                        self.check_match_path = folder_path
                    folder_dict.repo_dict[item] = self.folder_to_dict(item_path)
                    #如果是最後一層資料夾
                    if self.onefile.num!=0 and len(folder_dict.repo_dict[item].repo_dict) == 0:
                        folder_dict.repo_dict[item] = self.file_to_dict(str(item_path),indent+2)
                        if "match" in self.onefile.checkmatch or "tiny" in self.onefile.checkmatch:
                            if self.f:
                                # for path_name in self.print_path:
                                #     if path_name not in self.have_print:
                                #         print(path_name, file = self.f)
                                #         self.have_print.append(path_name)
                                # self.print_path.pop()
                                lb_path = item_path.split(os.sep)
                                img_path = self.match_path.split(os.sep)
                                print("  " * indent + f"目錄: {self.check_match_path.split(os.sep)[-1]}", file = self.f)
                                temp_indent = indent
                                for p in img_path:
                                    if p not in self.check_match_path:
                                        temp_indent += 2
                                        print("  " * temp_indent + f"目錄: {p}", file = self.f)
                                temp_indent = indent
                                for p in lb_path:
                                    if p not in self.check_match_path:
                                        temp_indent += 2
                                        print("  " * temp_indent + f"目錄: {p}", file = self.f)
                            folder_dict.checkmatch = self.match_path
                            self.checkmatch = str(self.onefile.checkmatch)
                            self.onefile.checkmatch = ""

                    if "no" in  self.onefile.checkmatch or self.onefile.checkmatch == "":
                        self.print_path.pop()

                    if "image" not in item_path.lower() and "label" not in item_path.lower() and self.onefile.checkmatch != '':
                        self.print_path.pop()

                    if self.checkmatch and folder_path == self.check_match_path:
                        folder_dict.checkmatch = "match"
                        self.checkmatch = ""

                    if self.onefile.num > 0:
                        for file in self.onefile.file_num_dict.keys():
                            folder_dict.repo_dict[item].repo_dict[file] = self.onefile.file_num_dict[file]
                        self.onefile.clear()
                else:
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

        return folder_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--path", type=str, default=r"D:\data\origin_dataset\4F_Presence_Append", help="your path")
    parser.add_argument("-sp","--savepath", type=str, default=r"D:\data\result", help="your save path")
    parser.add_argument("-sn","--savename", type=str, default="", help="your save name")
    parser.add_argument("-sfk","--supported_file_keys", type=list, default= ['.jpg','.bmp','.png','.jpeg'], help="your supported file keys")
    args = parser.parse_args()

    sp = args.savepath + os.sep + str(len(os.listdir(args.savepath))+1) + "-" + args.path.split(os.sep)[-1] + "-projectcheck"
    f = open(sp + args.savename + ".txt","w+")
    print(args.path,file=f)
    pc = CheckProject(args.supported_file_keys,f)
    if os.path.exists(args.path) and os.path.isdir(args.path):
        folder_structure = pc.folder_to_dict(args.path)
        print(folder_structure.print_structure())
    else:
        print("指定的文件夹路径不存在或不是文件夹。")
    f.close()
