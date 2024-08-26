import os
import shutil

from draw_annos import DrawAnnos


class DrawByFolder:
    def __init__(self) -> None:
        self.draw_path = ""
        self.SAVE_PATH = ""

    def draw_by_folder(self,folder_structure):
        Need_Img = 0
        da = DrawAnnos()
        folder_path = self.draw_path[-1].split(os.sep)
        self.draw_path.append(self.draw_path[-1] + os.sep + folder_structure.name)
        project_name = folder_structure.name + ".ditprj"
        if os.path.isdir(self.draw_path[-1]):
            if folder_structure.checkmatch == "match":
                print(self.draw_path[-1])
                if project_name in os.listdir(self.draw_path[-1]):
                    #畫project的
                    da.draw_annos_with_ditprj(self.draw_path[-1], self.SAVE_PATH)
                    Need_Img = 1
                else:
                    #畫沒project的
                     da.draw_annos_without_ditprj(self.draw_path[-1], self.SAVE_PATH)
            elif folder_structure.checkmatch == "tiny":
                da.draw_tiny(self.draw_path[-1], self.SAVE_PATH)

        for i in folder_structure.repo_dict:
                if "原始圖片" == i and Need_Img == 1:
                    Need_Img = 0
                    img_path = self.SAVE_PATH
                    for j in range(da.path_del_num, len(folder_path)):
                        img_path = img_path + os.sep + str(folder_path[j])
                        if not os.path.isdir(img_path):
                            os.mkdir(img_path)
                    if os.path.exists(img_path + os.sep + i):
                        shutil.rmtree(img_path + os.sep + i)
                    shutil.copytree(self.draw_path[len(self.draw_path)-1], img_path + os.sep + i)
                elif hasattr(folder_structure.repo_dict[i],"repo_dict"):
                    self.draw_by_folder(folder_structure.repo_dict[i])

        self.draw_path.pop()

    def draw_main(self, root_path, save_path):
        from project_check import CheckProject
        self.draw_path = [os.sep.join(root_path.split(os.sep)[0:-1])]
        self.SAVE_PATH = save_path
        if not os.path.isdir(save_path) and save_path:
            os.mkdir(save_path)

        if os.path.exists(root_path) and os.path.isdir(root_path):
            f = open(save_path + os.sep + root_path.split(os.sep)[-1] + ".txt","w+")
            cp = CheckProject(['.jpg','.bmp','.png','.jpeg'],f)
            folder_structure = cp.folder_to_dict(root_path)
            f.close()
        else:
            print("指定的文件夹路径不存在或不是文件夹。")

        self.draw_by_folder(folder_structure)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--path", type=str, default=r"D:\data\origin_dataset\customer\Samsung screws", help="your path")
    parser.add_argument("-sp","--savepath", type=str, default=r"D:\data\translate_output", help="your save path")

    args = parser.parse_args()

    df = DrawByFolder()
    df.draw_main(args.path, args.savepath)
