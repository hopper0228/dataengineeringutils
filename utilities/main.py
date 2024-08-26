import argparse
import os

from termcolor import colored

from src.convert_check import PreCheckTool
from src.folder_to_csv import FolderTOCsv
from src.mask_to_bbox import MaskToBbox
from src.preprocess_tool import DrawAnnos, PreProcessTool


def usage(error_type: int = 0) -> None:
# print usage message if user's input is incorrect
    output_message = ("unexpected problem, please contact the author", "dataset path does not exist", "saving path does not exist")
    print(colored("WRONG", "red"), "input arguments\nusage: main.py [-h] [-p PATH] [-sp SAVE_PATH] [-sn SAVENAME] [-fv FORMAT_VALID]")
    print("ERROR message: ", end = "")
    print(output_message[error_type])

def check_arguments_valid(args: argparse.Namespace) -> None:
    error_type = 1
    for arg_name, arg_value in vars(args).items():
        if(error_type < 3 and os.path.exists(arg_value) == False):
            usage(error_type)
            exit()
        error_type += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default=r"D:\data\new\customer\CviLux", help="your path")
    parser.add_argument("-sp", "--save_path", type=str, default=r"D:\data\new_test", help="your save path")
    parser.add_argument("-sn","--savename", type=str, default="bbox", help="your save name")
    parser.add_argument("-fv","--format_valid", type=bool, default=True, help="Wether the format is valid.")

    args = parser.parse_args()
    check_arguments_valid(args)

    DrawAnnos().draw_by_folder(args.path, args.save_path)
    save_path = PreProcessTool().get_save_path(args.path, args.save_path)
    MaskToBbox().mask_to_bbox_by_folder(save_path)
    FolderTOCsv().folder_to_csv(save_path)
    PreCheckTool().checking_all(args.path, args.save_path, args.format_valid)
    print('end')
