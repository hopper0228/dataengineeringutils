import os

import numpy as np

from src.mask_to_bbox import MaskToBbox

"""
I used test_mask_to_bbox.sh to build a recursive
structured directory to test mask_to_bbox_by_folder
since its solution is to find two folder with the
same amount of elements
"""

class TestMaskToBbox:
    def test_mask_to_bbox_by_folder(self):
        # create a recursive structured folder using shell script
        os.system("bash test_mask_to_bbox.sh /mnt/c/Users/edmond_huang/Desktop/Unit_Test 100")

        folderpath1 = r"C:\Users\edmond_huang\Desktop\Unit_Test\build_here"
        savepath1 = r"C:\Users\edmond_huang\Desktop\Unit_Test"
        savename1 = "test_mask_to_bbox_by_folder_result"
        folderpath2 = r"C:\Users\edmond_huang\Desktop\temp\segment"
        savepath2 = r"C:\Users\edmond_huang\Desktop\Unit_Test"

        check1 = MaskToBbox().mask_to_bbox_by_folder(folderpath1, savepath1, savename1) # return false
        check2 = MaskToBbox().mask_to_bbox_by_folder(folderpath2, savepath2, "")

        assert check1 == False
        assert check2 == True
