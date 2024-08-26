import os
from io import BytesIO

import cv2
import numpy as np


#### I/O Utils ####
def check_file_exist(filepath: str) -> bool:
    ret = True
    if filepath is None or not os.path.exists(filepath):
        ret = False
    return ret

def parse_file_extension(filepath: str) -> str:
    _, file_extension = os.path.splitext(filepath)
    return file_extension

def load_image(filepath: str, grayscale: bool = False) -> np.ndarray:
    if not check_file_exist(filepath):
        raise FileNotFoundError(filepath)

    if not check_supported_image_format(filepath):
        raise ValueError(f'Not supported to load {filepath}')

    if grayscale:
        image = cv2.cvtColor(
            cv2.imread(filepath, flags=cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR
        )
    else:
        image = cv2.imread(filepath, flags=cv2.IMREAD_COLOR)

    #verify_image(image)
    return image

def load_mask(filepath: str) -> np.ndarray:
    if not check_file_exist(filepath):
        raise FileNotFoundError(filepath)

    if not check_supported_image_format(filepath):
        raise ValueError(f'Not supported to load {filepath}')

    with open(file=filepath, mode='rb') as fin:
        data = fin.read()

    rgb_image = cv2.imread(filepath, flags=cv2.IMREAD_COLOR)
    image = cv2.imdecode(np.frombuffer(BytesIO(data).read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    verify_image(image)
    return image, rgb_image

def verify_image(image: np.ndarray):
    if image is None:
        raise ValueError('Corrupted image')

    if image.dtype != np.uint8 or \
            (len(image.shape) == 3 and image.shape[-1] not in [1, 3]) or \
            len(image.shape) not in [2, 3]:
        raise ValueError(
            'Not correct image format. An image should be a np.ndarray with dtype np.uint8, '
            'and its shape should be either [N, N, 3], [N, N, 1] or [N, N].'
            f'Given image dtype {image.dtype} and shape {image.shape}'
        )

def check_supported_image_format(filepath: str) -> bool:
    SUPPORTED_IMAGE_FILE_FORMAT = ['.bmp', '.jpg', '.jpeg', '.png', '.tmp']
    format_ = parse_file_extension(filepath)
    return format_.lower() in SUPPORTED_IMAGE_FILE_FORMAT

def load_npy(filepath: str) -> np.ndarray:
    if not check_file_exist(filepath):
        raise FileNotFoundError(filepath)
    if parse_file_extension(filepath) != '.npy':
        raise FileNotFoundError(f'Not supported to load {filepath}')

    with open(file=filepath, mode='rb') as fin:
        data = fin.read()

    nparray = np.load(BytesIO(data))
    return nparray

def read_file(filepath: str) -> BytesIO:
    if not check_file_exist(filepath):
        raise FileNotFoundError(filepath)

    io = BytesIO()
    with open(filepath, 'rb') as fin:
        io = BytesIO(fin.read())

    buffer = io.getvalue()

    return BytesIO(buffer)
