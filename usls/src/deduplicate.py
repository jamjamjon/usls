import hashlib
import os
from pathlib import Path
import time
import shutil
import cv2
from PIL import Image
from tqdm import tqdm
import rich


from.utils import IMG_FORMAT


def get_md5(img_path):
    m = hashlib.md5(open(img_path,'rb').read())
    return m.hexdigest()


def deduplicate(input_dir, move_dir= "deduplicate-dir", info=False):
    image_list = [x for x in Path(input_dir).iterdir() if x.suffix.lower() in IMG_FORMAT]
    rich.print(f"> Now only support: {IMG_FORMAT}\n> Has {len(image_list)} images before deduplicating!")

    img_md5_dict = {} # {md5: abs_img_path}

    if not Path(move_dir).exists():
        Path(move_dir).mkdir(exist_ok=True, parents=True)

    # t1 = time.time()
    for img in tqdm(image_list, 'De-duplicating...'):

        if cv2.imread(str(img.resolve())) is not None:
            md5 = get_md5(str(img.resolve()))

            if md5 in img_md5_dict.keys():
                similar_img_path = img_md5_dict[md5]
                shutil.move(str(img.resolve()), str(Path(move_dir).resolve()))
                if info:
                    rich.print(f"Move [{img} to {Path(move_dir)}] | similar image is [{similar_img_path}]")
            else:
                img_md5_dict[md5] = str(Path(img))

        else:
            shutil.move(str(img.resolve()), str(Path(move_dir).resolve()))
            if info:
                rich.print(f"Wrong image! Move [{img} to {Path(move_dir)}]")



    rich.print(f"> Now has {len([x for x in Path(input_dir).iterdir() if x.suffix in ('.png', '.jpg', '.jpeg')])} images.")



if __name__=='__main__':
    deduplicate()
