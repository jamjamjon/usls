import cv2
from pathlib import Path
import numpy as np
import rich
import shutil
import argparse
import os
from tqdm import tqdm
import sys
import random
import time
import logging
import glob
import re
import hydra
import rich
from rich.console import Console
import psutil
from datetime import datetime
import pynvml
import contextlib
import numpy as np
from dataclasses import dataclass
from typing import Union


IMG_FORMAT = ('.jpg', '.jpeg', '.png', '.bmp')
VIDEO_FORMAT = ['.mp4', '.flv', '.avi', '.MOV']
CONSOLE = Console()

def setup_logging_plain(
        stream_logger_name=None, 
        stream_level=logging.DEBUG,
    ):

    # stream logger 
    stream_logger = logging.getLogger(stream_logger_name)
    stream_logger.setLevel(stream_level)

    stream_handler = logging.StreamHandler() 
    stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    stream_handler.setLevel(stream_level)
    stream_logger.addHandler(stream_handler)  
    return stream_logger

LOGGER = setup_logging_plain(__name__)


# increase python path
def add_python_path(verbose=False):
	for child in list(Path(__file__).resolve().parents):
		sys.path.append(str(child))
	if verbose:	
		rich.print(sys.path)



class INSPECTOR(contextlib.ContextDecorator):

    def __init__(self, prefix='Time', cpu=False, mem=False, gpu=False):
        self.prefix = prefix
        self.cpu, self.mem, self.gpu = cpu, mem, gpu


    def __enter__(self):
        self.t0 = time.time()

        if self.cpu:
            self.cpu_usage_avg0 = psutil.cpu_percent(percpu=False, interval=None)

        if self.mem:
            self.mem0 = psutil.virtual_memory().used

        if self.gpu:
            pass

        return self


    def __exit__(self, type, value, traceback):
        self.duration = time.time() - self.t0  
        CONSOLE.log(f"{self.prefix} | Time consume: {(time.time() - self.t0) * 1e3:.2f} ms.")

        if self.cpu:
            CONSOLE.log(f"{self.prefix} | CPU cost: {(psutil.cpu_percent(percpu=False, interval=None) - self.cpu_usage_avg0)} %.")
            # CONSOLE.log(f"{self.prefix} | start: {self.cpu_usage_avg0} | end: {(psutil.cpu_percent(percpu=False, interval=None))} | CPU cost: {(psutil.cpu_percent(percpu=False, interval=None) - self.cpu_usage_avg0)} %.")

        if self.mem:
            CONSOLE.log(f"{self.prefix} | Memory cost: {(psutil.virtual_memory().used - self.mem0) / GB:.3f} %.")
            # CONSOLE.log(f"{self.prefix} | start: {self.mem0 / GB} | end: {psutil.virtual_memory().used / GB} | Memory cost: {(psutil.virtual_memory().used - self.mem0) / GB:.3f} %.")

        if self.gpu:
            pass



    def __call__(self, func):
        def wrapper(*args, **kwargs):
            t0 = time.time()
            ret = func(*args, **kwargs)
            CONSOLE.log(f"{self.prefix} consume: {(time.time() - t0) * 1e3:.2f} ms.")
            return ret
        return wrapper




def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


# TODO
def verify_images(path):

    try:
        # PIL read img
        im = Image.open(path)
        
        # PIL image quality check
        im.verify()

        # size < 10x10 will be  an error
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'

        # jpg & jpeg corrupt check
        if im.format.lower() in ('jpeg', 'jpg'):
            with open(path, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(path)).save(path, 'JPEG', subsampling=0, quality=100)
                    LOGGER.warning(f"WARNING: {path}: corrupt JPEG restored and saved")

    except Exception as e:
        LOGGER.warning(f"Exceptions: {e}")


def verify_label(path):
    pass



# img_list & label_list, relative path
def load_img_label_list(img_dir, label_dir, img_format, info=True):
    image_list = [x for x in Path(img_dir).iterdir() if x.suffix in img_format]
    label_list = list(Path(label_dir).glob("*.txt"))
    
    if info:
        rich.print(f"[green]> Images count: {len(image_list)}")
        rich.print(f"[green]> Labels count: {len(label_list)}")
        

    return image_list, label_list


# img_path => label_path(txt)
def get_corresponding_label_path(img_path, output_dir):
    label_name = Path(img_path).stem + '.txt'
    saveout = Path(output_dir) / label_name 
    return str(saveout)


# √ Check if a point belongs to a rectangle
def is_point_in_rect(x, y, l, t, r, b):
    return l <= x <= r and t <= y <= b


# colors palette
class Colors:
    '''
        # hex 颜色对照表    https://www.cnblogs.com/summary-2017/p/7504126.html
        # RGB的数值 = 16 * HEX的第一位 + HEX的第二位
        # RGB: 92, 184, 232 
        # 92 / 16 = 5余12 -> 5C
        # 184 / 16 = 11余8 -> B8
        # 232 / 16 = 14余8 -> E8
        # HEX = 5CB8E8
    '''

    # def __init__(self, random=0, shuffle=False):
    def __init__(self, shuffle=False):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('33FF00', '9933FF', 'CC0000', 'FFCC00', '99FFFF', '3300FF', 'FF3333', # new add
               'FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', 
               '1A9334', '00D4BB', '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', 
               '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        
        # shuffle color 
        if shuffle:
            hex_list = list(hex)
            random.shuffle(hex_list)
            hex = tuple(hex_list)

        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)
        # self.b = random   # also for shuffle color 


    def __call__(self, i, bgr=False):        
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod  
    def hex2rgb(h):  # int('CC', base=16) 将16进制的CC转成10进制 
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))






# -------------------------------------
# mode: check img_label_pair 
# 1. 检查是否所有的img都有对应lable；  done
# 2. 是否所有的label都有对应img; 
# 3. 并且检查是否所有的label都是有内容的
# 将不满足的img和label移除到 mv_dir
# 4. img dir 仅仅可以存在支持的IMG_FORMAT文件
# 5. label dir 仅仅可以存在.txt文件
# -------------------------------------
def img_label_dir_cleanup(
        input_img_dir, 
        input_label_dir, 
        mv_dir, 
        img_format, 
        info=True, 
        dont_clean_empty_txt=False,
        verbose=False,
    ):

    # load img and label
    image_list, label_list = load_img_label_list(input_img_dir, input_label_dir, img_format, info)

    # create mv-dir if not exist
    if not Path(mv_dir).exists():
        Path(mv_dir).mkdir()


    # 1. 检查是否所有的image都有对应的label
    # has img, no label => remove img
    rich.print(f"> Checking if all images has corresponding label...")
    for image_path in tqdm(image_list):

        # has corresponding label, continue
        if Path(input_label_dir) / (image_path.stem + '.txt') in label_list:
            continue

        # else remove img
        # rich.print(f"[bold red]No corresponding label: {image_path}, moved.")
        if verbose:
            LOGGER.warning(f"No corresponding label: {image_path}, moved.")
        shutil.move(str(image_path), mv_dir)  


    # 2. 剩余的所有image都有对应的label，size(img) <= size(label)
    # 检查是否所有的label都有对应的image
    # has label, no img  => remove label 
    rich.print(f"> Checking if all labels has corresponding image...")
    
    image_list, label_list = load_img_label_list(input_img_dir, input_label_dir, img_format, info)
    for label_path in tqdm(label_list):      

        # remove label file without corresponding img
        if Path(input_img_dir) / (label_path.stem + '.png') in image_list:
            continue
        elif Path(input_img_dir) / (label_path.stem + '.jpg') in image_list:
            continue
        elif Path(input_img_dir) / (label_path.stem + '.jpeg') in image_list:
            continue
        else:
            # rich.print(f"[bold red]No corresponding img: {label_path}, moved.")
            if verbose:
                LOGGER.warning(f"No corresponding img: {label_path}, moved.")
            shutil.move(str(label_path), mv_dir)



    # 3. 检查所有label都有内容，不是空的; 此刻，size(img) = size(label)
    # empty label => remove img & label 
    if not dont_clean_empty_txt:
        rich.print(f"> Checking if all labels are not empty...")
        image_list, label_list = load_img_label_list(input_img_dir, input_label_dir, img_format, info)
        for label_path in tqdm(label_list):   

            # size < 10 => empty
            if os.path.getsize(str(label_path)) < 10:

                # revome label & img
                # rich.print(f"[bold red]Empty label file: {label_path}, moved.")
                if verbose:
                    LOGGER.warning(f"Empty label file: {label_path}, moved.")
                shutil.move(str(label_path), mv_dir)
                label_list.remove(label_path)

                # remove corresponding img
                img_path_png = Path(input_img_dir) / (label_path.stem + '.png')
                img_path_jpg = Path(input_img_dir) / (label_path.stem + '.jpg')
                img_path_jpeg = Path(input_img_dir) / (label_path.stem + '.jpeg')

                # PNG
                if img_path_png in image_list:
                    # rich.print(f"[bold red]=>corresponding img file: {label_path}, moved.")
                    if verbose:
                        LOGGER.warning(f"-> corresponding img file: {img_path_png}, moved.")
                    shutil.move(str(img_path_png), mv_dir)
                    
                # JPG
                elif img_path_jpg in image_list:
                    # rich.print(f"[bold red]=>corresponding img file: {label_path}, moved.")
                    if verbose:
                        LOGGER.warning(f"-> corresponding img file: {img_path_jpg}, moved.")
                    shutil.move(str(img_path_jpg), mv_dir)
                    
                # JPEG
                elif img_path_jpeg in image_list:
                    # rich.print(f"[bold red]=>corresponding img file: {label_path}, moved.")
                    if verbose:
                        LOGGER.warning(f"-> corresponding img file: {img_path_jpeg}, moved.")
                    shutil.move(str(img_path_jpeg), mv_dir)
                

    # 4. clean up IMG-dir, Label-dir
    rich.print(f"> Cleaning-up img-dir * label-dir...")
    item_list = list(Path(input_img_dir).iterdir())
    for p in tqdm(item_list):
        if p.suffix in list(img_format) + ['.txt']:
            continue
        if verbose:
            LOGGER.warning(f"Not support format: {p.suffix} --> {p}, moved.")
        shutil.move(str(p.resolve()), mv_dir)

    # show after check result info
    image_list, label_list = load_img_label_list(input_img_dir, input_label_dir, img_format, False)
    rich.print(f"[green]> Valid Labels count: {len(label_list)}")
    rich.print(f"[green]> Valid Images count: {len(image_list)}")

    # 如果mv_dir为空，就删掉
    mv_dir_list = list(Path(mv_dir).iterdir())
    if len(mv_dir_list) == 0:
        Path(mv_dir).rmdir()

    # prompt
    rich.print(f"[bold gold1]> You should run this command servalal times until it's unchanged!")


#---------
# Usage:
    '''
    save_dir = increment_path(Path(project) / name, exist_ok=False, sep='-')  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir 中间目录存在不报错
    '''
#---------

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path


# letter box: but not rect box
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # if not scaleup:  # only scale down, do not scale up (for better val mAP)
    #     r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    # if auto:  # minimum rectangle
    #     dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    # if scaleFill:  # stretch
    #     dw, dh = 0.0, 0.0
    #     new_unpad = (new_shape[1], new_shape[0])
    #     ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)



# get path (img)
def get_img_path(img_dir, output_txt, img_format, append=True):

    img_dir = Path(img_dir)
    saveout = Path(output_txt)

    if append:
        f = open(str(saveout), 'a')
    else:
        f = open(str(saveout), 'w')

    image_list = [x for x in Path(img_dir).iterdir() if x.suffix in img_format]
    LOGGER.info(f"image num: {len(image_list)}")

    for p in image_list:
        f.write(str(p.resolve()) + '\n')
    f.close()





def resource_info(refresh_time=0.5, display=False):

    # cpu info
    cpu_count = psutil.cpu_count(logical=False), psutil.cpu_count(logical=True)  # logical, virtual
    cpu_usage = psutil.cpu_percent(percpu=True, interval=refresh_time)
    cpu_usage_avg = psutil.cpu_percent(percpu=False, interval=refresh_time)
    cpu_load_average = psutil.getloadavg()  # sum(cpu_usage) / cpu_count[1]


    # mem info
    mem = psutil.virtual_memory()
    mem_swap = psutil.swap_memory()

    # create table
    table = rich.table.Table(title="\n[bold cyan]CPU & MEM INFO", 
                            box=rich.box.ASCII2, 
                            show_lines=False, 
                            caption=f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} (refresh time: {refresh_time})\n",  # Time
                            caption_justify='center',
                        )

    # add table column
    table.add_column(f"CPU\nUSAGE", justify="center", style="cyan", no_wrap=True)
    # table.add_column(f"CPU\nUSAGE_PER_CORE", justify="center", style="cyan", no_wrap=True)
    table.add_column("CPU\nCOUNT", justify="center", style="cyan", no_wrap=True)
    table.add_column(f"CPU\nLOAD_AVERAGE", justify="center", style="cyan", no_wrap=True)
    table.add_column(f"MEM\nUSAGE", justify="center", style="cyan", no_wrap=True)
    table.add_column(f"MEM\nTOTAL(GB)", justify="center", style="cyan", no_wrap=True)
    table.add_column(f"MEM\nUSED(GB)", justify="center", style="cyan", no_wrap=True)
    table.add_column(f"MEM\nAVAILABLE(GB)", justify="center", style="cyan", no_wrap=True)
    table.add_column(f"MEM_SWAP\nUSAGE(GB)", justify="center", style="cyan", no_wrap=True)
    table.add_column(f"MEM_SWAP\nTOTAL(GB)", justify="center", style="cyan", no_wrap=True)

    # add table row
    table.add_row(
        f"{cpu_usage_avg}%", 
        # f"{cpu_usage}", 
        f"{cpu_count}",
        f"{cpu_load_average}%", 
        f"{mem.percent}%", 
        f"{(mem.total / (1 << 30)):.3f}",   # 2^30, kb=2^10
        f"{(mem.used / (1 << 30)):.3f}", 
        f"{(mem.available / (1 << 30)):.3f}", 
        f"{mem_swap.percent}%", 
        f"{(mem_swap.total / (1 << 30)):.4}", 
        end_section=True
    )

    # display
    if display is True:
        CONSOLE.print(table)

    return table


def gpu_info(display=False):


    # # check if has gpu
    # if onnxruntime.get_device() != 'GPU':
    #     LOGGER.error(f"This machine has no GPU device!")
    #     return
    # else:
    #     pynvml.nvmlInit()  # init
    try:
        pynvml.nvmlInit()  # init
    except Exception:
        LOGGER.error(f"This machine has no GPU device!")
        return   

    # create table
    table = rich.table.Table(title="\n[bold cyan]GPU INFO", 
                            box=rich.box.ASCII2, 
                            show_lines=False, 
                            caption=f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}\n",  # Time
                            caption_justify='center',
                        )

    # add table column
    table.add_column("ID", justify="center", style="cyan", no_wrap=True)
    table.add_column("NAME", justify="center", style="cyan", no_wrap=True)
    table.add_column("USED", justify="center", style="cyan", no_wrap=True)
    table.add_column("TOTAL", justify="center", style="cyan", no_wrap=True)
    table.add_column(f"USAGE", justify="center", style="cyan", no_wrap=True)


    pynvml.nvmlInit()  # init
    for index in range(pynvml.nvmlDeviceGetCount()):   # num_gpu
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)  # handle
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)  # gpu mem

        # add table row
        table.add_row(
            f"{index}", 
            f"{str(pynvml.nvmlDeviceGetName(handle), encoding='utf-8')}",  
            f"{(mem.used / MB):.3f} MB",  
            f"{(mem.total / MB):.3f} MB",  
            f"{(mem.used / MB) / (mem.total / MB) * 100:.4f} %",
            end_section=True
        )

    # display
    if display:
        CONSOLE.print(table)

    pynvml.nvmlShutdown()  # close
    return table




