
import argparse
import os
import re
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys
import shutil
import random
import time
# from loguru import logger as LOGGER
import json



#--------------------------------------------
#       ADD TO PYTHON_PATH
#--------------------------------------------
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]   # FILE.parent 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
#--------------------------------------------

from utils import Colors, IMG_FORMAT, is_point_in_rect, LOGGER


#--------------------------------------------
#          Global Variables
#--------------------------------------------

# windows 
WINDOW_NAME = 'MLOps-Dataset-Classifying'

# tracker bars
TRACKBAR_IMG = 'IMAGES'


# input dir
INPUT_IMG_DIR  = ""
IMAGE_CLASSES_JSON_PATH = None


# mode 
CLASSIFY_MODE = False


# images
IMAGE_PATH_LIST = [] 
IMG_IDX_CURRENT = 0         # 当前的img index
IMG_IDX_LAST = 0            # last 的img index
IMG_CURRENT = None          # 当前页面显示的img
WRONG_IMG_SET = set()       # 无法正常读取的image
IMG_COUNT = 0

# classes
CLASS_LIST = []             # all class ID
CLS_IDX_CURRENT = 0         # currentclass index
CLS_COUNT = 0
IMAGE_CLASSES = {}   # {image_path: class_id}


# point_xy & mouse_xy
MOUSE_X = 0
MOUSE_Y = 0


#--------------------------------------------
#          Functions
#--------------------------------------------

# √ display text in the [overlap, terminal, status_bar]
def print_info(text="for example", ms=1000, where=None):
    global WINDOW_NAME

    if where == 'Overlay':
        cv2.displayOverlay(WINDOW_NAME, text, ms)
    elif where == 'Statusbar':
        cv2.displayStatusBar(WINDOW_NAME, text, ms)
    else:
        LOGGER.info(f"{text}")


# set current img index & imshow image
def set_img_index(x):
    global IMG_IDX_CURRENT, IMG_CURRENT, WINDOW_NAME, IMAGE_PATH_LIST, WRONG_IMG_SET

    IMG_IDX_CURRENT = x
    img_path = IMAGE_PATH_LIST[IMG_IDX_CURRENT]
    
    # opencv read img
    IMG_CURRENT = cv2.imread(img_path)
    if IMG_CURRENT is None:
        IMG_CURRENT = np.ones((1000, 1000, 3))  # create a empty img
        # show notification
        cv2.putText(IMG_CURRENT, "Wrong image format! It will delete after pressing ESC.", 
                    (10, IMG_CURRENT.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0,0,0), thickness=2, lineType=cv2.LINE_AA)
        # save wrong images path, delete all these image at the end of the program
        WRONG_IMG_SET.add(img_path)



# mouse callback function
def mouse_listener(event, x, y, flags, param):
    global MOUSE_X, MOUSE_Y, CLASSIFY_MODE

    # mark mode
    if CLASSIFY_MODE:
        if event == cv2.EVENT_MOUSEMOVE:
            MOUSE_X = x
            MOUSE_Y = y



def cmp(s, r=re.compile('([0-9]+)')):
    # key for sort
    return [int(x) if x.isdigit() else x.lower() for x in r.split(s)]


def opencv_window_init():
    # init window with overlap
    try:
        cv2.namedWindow('Test')   
        cv2.displayOverlay('Test', 'Test overlap', 10)  
        cv2.displayStatusBar('Test', 'Test status bar', 10)
    except cv2.error:
        print('-> Please ignore this error message\n')
    cv2.destroyAllWindows()   




# ---------------------------------------------------
#   classify 
#--------------------------------------------------
def classify( img_dir, 
             label_dir,
             mv_dir,
             wrong_img_dir,
             classes,
             window_width=800,
             window_height=600,
             ):
    
    # global vars
    global WINDOW_NAME,\
           CLASSIFY_MODE, IMAGE_CLASSES, \
           IMAGE_PATH_LIST, IMG_IDX_CURRENT, IMG_IDX_LAST, IMG_CURRENT, WRONG_IMG_SET,\
           CLASS_LIST, CLS_IDX_CURRENT, \
           TRACKBAR_IMG,\
           MOUSE_X, MOUSE_Y,\
           IMG_COUNT, CLS_COUNT
    

    # input img dir & label dir
    INPUT_IMG_DIR  = img_dir
    LOGGER.info(f"IMG   DIR:\t{Path(INPUT_IMG_DIR).resolve()}")

    #-----------------------------------------   
    WINDOW_INIT_WIDTH = window_width    # initial window width
    WINDOW_INIT_HEIGHT = window_height    # initial window height

    # mark mode 
    CLASSIFY_MODE = False

    # wrong dir & move dir
    WRONG_IMG_DIR = wrong_img_dir
    MV_DIR = mv_dir


    # only show one specific class
    SINGLE_CLS = None

    # min line width
    MIN_LINE_WIDTH = False

    # bboxes blink
    BLINK_OR_NOT = False

    # line thickness  &  line thickes adjust
    LINE_THICKNESS = 1            
    LINE_THICKNESS_ADJUST = False   # line thickness adjust flag

    # ----------------------
    # CLASS_LIST
    # ----------------------
    if len(classes) == 0:   # no classes
        LOGGER.error("Error: <--inspect> should work with <--classes>! EXIT!")
        exit(-1)
    elif len(classes) == 1 and classes[0].endswith('.txt'):    # txt input
        with open(classes[0]) as f:
            for line in f:
                CLASS_LIST.append(line.strip())
    else: # args classes 
        CLASS_LIST = classes

    # repeat class check 
    if not (len(CLASS_LIST) == len(set(CLASS_LIST))):
        LOGGER.error("Repeat class name!!!")
        exit(-1)

    # opencv windows init
    opencv_window_init()

    # read all input images
    LOGGER.info(f"Loading all images...")
    IMAGE_PATH_LIST = sorted([str(x) for x in Path(INPUT_IMG_DIR).iterdir() if x.suffix in IMG_FORMAT], key=cmp)

    # image class json file init
    IMAGE_CLASSES_JSON_PATH = Path(INPUT_IMG_DIR).with_name('classify.json')

    # load json for classify task
    if (IMAGE_CLASSES_JSON_PATH).exists():
        IMAGE_CLASSES = json.load(open(IMAGE_CLASSES_JSON_PATH, encoding="utf-8"))

    # img & class count
    IMG_COUNT = len(IMAGE_PATH_LIST) - 1  
    CLS_COUNT = len(CLASS_LIST) - 1


    # create window 
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)  # cv2.WINDOW_FREERATIO   cv2.WINDOW_KEEPRATIO, WINDOW_GUI_NORMAL, WINDOW_GUI_EXPANDED
    cv2.resizeWindow(WINDOW_NAME, WINDOW_INIT_WIDTH, WINDOW_INIT_HEIGHT)

    # mouse listen callback
    cv2.setMouseCallback(WINDOW_NAME, mouse_listener)

    # images trackbar
    if IMG_COUNT != 0:
        cv2.createTrackbar(TRACKBAR_IMG, WINDOW_NAME, 0, IMG_COUNT, set_img_index)   
 

    # initialize the img index
    set_img_index(0)

    # colors palette
    COLOR_PALETTE = Colors(shuffle=False)  
    LOGGER.info(f"running...")

    # loop
    while True:
        color = COLOR_PALETTE(int(CLS_IDX_CURRENT), bgr=False)  # color for every class
        tmp_img = IMG_CURRENT.copy()    # clone the img   
        img_height_current, img_width_current = tmp_img.shape[:2]   # height, width

        # calculate line-thickness
        if MIN_LINE_WIDTH:
            LINE_THICKNESS = 1
        else:
            LINE_THICKNESS = max(round(sum(tmp_img.shape) / 2 * 0.003), 1) if not LINE_THICKNESS_ADJUST else LINE_THICKNESS      # line width

        # current class index and it's class name
        # class_name = CLASS_LIST[CLS_IDX_CURRENT]
        
        # current image path, relative path: img/img_1.jpg
        img_path = IMAGE_PATH_LIST[IMG_IDX_CURRENT]   
        
        # statusbar info
        status_msg = (f"CURSOR: ({MOUSE_X}, {MOUSE_Y})" + "\t" * 8 + 
                      f"CLASS: {CLASS_LIST}" + "\t" * 8 +
                      f"IMG RESOLUTION: ({tmp_img.shape[0]}, {tmp_img.shape[1]})" + "\t" * 5 +
                      f"IMAGE PATH: {Path(img_path).name}"  + "\t" * 10
                    )
        cv2.displayStatusBar(WINDOW_NAME, status_msg)

        


        # ---------------------------- 
        #   CLASSIFY_MODE
        # ----------------------------
        # if CLASSIFY_MODE:

        # hashmap to save current image and its class {image_path: class}
        img_cls = IMAGE_CLASSES.get(img_path)

        # text attr
        if img_cls is not None:
            txt_msg = CLASS_LIST[img_cls]   # str(img_cls)
            color_ = COLOR_PALETTE(int(img_cls), bgr=False)  # color for every class
            pos_ = (tmp_img.shape[0] // 10, tmp_img.shape[1] // 5)
            scale_ = LINE_THICKNESS / 1.5
        else:
            txt_msg = "Not classify!"
            color_ = (0, 0, 255)
            pos_ = (tmp_img.shape[0] // 10, tmp_img.shape[1] // 5)
            scale_ = LINE_THICKNESS / 2

        # put text
        cv2.putText(tmp_img, 
                    txt_msg, 
                    pos_,
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    scale_, 
                    color_, 
                    thickness=LINE_THICKNESS, 
                    lineType=cv2.LINE_AA)


        # dump to json file
        json.dump(IMAGE_CLASSES, open(IMAGE_CLASSES_JSON_PATH, 'w'), indent=4)
        # ----------------------------


        # current show
        cv2.imshow(WINDOW_NAME, tmp_img)
        
        # -----------------------------------------
        # opencv key listening
        # -----------------------------------------
        # key listen
        pressed_key = cv2.waitKey(1)

        # h/H => help 
        if pressed_key in (ord('h'), ord('H')):
            text = ('[ESC] to quit;\n'
                    '[r]switch mode: mark? read?;\n'
                    '[a/d] to switch Image;\n'
                    '[w/s] to switch Class;\n'
                    '[double click to select] + w/s can change class;\n'
                    '[-/+] to adjust line-thickness;\n'
                    '[n] to hide labels;\n'
                    '[b] to blink the bboxes in the img;\n'
                    '[l] to shuffle bbox colors;\n'
                    '[c] to to remove all bboxes;\n'
                    )

            print_info(text, ms=1000, where="Overlay")

        # ---------------------------------------
        # a,d -> images [previous, next]
        # ---------------------------------------
        elif pressed_key in (ord('a'), ord('A'), ord('d'), ord('D')):
            IMG_IDX_LAST = IMG_IDX_CURRENT  # last image index

            # show previous image
            if pressed_key in (ord('a'), ord('A')):     
                IMG_IDX_CURRENT = 0 if IMG_IDX_CURRENT - 1 < 0 else IMG_IDX_CURRENT - 1

            # show next image index
            elif pressed_key in (ord('d'), ord('D')):
                IMG_IDX_CURRENT = IMG_COUNT if IMG_IDX_CURRENT + 1 > IMG_COUNT else IMG_IDX_CURRENT + 1

            # update img trackbar 
            cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, IMG_IDX_CURRENT)

            # set the adjust flag False
            LINE_THICKNESS_ADJUST = False


        # ---------------------------------------
        # w,s -> class  [previous, next]
        # ---------------------------------------
        elif pressed_key in (ord('s'), ord('S'), ord('w'), ord('W')):
            
            if CLASSIFY_MODE:
                if pressed_key in (ord('s'), ord('S')):     # next class
                    CLS_IDX_CURRENT = CLS_COUNT if CLS_IDX_CURRENT - 1 < 0 else CLS_IDX_CURRENT - 1     # loop
                elif pressed_key in (ord('w'), ord('W')):   # last class
                    CLS_IDX_CURRENT = 0 if CLS_IDX_CURRENT + 1 > CLS_COUNT else CLS_IDX_CURRENT + 1

                # 
                # if IMAGE_CLASSES.get(img_path) is not None:
                IMAGE_CLASSES.update({img_path: CLS_IDX_CURRENT})



        # ---------------------------------------
        # '+ -' => bold line thickness
        # ---------------------------------------
        elif pressed_key in (ord('='), ord('+')):

            # set the adjust flag TRUE
            LINE_THICKNESS_ADJUST = True
            
            # get the max line width
            max_t = max(round(sum(tmp_img.shape) / 2 * 0.003), 2) + 5

            # increate the line width
            if LINE_THICKNESS <= max_t:
                LINE_THICKNESS += 1
                print_info(f'Line Thickness +1, now = {LINE_THICKNESS}', ms=1000, where="Overlay")
            else:
                print_info('Line Thickness has reach the max value!', ms=1000, where="Overlay")

        elif pressed_key in (ord('-'), ord('_')):
            LINE_THICKNESS_ADJUST = True
            min_t = 1
            if LINE_THICKNESS > min_t:
                LINE_THICKNESS -= 1
                print_info(f'Line Thickness -1, now = {LINE_THICKNESS}', ms=1000, where="Overlay")
            else: 
                print_info('Line Thickness has reach the min value!', ms=1000, where="Overlay")


        # ---------------------------------------
        # c/C  =>  Remove all bboxes in this img, specifically, delete the annotation file(.txt)
        # ---------------------------------------
        elif pressed_key in (ord('c'), ord('C')):
            # delete item in image_class
            if IMAGE_CLASSES.get(img_path) is not None:
                IMAGE_CLASSES.pop(img_path)
                print_info(f"class removed", ms=1000, where="Overlay")

        # ---------------------------------------
        # r/R  =>  switch mode
        # ---------------------------------------
        elif pressed_key in (ord('r'), ord('R')):
            print_info(f"Switch mode between READ and MARK", ms=1000, where="Overlay")
            CLASSIFY_MODE = not CLASSIFY_MODE

        # ---------------------------------------
        # l/L  =>  shuffle bbox color
        # ---------------------------------------
        elif pressed_key in (ord('l'), ord('L')):
            COLOR_PALETTE = Colors(shuffle=True)
            print_info(f"Colors palette shuffled!", ms=1000, where="Overlay")


        # ---------------------------------------
        # t/T  =>  min line width
        # ---------------------------------------
        elif pressed_key in (ord('t'), ord('T')):
            MIN_LINE_WIDTH = not MIN_LINE_WIDTH


        # ---------------------------------------
        # 0-9 -> change img's class
        # ---------------------------------------
        elif pressed_key in range(48, 58):  # 0-8 => 48-56

            if CLASSIFY_MODE:
                value = int(chr(pressed_key))
                if value < len(CLASS_LIST): 
                    # CLS_IDX_CURRENT = value
                    IMAGE_CLASSES.update({img_path: value})
                else:
                    print_info(f"max class id is {len(CLASS_LIST)}", ms=1000, where="Overlay")

            else:
                print_info(f"Not in classify mode! press r/R to start!", ms=1000, where="Overlay")


        # ---------------------------------------
        # ESC -> quit key listener
        # ---------------------------------------
        elif pressed_key == 27:
            break
        # ---------------- Key Listeners END ------------------------

        # if window gets closed then quit
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


    # ---------------------------
    #  deal with classify dict
    # ---------------------------
    if IMAGE_CLASSES:

        # check if json file is same with IMAGE_CLASS
        assert json.load(open(IMAGE_CLASSES_JSON_PATH, encoding="utf-8")) == IMAGE_CLASSES

        # in case of other keys input
        while True: 
            response = input('> Find classify.json! Do classifying? [yes / no]: ')
            if response.lower() in ('n', 'no'):   # exit
                LOGGER.info('Not doing classifying')
                break

            elif response.lower() in ('y', 'yes'):
                LOGGER.info('Doing classifying')

                # iterate IMAGE_CLASSES to move image to dirs
                for idx, (k, v) in enumerate(tqdm(IMAGE_CLASSES.items(), desc='Classifying')):
                    # mkdir dirs for different class: CLASS/class_a, CLASS/class_b, CLASS/class_c 
                    dst_class_dir = Path('IMAGE-CLASS') / CLASS_LIST[v-1]
                    if not dst_class_dir.exists():
                        dst_class_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(k, str(dst_class_dir))

                break

    else:
        # delete json.file if it is empty
        Path(IMAGE_CLASSES_JSON_PATH).unlink()



    # ---------------------------
    # deal with wrong img
    # last step: 删除所有无法被opencv读取的图像
    # ---------------------------
    if len(WRONG_IMG_SET) > 0:
        LOGGER.warning(f"has {len(WRONG_IMG_SET)} images can not be read by OpenCV, moving to {WRONG_IMG_DIR}")
        
        # create dir if not exist
        if not Path(WRONG_IMG_DIR).exists():
            Path(WRONG_IMG_DIR).mkdir()

        # remove
        for img in WRONG_IMG_SET:
            shutil.move(img, WRONG_IMG_DIR)
            LOGGER.info(f"{Path(img).resolve()}")
    else:
        LOGGER.info(f"Every image can be read by OpenCV.")



# ----------------options ------------------------
def parse_opt():
    parser = argparse.ArgumentParser(description='Open-source image labeling tool')
    parser.add_argument('--img-dir', default='img-dir', type=str, help='Path to input directory')
    parser.add_argument('--label-dir', default='', type=str, help='Path to output directory')
    parser.add_argument('--mv-dir', default="moved_dir", type=str, help='mv-dir to save moved data[img, label]')
    parser.add_argument('--wrong-img-dir', default="wrong-img-dir", type=str, help='wrong format img to save imgs opencv cant read')
    parser.add_argument('--classes', default='', nargs="+", type=str, help='classes list text')
    parser.add_argument('--window_width', default=800, type=int, help='classes list text')
    parser.add_argument('--window_height', default=600, type=int, help='classes list text')
    opt = parser.parse_args()

    return opt



# ---------------------------------------------------
#   main
#--------------------------------------------------
if __name__ == '__main__':

    opt = parse_opt()
    inspect( opt.img_dir, 
             opt.label_dir,
             opt.mv_dir,
             opt.wrong_img_dir,
             opt.classes,
             opt.window_width,
             opt.window_height,
             )
