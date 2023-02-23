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
from omegaconf import OmegaConf, DictConfig

# ---------------------------------------------------------------------------------------------
from usls.src.utils import (
    get_corresponding_label_path, Colors, load_img_label_list,
    img_label_dir_cleanup, get_img_path, LOGGER, IMG_FORMAT, 
    is_point_in_rect
)
from usls.src.labelling_det import inspect
from usls.src.labelling_cls import classify
from usls.src.video_tools import play_and_record, video_to_images, videos_to_images, images_to_video
from usls.src.spider import spider_img_baidu
from usls.src.dir_combine import dir_combine
from usls.src.label_combine import combine_labels
from usls.src.deduplicate import deduplicate
from usls.src.class_modify import class_modify
# ---------------------------------------------------------------------------------------------


def run(opt: DictConfig):

    # ------------------------
    #   common vars
    # ------------------------


    # -------------------------------------
    #   img & label dir info 
    # -------------------------------------
    if opt.task == 'info':

        img_dir = opt.get('img_dir')
        label_dir = opt.label_dir if opt.get('label_dir') else img_dir

        assert all((img_dir, label_dir)), f"Not `img_dir=???` or `label_dir=???` input!"
        assert Path(img_dir).is_dir() and Path(label_dir).is_dir(), f"`img_dir={img_dir}` or `label_dir={label_dir}` is not a dir path"


        rich.print(f"\n-------INFO----------")
        image_list = [x for x in Path(img_dir).iterdir() if x.suffix in IMG_FORMAT]
        label_list = list(Path(label_dir).glob("*.txt"))
        rich.print(f"> Images count: {len(image_list)}")
        rich.print(f"> Labels count: {len(label_list)}")    
        rich.print(f'---------------------')


    # -------------------------------------
    #   inspect 
    # -------------------------------------
    if opt.task in ('inspect', 'classify'):

        img_dir = opt.get('img_dir')
        label_dir = opt.label_dir if opt.get('label_dir') else img_dir

        assert all((img_dir, label_dir)), f"Not `img_dir=???` or `label_dir=???` input!"
        assert Path(img_dir).is_dir() and Path(label_dir).is_dir(), f"`img_dir={img_dir}` or `label_dir={label_dir}` is not a dir path"

        mv_dir = opt.mv_dir if opt.get('mv_dir') else 'moved_dir'  # save moved dir
        wrong_img_dir = opt.wrong_img_dir if opt.get('wrong_img_dir') else 'wrong_img_dir'  # wrong dir


        assert opt.get('classes'), f"No `classes=???` args when task is `inspect`!"
        classes = opt.classes.split(',')
        window_width = opt.window_width if opt.get('window_width') else 800  # save moved dir
        window_height = opt.window_height if opt.get('window_height') else 600  # save moved dir

        # det
        if opt.task == 'inspect':

            inspect( 
                img_dir, 
                label_dir,
                mv_dir,
                wrong_img_dir,
                classes,
                window_width,
                window_height,
            )

        # cls
        if opt.task == 'classify':
            classify( 
                img_dir, 
                label_dir,
                mv_dir,
                wrong_img_dir,
                classes,
                window_width,
                window_height,
            )




    # -------------------------------------
    #   img & label dir clean up 
    # -------------------------------------
    if opt.task in ('clean', 'cleanup'):
        img_dir = opt.get('img_dir')
        label_dir = opt.label_dir if opt.get('label_dir') else img_dir

        assert img_dir, f"No `img_dir=???` input!"
        assert label_dir, f"No `label_dir=???` input!"
        assert Path(img_dir).is_dir() and Path(label_dir).is_dir(), f"`img_dir={img_dir}` or `label_dir={label_dir}` is not a dir path"

        mv_dir = opt.mv_dir if opt.get('mv_dir') else 'moved_dir'  # save moved dir

        if opt.get('clean_empty'):  
            if opt.clean_empty.lower() == 'false':
                clean_empty = True
            else:
                clean_empty = False
        else:
            clean_empty = False

        rich.print(f"-------CLEAN_UP-------")
        rich.print("> IMG Directory:", Path(img_dir).resolve())
        rich.print("> LABEL Directory:", Path(label_dir).resolve())

        # in case of other keys input
        while True: 
            user_answer = input('> do clean-up? ---> [yes/no]: ')
            if user_answer.lower() in ('n', 'no'):
                sys.exit("Cancelled cleanup!")

            elif user_answer.lower() in ('y', 'yes'):
                img_label_dir_cleanup(
                    img_dir, 
                    label_dir, 
                    mv_dir, 
                    IMG_FORMAT, 
                    info=True, 
                    dont_clean_empty_txt=clean_empty  # false default
                )

                break



    # -------------------------------------
    #   label combine
    # -------------------------------------
    if opt.task == 'label_combine':

        assert opt.get('input_dir'), f"No `input_dir=???` args when task is `label_combine`! default: `output_dir=output-label-combine`"
        input_dir = opt.input_dir
        output_dir = opt.output_dir if opt.get('output_dir') else 'output-label-combine'
        combine_labels(input_dir=input_dir, output_dir=output_dir)


    # -------------------------------------
    #   dir combine
    # -------------------------------------
    if opt.task == 'dir_combine':
        assert opt.get('input_dir'), f"No `input_dir=???` args when task is `label_combine`! default: `output_dir=output-label-combine`"
        input_dir = opt.input_dir
        output_dir = opt.output_dir if opt.get('output_dir') else 'output-dir-combine'

        if opt.get('suffix'):
            suffix = opt.suffix.split(',')
        else:
            suffix = []

        if opt.get('move'):
            if opt.move.lower() == 'true':
                move = True
            else:
                move = False
        else:
            move = False

        dir_combine(
            input=input_dir,
            output=output_dir,
            suffix=suffix,
            move=move
        )


    # -------------------------------------
    #  video_tools => video to images 
    # -------------------------------------
    if opt.task in ('v2is', 'vs2is'):

        frame = float(opt.frame) if opt.get('frame') else 20

        # view
        if opt.get('view'):
            if opt.view.lower() == 'true':
                view = True
            else:
                view = False
        else:
            view = False

        # flip
        if opt.get('flip'):
            if opt.flip.lower() == 'true':
                flip = True
            else:
                flip = False
        else:
            flip = False

        # fmt_img
        fmt_img = opt.fmt_img if opt.get('fmt_img') else '.jpg' 

        if opt.task == 'v2is':
            assert opt.get('source'), f"No `source=???` args when task is `v2is`!"
            source = opt.source
            output_dir = opt.output_dir if opt.get('output_dir') else 'v2is'

            video_to_images(
                source=source,      
                output=output_dir,      
                x=frame,
                view=view,
                flip=flip,
                img_fmt=fmt_img
            )
        elif opt.task == 'vs2is':
            assert opt.get('input_dir'), f"No `input_dir=???` args when task is `vs2is`!"
            input_dir = opt.input_dir
            output_dir = opt.output_dir if opt.get('output_dir') else 'vs2is'

            videos_to_images(
                input_dir=input_dir,
                output_dir=output_dir,
                x=frame,
                view=view,
                flip=flip,
                img_fmt=fmt_img
            )



    # -------------------------------------
    #   video_tools => play and rec 
    # -------------------------------------
    if opt.task == 'play':
        assert opt.get('source'), f"No `source=???` args when task is `play`!"
        source = opt.source

        delay = int(opt.delay) if opt.get('delay') else 1
        # flip
        if opt.get('flip'):
            if opt.flip.lower() == 'true':
                flip = True
            else:
                flip = False
        else:
            flip = False

        play_and_record(
            source=source, 
            delay=delay, 
            flip=flip
        )


    # -------------------------------------
    # video_tools => images to video
    # -------------------------------------
    if opt.task == 'is2v':
        assert opt.get('input_dir'), f"No `input_dir=???` args when task is `is2v`!"
        input_dir = opt.input_dir
        fps = int(opt.fps) if opt.get('fps') else 30
        last4 = int(opt.last4) if opt.get('last4') else 60

        if opt.get('video_size'):
            if ',' in opt.video_size:
                video_size = list(map(int, opt.video_size.split(',')))
            else:
                video_size = (int(opt.video_size), int(opt.video_size))

        else:
            video_size = (640, 640)

        images_to_video(
            source=input_dir, 
            last4=last4, 
            fps=fps, 
            size=video_size,
        )


    # -------------------------------------
    #   spider image from baidu
    # -------------------------------------
    if opt.task == 'spider':
        assert opt.get('words'), f"No `words=???` args when task is `spider`!"
        words = opt.words.split(',')
        spider_img_baidu(words)

    # -------------------------------------
    #   de-duplicate
    # -------------------------------------
    if opt.task == 'deduplicate':
        assert opt.get('input_dir'), f"No `input_dir=???` args when task is `deduplicate`!"
        input_dir = opt.input_dir
        mv_dir = opt.mv_dir if opt.get('mv_dir') else 'move_deduplicate_dir'

        deduplicate(
            input_dir=input_dir, 
            move_dir=mv_dir, 
            info=False
        )

    # -------------------------------------
    #   class modify
    # -------------------------------------
    if opt.task == 'class_modify':
        assert opt.get('input_dir'), f"No `input_dir=???` args when task is `class_modify`!"
        input_dir = opt.input_dir
        
        assert opt.get('to'), f"No `to=???` args when task is `class_modify`!"
        to = opt.to

        class_modify(
            input_dir=input_dir, 
            to=to
        )

