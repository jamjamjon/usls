#!/usr/bin/env python
# -*- coding:utf-8 -*- 

import cv2
import os
import rich
from tqdm import tqdm
import argparse
from pathlib import Path
import sys


#--------------------------------------------
#       ADD TO PYTHON_PATH
#--------------------------------------------
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]   # FILE.parent 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # abs path => relative
#--------------------------------------------


from utils import increment_path
from utils import LOGGER, VIDEO_FORMAT, IMG_FORMAT, letterbox




# video play & record
def play_and_record(source, delay=1, flip=False):

    # check video path
    # if Path(source).is_dir():
    #     LOGGER.error("video path is WRONG!")
    #     exit(-3)

    # video_cap 
    videoCapture = cv2.VideoCapture(source)

    fps = int(videoCapture.get(cv2.CAP_PROP_FPS))
    w = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_size = (w, h)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    LOGGER.info(f"Video info: width={w}, height={h}, FPS={fps}")


    # record flag
    do_rec = False
    LOGGER.info(f"Not recording! Press [r] to record, Press again to stop recording.")


    while True:
        ret, frame = videoCapture.read()
        if ret:
            cv2.imshow('frame', frame)
            # flip
            if flip:
                frame = cv2.flip(frame, 0)

            # rec
            if do_rec:
                video_writer.write(frame)

            # key detect
            key = cv2.waitKey(delay)

            # q -> quit
            if key == ord('q'):
                break

            # r -> record
            if key == ord('r'):

                # ~                     
                do_rec = not do_rec 

                # rec 
                if do_rec:
                    LOGGER.info(f"Rec...")

                    dir_name = 'video_record'
                    sub_name = 'rec'
                    save_dir = increment_path(Path(dir_name) / sub_name, exist_ok=False, sep='')  # increment run
                    save_dir.mkdir(parents=True, exist_ok=True)

                    video_name = 'rec.mp4'
                    saveout = save_dir / video_name 
                    # print(saveout)
                    video_writer = cv2.VideoWriter(str(saveout), fourcc, fps, video_size)

                else:
                    LOGGER.info(f"Done rec. Saved at: {saveout.resolve()}")

        else:
            # print('can not read frame!')
            break


    # release cap & video cap
    videoCapture.release()
    cv2.destroyAllWindows()

    LOGGER.info(f"Done.")



# video -> images
def video_to_images(source,                             # 0: cam
                    output="V2I",      # save dir
                    x=20,         # every 20 frame  to save
                    view=False,
                    flip=False,
                    img_fmt=".jpg",
                    verbose=True
                    ):

    # check video path
    if not Path(source).is_file():
        LOGGER.error("video path is WRONG!")
        exit(-3)

    # frame count
    idx = 0 

    # save_dir 
    save_dir = increment_path(Path(output), exist_ok=False, sep='_')  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir 中间目录存在不报错


    # load video
    cap = cv2.VideoCapture(str(source))

    # read video
    if verbose:
        rich.print(f"> [cyan]Spliting...")   
    
    while True:
        ret, frame = cap.read()
        if ret == True:

            # flip frame
            if flip:
                frame = cv2.flip(frame, 0)   

            # rotate
            # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # show video 
            if view:       
                cv2.imshow('video', frame)

            # video to img
            if idx % x == 0:                
                img_saveout = save_dir / (Path(source).stem + '_' + str(idx) + img_fmt)
                cv2.imwrite(str(img_saveout), frame)
            
            # frame index counting
            idx += 1

            # 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # LOGGER.error(f"Not capture video frame! Please check the video path.")
            break


    # release cap
    cap.release()

    # close opencv windows if opened.
    if view:
        cv2.destroyAllWindows()


    if verbose:
        # successful info
        LOGGER.info(f"Images saved at: {save_dir.resolve()}")



# batch videos -> images
def videos_to_images(
                    input_dir,
                    output_dir='',
                    x=30,
                    view=False,
                    flip=False,
                    img_fmt=".jpg",
                    ):

    # video list
    video_list = [x for x in Path(input_dir).iterdir() if x.suffix in VIDEO_FORMAT]
    # rich.print(f"[italic magenta]==>Videos to be split:[/italic magenta]\n{video_list}\n\n")
    rich.print(f"> Video spliting list:")
    rich.print([str(x.resolve()) for x in video_list])

    # if empty dir, stop
    if len(video_list) == 0:
        LOGGER.warning(f"Empty video directory")
        exit(-3)


    # split videos one by one
    for video in tqdm(video_list, f"> Spliting... "):

        save_out = Path(input_dir) / video.stem if not output_dir else Path(output_dir) / video.stem
        video_to_images(source=video,           
                        output=save_out,      # save dir
                        x=x,
                        view=view,
                        flip=flip,
                        img_fmt=img_fmt,
                        verbose=False
                      )

    # successful info
    LOGGER.info(f"Images saved at: {Path(output_dir).resolve()}")



# images -> video
# 所有图片应当是统一尺寸
# todo: 当有错误图片的时候，无法转换，新增错误处理
def images_to_video(source, 
                    size,
                    last4=30, 
                    fps=30, 
                    ):
    
    # load images
    # IMG_FORMAT = [".jpg", ".png", ".jpeg", ".bmp"]  
    images_list = [x.resolve() for x in Path(source).iterdir() if x.suffix in IMG_FORMAT]
    LOGGER.info(f"{len(images_list)} found.")


    # save out dir
    dir_name = 'is2v'
    sub_name = 'video'
    save_dir = increment_path(Path(dir_name) / sub_name, exist_ok=False, sep='')  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)

    video_name = 'imgs_2_video.mp4'
    saveout = save_dir / video_name 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4
    video_writer = cv2.VideoWriter(str(saveout), fourcc, fps, size)

    # key detect
    key = cv2.waitKey(1)

    # loop
    for img in tqdm(images_list):
        frame = cv2.imread(str(img))  
        # resize
        # todo: letter_box()
        # frame = cv2.resize(frame, size)
        frame = letterbox(frame, size)[0]

        for idx in range(int(last4)):
            video_writer.write(frame)


    # done 
    LOGGER.info(f"Done. Video saved at: {saveout.resolve()}") 
    video_writer.release()




# options
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='', help='img dir path(s)')
    parser.add_argument('--output', type=str, default='v2is', help='output dir')
    parser.add_argument('--format', type=str, default='.jpg', help='img suffix')
    parser.add_argument('--frame', type=float, default=30, help='N frame')
    parser.add_argument('--view', action='store_true',help='imshow')
    parser.add_argument('--flip', action='store_true',help='Flip frame')
    parser.add_argument('--fps', default=30, help='Flip frame')
    parser.add_argument('--mode', default='play', help='Flip frame')

    opt = parser.parse_args()
    rich.print(opt, end="\n\n")
    return opt





if __name__ == '__main__':

    opt = parse_opt()

    if opt.mode == 'v2is':
        # test 1
        video_to_images(source=opt.input,                             # 0: cam
                        output=opt.output,      # save dir
                        x=opt.frame,
                        view=opt.view,
                        flip=opt.flip,
                        )   
    elif opt.mode == 'vs2is':
        # test 2
        videos_to_images(
                        input_dir=opt.input,
                        output_dir=opt.output,
                        x=opt.frame,
                        view=opt.view,
                        flip=opt.flip,
                        img_fmt=opt.format
                        )
    elif opt.mode == 'play':
        # # test 3
        play_and_record(source=opt.input, delay=1, flip=False)

    elif opt.mode == 'is2v':
        # test 4
        images_to_video(source=opt.input, 
                        last4=opt.fps, 
                        fps=opt.fps, 
                        size=(2000, 2000),
                        )
