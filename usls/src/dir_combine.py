#!/usr/bin/env python
# -*- coding:utf-8 -*- 

from tqdm import tqdm 
from pathlib import Path
import argparse
import sys
import rich
import os
import shutil


#--------------------------------------------
#       PYTHON_PATH
#--------------------------------------------
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # abs path => relative
# rich.print(list(FILE.parents))
# rich.print(f"[italic magenta]==>Current Python PATH: {ROOT}")
#--------------------------------------------



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./', help='img dir path(s)')
    parser.add_argument('--output', type=str, default='raw_video/img_combined', help='output dir')
    parser.add_argument('--suffix', nargs='+', type=str, default=[], help=".py', '.jpg', '.txt")
    parser.add_argument('--move', action='store_true', help='copy or move')

    opt = parser.parse_args()
    rich.print(opt, end="\n\n")
    return opt



def dir_combine(input,
				output,
				# multi=False,
				suffix=[],  # 
				move=False,
				):


	saveout_dir = Path(output).resolve()

	# mkdir if now exists OR check if has data in exist dir
	if not saveout_dir.exists():
		rich.print("> Output dir is not exist! Create automaticlly.", end="\n")
		saveout_dir.mkdir()
	else:
		rich.print(f"> Output dir: [u]{saveout_dir}[/u] is exist!", end="\t")
		size = len([x for x in saveout_dir.iterdir()])
		if size > 1:
			rich.print("[bold red]And it has content, Break! Please check!")
			sys.exit(-1)
			# raise StopIteration
		else:
			rich.print("[green]No content in it. Don't worry about it")
		

	# glob
	item_list = []

	if len(suffix) == 0:
		item_list += [x for x in Path(input).glob("**/*") if x.is_file()]

	else:
		for s in suffix:
			s = "**/*" + s
			item_list += [x for x in Path(input).glob(s)]

	# rich.print(f"[italic blue]==>Dir to be combined:[/italic blue]\n{item_list}\n\n")


	# all dirs to be combined
	for d in tqdm(item_list, '> Processing...'):

		# s = str(d).replace('/', '_')
		s = str(d).replace(os.path.sep, '_')
		des_path = saveout_dir.resolve() / s 
		if move:
			shutil.move(str(d.resolve()), str(des_path))
		else:
			shutil.copy(str(d.resolve()), str(des_path))



if __name__ == "__main__":
    opt = parse_opt()
    dir_combine(input=opt.input,
				output=opt.output,
				suffix=opt.suffix,
				move=opt.move)
