import os 
from pathlib import Path
from tqdm import tqdm
import argparse
import shutil
import rich



# ----------------options ------------------------
def parse_opt():
    parser = argparse.ArgumentParser(description='Open-source image labeling tool')

    # basic directory
    parser.add_argument('--input', default='', type=str, help='Path to input directory')
    parser.add_argument('--to', default='', type=str, help='Path to input directory')

    opt = parser.parse_args()
    rich.print(f"[bold]{opt}\n")
    return opt



# main 
# def label_modify(opt):
def class_modify(input_dir, to):

	# 0. create transfer dir
	tmp_dir = Path(".temp_to_save_modified_labels")	
	if  tmp_dir.exists():
		shutil.rmtree(str(tmp_dir.resolve()))
	else:
		tmp_dir.mkdir()

	# 1. modifing labels
	# rich.print(f"[green]> modifing labels...")
	label_list = [x for x in Path(input_dir).iterdir() if x.suffix == '.txt']
	for file in tqdm(label_list, '> Generating...'):
		# p_r = os.path.join(label_dir, file)
		# p_w = os.path.join(des_label_dir, file)
		p_r = Path(input_dir).resolve() / file.name
		p_w = tmp_dir.resolve() / file.name

		f_w = open(p_w, "w")
		with open(p_r, "r") as f:

			for line in f:
				l = line.split()
				l[0] = to
				new_line = ' '.join(l)
				# print(new_line)
				f_w.write(new_line + '\n')

		f_w.close()


	# 2.delete all old labels
	# rich.print(f"[green]> delete all old labels")
	old_label_list = [x for x in Path(input_dir).iterdir() if x.suffix == '.txt']
	for label in tqdm(old_label_list, '> Deleting...'):
		p = Path(label).resolve()	
		if p.exists():
			p.unlink()


	# 3.move new labels into ori dir
	# rich.print(f"[green]> move new labels into ori dir")
	new_label_list = [x for x in tmp_dir.iterdir() if x.suffix == '.txt']
	for label in tqdm(new_label_list, '> Modifing...'):
		shutil.move(str(label.resolve()), str(input_dir))

	# 4. rm tmp_dir
	shutil.rmtree(tmp_dir)


	rich.print(f"> Label Output: [u]{Path(input_dir).resolve()}")




# ---------------------------------------------------
#   main
#--------------------------------------------------
if __name__ == '__main__':
	# options
	opt = parse_opt()
	label_modify(opt)


