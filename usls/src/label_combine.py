from pathlib import Path
import shutil
import os
from tqdm import tqdm
import argparse
import rich
from rich.progress import track as ProgressBar

console = rich.console.Console()


def combine_AB(A, B, ALL):
		
	A_list = [x for x in Path(A).iterdir() if x.suffix == ".txt"]
	B_list = [x for x in Path(B).iterdir() if x.suffix == ".txt"]
	# A_list = [x for x in Path(A).iterdir() if x.suffix == ".txt" and x.stat().st_size != 0]
	# B_list = [x for x in Path(B).iterdir() if x.suffix == ".txt" and x.stat().st_size != 0]


	# A update B -> ALL
	for a in A_list:
		if Path(B) / a.name not in B_list:
			shutil.copy(str(a), str(Path(ALL)))
		else:
			saveout = Path(ALL) / a.name
			with open(saveout, "a") as f_output:
				f_A = open(a, "r").read()   # read a
				f_B = open(Path(B) / a.name, "r").read()  # read b

				# deal with label item has extra line
				# remove all last \n line
				while f_A[-1] == '\n':
					f_A = f_A[:-1]

				while f_B[-1] == '\n':
					f_B = f_B[:-1]


				f_output.write(f_A + '\n' + f_B)  # write to output
				# f_output.write(f_A + f_B)  # write to output


	# B -> ALL
	for b in B_list:
		if Path(A) / b.name not in A_list:
			shutil.copy(str(b), str(Path(ALL)))
		else:
			continue




def combine_labels(input_dir, output_dir):
	# print('=================>')
	# print(f'input_dir =================> {input_dir}')
	# print(f'output_dir =================> {output_dir}')

	# make saveout dir
	if not Path(output_dir).exists():
		Path(output_dir).mkdir()
	else:
		shutil.rmtree(output_dir)
		Path(output_dir).mkdir()
	console.log(f'> Build output dir.')
	# print('=================>')


	temp_dir = 'temp_dir'
	dir_list = [x for x in Path(input_dir).iterdir() if x.is_dir()]
	console.log(f'> Has {len(dir_list)} dirs.')

	console.log(f'> Start combining.')
	for x in ProgressBar(dir_list, description="Combining..."):
		if Path(temp_dir).exists():
			shutil.rmtree(temp_dir)
		shutil.copytree(output_dir, temp_dir)  # make a copy
		shutil.rmtree(output_dir)  # clean output dir	
		Path(output_dir).mkdir()  # build output dir again
		combine_AB(x, temp_dir, output_dir)
	console.log(f'> Done combining.')

	# remove temp dir
	if Path(temp_dir).exists():
		shutil.rmtree(temp_dir)
		console.log(f'> `temp_dir` removed.')

	# output dir info
	console.log(f"> Has {len([x for x in Path(output_dir).iterdir() if x.suffix == '.txt'])} labels.")
	console.log(f"> Output: {Path(output_dir).resolve()}")



def main(opt):


	# combine_AB(opt.A, opt.B, opt.output)  # only two dirs
	combine_labels(opt.input, opt.output)

	

#--------------------------------------------------
def parse_opt():
	parser = argparse.ArgumentParser(description='label-txt-combine')
	parser.add_argument('--A', default='', type=str, help='label dir A')
	parser.add_argument('--B', default='', type=str, help='label dir B')
	parser.add_argument('--input', default='', type=str, help='labels dir')
	parser.add_argument('--output', default='output', type=str, help='Path to output directory')
	opt = parser.parse_args()
	console.log(f"[bold]{opt}\n")
	return opt


#--------------------------------------------------
if __name__ == '__main__':
	# options
	opt = parse_opt()
	main(opt)



















