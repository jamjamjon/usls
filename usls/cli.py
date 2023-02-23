import sys
import rich
import re
from omegaconf import OmegaConf, DictConfig

from usls import __version__
from usls.run import run
from usls.src.utils import LOGGER



CLI_MSG = '''
  Uselesss Usage: 
	-----------------------------
	Special options:
	-----------------------------
	-h, -H, --help     	 package usage
	-v, -V, --version  	 package version

	-----------------------------
	Format:
	-----------------------------
	> fx task=xxx sth=sth 

	-----------------------------
	common toolkits
	-----------------------------

	-----------------------------
	Detection Labelling 
	-----------------------------
	> usls task=inspect img_dir=sth {label_dir=sth} classes=sth,sth,sth {window_width=800 window_height=600 mv_dir=moved_dir wrong_img_dir=sth}

	-----------------------------
	Classification Labelling 
	-----------------------------
	> usls task=classify img_dir=sth {label_dir=sth} classes=sth,sth,sth {window_width=800 window_height=600 mv_dir=moved_dir wrong_img_dir=sth}

	-----------------------------
	Image & Label dir info 
	-----------------------------
	> usls task=info img_dir=sth {label_dir=sth}

	-----------------------------
	Clean-up Image & Label Dir 
	-----------------------------
	> usls task=clean img_dir=sth {label_dir=sth mv_dir=sth clean_empty=True}

	-----------------------------
	Combine labels
	-----------------------------
	> usls task=label_combine input_dir=sth {output_dir=output-label-combine}

	-----------------------------
	Combine dirs
	-----------------------------
	> usls task=dir_combine input_dir=sth {output_dir=output-dir-combine suffix=[] move=False}

	-----------------------------
	Video -> Images
	-----------------------------
	> usls task=v2is source=sth {output_dir=v2is frame=20 view=False flip=False fmt_img=.jpg}
	
	-----------------------------
	Videos -> Images
	-----------------------------
	> usls task=vs2is input_dir=sth {output_dir=vs2is frame=20 view=False flip=False fmt_img=.jpg}

	-----------------------------
	Images -> Video
	-----------------------------
	> usls task=is2v input_dir=sth {output_dir=vs2is fps=30 last4=60 video_size=640}

	-----------------------------
	Play videos/streams & Record(Press `r/R` to record when playing.)
	-----------------------------
	> usls task=play source=sth {delay=1 flip=False}

	-----------------------------
	Spider
	-----------------------------
	> usls task=spider words=sth

	-----------------------------
	De-duplicate
	-----------------------------
	> usls task=deduplicate input_dir=sth {mv_dir=sth}

	-----------------------------
	Label Class Modify
	-----------------------------
	> usls task=class_modify input_dir=sth to=7


'''


SPECIALS = {
	'--help': lambda: rich.print(CLI_MSG),
	'-H': lambda: rich.print(CLI_MSG),
	'-h': lambda: rich.print(CLI_MSG),
	'--version': lambda: rich.print(f"> uselesss version: {__version__}"),
	'-V': lambda: rich.print(f"> uselesss version: {__version__}"),
	'-v': lambda: rich.print(f"> uselesss version: {__version__}"),
}


# support task list
TASKS = (
	'info', 
	'inspect', 
	'dir_combine', 
	'label_combine',
	'spider',
	'clean', 
	'cleanup',
	'v2is',
	'vs2is',
	'play',
	'is2v',
	'classify',
	'deduplicate',
	'class_modify',
)


def cli() -> None:
	args = sys.argv[1:]

	if not args:
		rich.print(f">\n{CLI_MSG}")
		return 

	cmd = {'task': 'untitled'} 	# default  TODO: msg

	# argv[1:]
	for idx, x in enumerate(args):

		# special cmd with `-` or `--`
		if x.startswith('-'):
			if x in SPECIALS.keys():
				SPECIALS[x]()
				return

		else:	# must use '=' to specify args
			if '=' in x:
				try:
					k, v = x.strip().split('=')
					if k == 'task':
						assert v in TASKS, f"> Error: Task support: {TASKS} for now!"
					cmd.update({k: v})
				except Exception as E:
					rich.print(f'{E}')
					sys.exit(1) 

			elif x.lower() in TASKS and idx == 0:
				cmd.update({'task': x.lower()})
			else:
				if idx == 0:
					rich.print(f"> Warning: `{x}` is not in supported TASKS: {TASKS}")
				else:
					rich.print(f"> Warning: `{x}` is not supported, ignored by deault! You can use `sth={x}`")

	rich.print(f"> args: {cmd}")


	# check if has task
	if cmd.get('task') == 'untitled':
		LOGGER.error(f"> `task` not specified!")
	else:
		conf = OmegaConf.create(cmd) 
		run(conf) 	# run




if __name__ == '__main__':
	cli()
