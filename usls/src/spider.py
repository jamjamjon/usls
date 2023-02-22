from baiduspider import BaiduSpider
from pprint import pprint
from tqdm import tqdm
import urllib
from pathlib import Path
import rich
import sys
import argparse


#--------------------------------------------
#       ADD TO PYTHON_PATH
#--------------------------------------------
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]   # FILE.parent 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # abs path => relative
#--------------------------------------------


from utils import LOGGER, increment_path





# ----------------options ------------------------
def parse_opt():
    parser = argparse.ArgumentParser(description='Open-source image labeling tool')

    # basic directory
    parser.add_argument('--words', default='', type=str, nargs="+", help='multi word use space to sep.', required=True)
 
    opt = parser.parse_args()
    rich.print(f"[bold white]=>({FILE.name})[/bold white] : [bold]{opt}\n")
    return opt



# main
def spider_img_baidu(words):

	rich.print(f"[bold magenta]------baidu spider for images--------")

	# print(opt.keywords)
	keywords = " ".join(words)

	# get info
	response = BaiduSpider().search_pic(keywords)
	image_num = response.total
	page_num = response.pages
	LOGGER.info(f"Image_num: {image_num}  |  Page_num: {page_num}")


	# interact with user about page start & end.
	input_c = input("=> Input two numbers, page_start & page_end, use space to seperate: ")
	c = input_c.strip().split()
	if len(c) != 2:
		LOGGER.error(f"Wrong input! Should has two numbers!")
		exit()

	page_start = int(c[0])
	page_end = int(c[1])
	if page_start > page_end:
		LOGGER.error(f"Error: page start > page end!")
		exit()

	LOGGER.info(f"It will spider from page {page_start} to page {page_end}")


	# saveout dir
	dir_name = "baidu-img-spider"
	project = "_".join(words)
	save_dir = increment_path(Path(dir_name)/project, exist_ok=False, sep='')  # increment run

	# loop
	for n in tqdm(range(page_start, page_end)):

		print(f"[bold green]==> {n}")

		# spider
		res = BaiduSpider().search_pic(keywords, n)

		# make dir for every page
		(save_dir / str(n)).mkdir(parents=True, exist_ok=True)

		# loop to save
		plain = res.__dict__['plain']
		for idx, item in tqdm(enumerate(plain)):
			url = item['url']

			try:
				saveout = Path(save_dir / str(n) / (str(idx) + '.jpg')) 
				urllib.request.urlretrieve(str(url), filename=str(saveout))
			# except [urllib.error.ContentTooShortError, urllib.error.HTTPError]:
			except Exception as error:
				print("[error]:", error)

				# continue
			# finally:
				# continue


# ---------------------------------------------------
#   main
#--------------------------------------------------
if __name__ == '__main__':
	# options
	opt = parse_opt()
	spider_img_baidu(opt.words)


	# res = BaiduSpider().search_pic(query='安全带')
	# plain = res.__dict__['plain']
	# pprint(type(plain))
	#pprint(plain[-1])
	# pprint(res.total)
	# pprint(res.pages)
	# pprint(len(res.plain))

	# for idx, item in tqdm(enumerate(plain)):
	# 	# pprint(item['url'])
	# 	# print('-----')

	# 	url = item['url']
	# 	saveout_dir = "test"
	# 	saveout = Path(saveout_dir) / (str(idx)+'.jpg')
	# 	urllib.request.urlretrieve(str(url) , filename=str(saveout))

