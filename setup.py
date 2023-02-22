import setuptools
import re
from pathlib import Path
import pkg_resources as pkg
import rich

FILE = Path(__file__).resolve()
PARENT = FILE.parent 
README = (PARENT / 'README.md').read_text(encoding='utf-8')
REQUIREMENTS = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements((PARENT / 'requirements.txt').read_text())]
PKG_NAME = 'usls'


def get_version():
    file = PARENT / PKG_NAME / '__init__.py'
    return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(encoding='utf-8'), re.M)[1]


setuptools.setup(
	name=PKG_NAME,
	version=get_version(),
	author='jamjamjon',
	python_requires='>=3.7',
	description='Useless CV toolkits',
	packages=setuptools.find_packages(),
	install_requires=REQUIREMENTS,
    include_package_data=True,
    license='GPL-3.0',
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/jamjamjon/usls",
	entry_points={
		'console_scripts': [
			'usls = usls:cli'
		]
	}
)