from setuptools import setup, find_packages

import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()
    
def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")
    
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='modopt',
    version=get_version('modopt/__init__.py'),
    author='Anugrah',
    author_email='ajoshy@ucsd.edu',
    license='LGPLv3+',
    keywords='design optimization algorithm optimizer library',
    url='https://github.com/LSDOlab/modopt',
    download_url='https://pypi.python.org/pypi/modopt-lib',
    description='A modular development environment and library for optimization algorithms',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    #packages=['modopt'],
    python_requires='>=3.7',
    platforms=['any'],
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'array_manager @ git+https://github.com/anugrahjo/array_manager.git@modopt',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        # 'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Natural Language :: English',
        'Topic :: Education',
        'Topic :: Education :: Computer Aided Instruction (CAI)',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
    ],
    version_config=True,
    setup_requires=['setuptools-git-versioning'],
)
