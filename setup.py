from setuptools import setup

setup(
    name='modopt',
    packages=[
        'modopt',
    ],
    version='0.1',
    install_requires=[
        'numpy',
        'scipy',
        'dash==1.2.0',
        'dash-daq==0.1.0',
        'pint',
    ],
    version_config=True,
    setup_requires=['setuptools-git-versioning'],
)