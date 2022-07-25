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
        'pint',
    ],
    version_config=True,
    setup_requires=['setuptools-git-versioning'],
)
