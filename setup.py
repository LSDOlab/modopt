from setuptools import setup, find_packages

setup(
    name='modopt',
    packages=find_packages(),
    #packages=['modopt'],
    version='0.1',
    install_requires=[
        'numpy',
        'scipy',
        'pint',
        'pandas',
        'osqp',
        'dill',
    ],
    version_config=True,
    setup_requires=['setuptools-git-versioning'],
)
