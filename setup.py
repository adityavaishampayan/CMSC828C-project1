from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')
    
requirements = ["ipython>=6", "nbformat>=4", "nbconvert>=5", "requests>=2"]

setup(
    name="CMSC_828C_Project1",
    version="0.0.1",
    description = 'Python implementation of fundamenta Machine Learning algorithms on Fashion-MNIST dataset',
    url='https://github.com/adityavaishampayan/CMSC828C-project1.git'
    license='LICENSE.txt',
    packages = find_packages(),
    author="Aditya Vaishampayan",
    author_email="adityav@terpmail.umd.edu",
    long_description=readme,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    setup_requires=['numpy>=1.15', 'scipy>=0.17','matplotlib>=3.0.0','scikit-learn=0.20.2']
    
)
