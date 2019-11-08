from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["ipython>=6", "nbformat>=4", "nbconvert>=5", "requests>=2"]

setup(
    name="CMSC_828C_Project1",
    version="0.0.1",
    author="Aditya Vaishampayan",
    author_email="adityav@terpmail.umd.edu",
    description="A project that contains bayes classifier implementation ",
    long_description=readme,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    license='LICENSE.txt',
)