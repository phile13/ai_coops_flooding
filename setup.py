# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["requests>=2"]

setup(
    name="ai_coops",
    version="0.0.14",
    author="Erick DiFiore",
    author_email="47676503+phile13@users.noreply.github.com",
    description="A Package for projects with NOAA CO-OPS",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/phile13/ai_coops",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
