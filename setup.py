"""
Setup script for the text_processing_project.
"""

from setuptools import find_packages, setup

setup(
    name="text_processing_project",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "text-processor = src.main:main",
        ],
    },
)
