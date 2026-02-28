"""
Setup script for the text_processing_project.
"""

from setuptools import find_packages, setup

setup(
    name="jupyterproject",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "text-processor = jupyterproject.main:main",
        ],
    },
)
