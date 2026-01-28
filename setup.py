from setuptools import setup, find_packages

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
