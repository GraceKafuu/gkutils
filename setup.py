import setuptools
import os
import re
from setuptools import find_packages
from setuptools import setup

"""
https://www.mssqltips.com/sqlservertip/6802/create-wheel-file-python-package-distribute-custom-code/
https://www.realpythonproject.com/how-to-create-a-wheel-file-for-your-python-package-and-import-it-in-another-project/
"""


def get_version():
    filename = "gkutils/__init__.py"
    with open(filename) as f:
        match = re.search(
            r"""^__version__ = ['"]([^'"]*)['"]""", f.read(), re.M
        )
    if not match:
        raise RuntimeError("{} doesn't contain __version__".format(filename))
    version = match.groups()[0]
    return version


def get_install_requires():
    install_requires = [
        "numpy",
        "Pillow",
        "opencv-python",
        "scikit-image",
        "scipy",
        "pypyodbc",
        "imgaug",
        "argparse",
        "open3d",
        "lxml",
    ]

    return install_requires


def get_long_description():
    with open("README.md") as f:
        long_description = f.read()
    
    return long_description


def main():
    version = get_version()

    setup(
        name="gkutils",
        version=version,
        author="GraceKafuu",
        author_email="gracekafuu@gmail.com",
        description="GraceKafuu useful utils",
        long_description=get_long_description(),
        long_description_content_type="text/markdown",

        install_requires=get_install_requires(),
        # keywords="Image Annotation, Machine Learning",
        # classifiers=[
        #     "Development Status :: 5 - Production/Stable",
        #     "Intended Audience :: Developers",
        #     "Intended Audience :: Science/Research",
        #     "Natural Language :: English",
        #     "Operating System :: OS Independent",
        #     "Programming Language :: Python",
        #     "Programming Language :: Python :: 3.5",
        #     "Programming Language :: Python :: 3.6",
        #     "Programming Language :: Python :: 3.7",
        #     "Programming Language :: Python :: 3.8",
        #     "Programming Language :: Python :: 3.9",
        #     "Programming Language :: Python :: 3 :: Only",
        # ],
        packages=find_packages(),
        classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],

        # package_data={"labelme": ["icons/*", "config/*.yaml"]},
        entry_points={
            "console_scripts": [
                "gkutils=gkutils.__main__:main",
            ],
        },
    )


if __name__ == "__main__":
    main()
