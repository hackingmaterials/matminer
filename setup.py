#!/usr/bin/env python

import os

from setuptools import find_packages, setup

module_dir = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    setup(
        long_description=open(os.path.join(module_dir, "README.md")).read(),
        long_description_content_type="text/markdown",
        packages=find_packages(),
        include_package_data=True,
        zip_safe=False,
        scripts=[],
    )
