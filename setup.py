#!/usr/bin/env python

import os

from setuptools import find_packages, setup

module_dir = os.path.dirname(os.path.abspath(__file__))

extras_require = {
    "mpds": ["ujson", "jmespath", "httplib2", "ase", "jsonschema"],
    "dscribe": ["dscribe~=2.1"],
    "mdfforge": ["mdf-forge"],
    "aflow": ["aflow"],
    "citrine": ["citrination-client"],
    "dev": [
        "pytest",
        "pytest-cov",
        "pytest-timeout",
        "coverage",
        "coveralls",
        "flake8",
        "black",
        "pylint",
        "sphinx",
    ],
}
tests_require = [r for v in extras_require.values() for r in v]

extras_require["tests"] = tests_require

if __name__ == "__main__":
    setup(
        long_description=open(os.path.join(module_dir, "README.md")).read(),
        long_description_content_type="text/markdown",
        packages=find_packages(),
        include_package_data=True,
        zip_safe=False,
        install_requires=[
            "numpy >= 1.23",
            "requests ~= 2.31",
            "pandas >= 1.5, < 3",
            "tqdm ~= 4.66",
            "pymongo ~= 4.5",
            "scikit_learn ~= 1.3",
            "sympy ~= 1.11",
            "monty >= 2023",
            "pymatgen >= 2023",
        ],
        extras_require=extras_require,
        test_suite="matminer",
        tests_require=tests_require,
        scripts=[],
    )
