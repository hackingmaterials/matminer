#!/usr/bin/env python

import os

from setuptools import find_packages, setup


def local_version(version):
    # https://github.com/pypa/setuptools_scm/issues/342
    return ""


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
        name="matminer",
        use_scm_version={
            "root": ".",
            "relative_to": __file__,
            "local_scheme": local_version,
        },
        setup_requires=["setuptools_scm"],
        description="matminer is a library that contains tools for data mining in Materials Science",
        long_description=open(os.path.join(module_dir, "README.md")).read(),
        url="https://github.com/hackingmaterials/matminer",
        long_description_content_type="text/markdown",
        author="Anubhav Jain",
        author_email="anubhavster@gmail.com",
        license="modified BSD",
        packages=find_packages(),
        include_package_data=True,
        zip_safe=False,
        install_requires=[
            "numpy>=1.20.1",
            "requests",
            "pandas",
            "tqdm",
            "pymongo",
            "future",
            "scikit_learn",
            "sympy",
            "monty",
            "pymatgen",
        ],
        extras_require=extras_require,
        classifiers=[
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "Intended Audience :: System Administrators",
            "Intended Audience :: Information Technology",
            "Operating System :: OS Independent",
            "Topic :: Other/Nonlisted Topic",
            "Topic :: Scientific/Engineering",
        ],
        test_suite="matminer",
        tests_require=tests_require,
        scripts=[],
    )
