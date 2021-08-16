#!/usr/bin/env python

import os

from setuptools import setup, find_packages

module_dir = os.path.dirname(os.path.abspath(__file__))

# Requirements
reqs_file = os.path.join(module_dir, "requirements.txt")
with open(reqs_file, "r") as f:
    reqs_raw = f.read()
reqs_list = [r.replace("==", ">=") for r in reqs_raw.split("\n")]

# Optional requirements
extras_file = os.path.join(module_dir, "requirements-optional.txt")
with open(extras_file, "r") as f:
    extras_raw = f.read()
extras_raw = [r for r in extras_raw.split("##") if r.strip() and "#" not in r]
extras_dict = {}
for req in extras_raw:
    items = [i.replace("==", ">=") for i in req.split("\n") if i.strip()]
    dependency_name = items[0].strip()
    dependency_reqs = [i.strip() for i in items[1:] if i.strip()]
    extras_dict[dependency_name] = dependency_reqs
extras_list = [r for d in extras_dict.values() for r in d]

if __name__ == "__main__":
    setup(
        name='matminer',
        version='0.7.4',
        description='matminer is a library that contains tools for data '
                    'mining in Materials Science',
        long_description=open(os.path.join(module_dir, 'README.md')).read(),
        url='https://github.com/hackingmaterials/matminer',
        author='Anubhav Jain',
        author_email='anubhavster@gmail.com',
        license='modified BSD',
        packages=find_packages(),
        include_package_data=True,
        zip_safe=False,
        install_requires=reqs_list,
        extras_require=extras_dict,
        classifiers=[
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Intended Audience :: System Administrators',
            'Intended Audience :: Information Technology',
            'Operating System :: OS Independent',
            'Topic :: Other/Nonlisted Topic',
            'Topic :: Scientific/Engineering'],
        test_suite='matminer',
        tests_require=extras_list,
        scripts=[]
    )
