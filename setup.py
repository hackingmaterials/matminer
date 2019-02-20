#!/usr/bin/env python

import os

from setuptools import setup, find_packages

module_dir = os.path.dirname(os.path.abspath(__file__))
reqs_raw = open(os.path.join(module_dir, "requirements.txt")).read()
reqs_list = [r.replace("==", ">=") for r in reqs_raw.split("\n")]

extras_dict = {'mpds': ['jmespath>=0.9.3', 'ujson>=1.35', 'httplib2>=0.10.3',
                        'ase>=3.14.1'],
               'mdf': ['mdf_forge==0.6.1'],
               'aflow': ['aflow==0.0.9'],
               'citrine': ['citrination-client==4.0.1'],
               'dscribe': ['dscribe==0.1.8']}
extras_list = []
for val in extras_dict.values():
    extras_list.extend(val)


if __name__ == "__main__":
    setup(
        name='matminer',
        version='0.5.3',
        description='matminer is a library that contains tools for data '
                    'mining in Materials Science',
        long_description=open(os.path.join(module_dir, 'README.md')).read(),
        url='https://github.com/hackingmaterials/matminer',
        author='Anubhav Jain',
        author_email='anubhavster@gmail.com',
        license='modified BSD',
        packages=find_packages(),
        package_data={
            'matminer.datasets': ['*.json'],
            'matminer.featurizers': ["*.yaml"],
            'matminer.utils.data_files': ['*.csv', '*.tsv', '*.json',
                                          'magpie_elementdata/*.table',
                                          'jarvis/*.json']},
        zip_safe=False,
        install_requires=reqs_list,
        extras_require=extras_dict,
        classifiers=['Programming Language :: Python :: 2.7',
                     'Programming Language :: Python :: 3.6',
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
        # scripts=[os.path.join('scripts', f) for f in os.listdir(os.path.join(module_dir, 'scripts'))]
)