#!/usr/bin/env python

import os

from setuptools import setup, find_packages

module_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    setup(
        name='matminer',
        version='0.4.2',
        description='matminer is a library that contains tools for data mining in Materials Science',
        long_description=open(os.path.join(module_dir, 'README.md')).read(),
        url='https://github.com/hackingmaterials/matminer',
        author='Anubhav Jain',
        author_email='anubhavster@gmail.com',
        license='modified BSD',
        packages=find_packages(),
        package_data={'matminer.datasets': ['*.csv'],
                      'matminer.featurizers': ["*.yaml"],
                      'matminer.utils.data_files': ['*.csv', '*.tsv', '*.json', 'magpie_elementdata/*.table']},
        zip_safe=False,
        install_requires=['pymatgen>=2018.8.7', 'tqdm>=4.14.0', 'pandas>=0.20.1',
                          'pymongo>=3.4.0', 'pint>=0.8.1', 'six>=1.10.0',
                          'citrination-client>=4.0.0', 'plotly>=3.2.1',
                          'mdf_forge>=0.6.1', 'scikit-learn>=0.19.0',
                          'sympy>=1.1.1'],
        extras_require={'mpds': ['jmespath>=0.9.3', 'ujson>=1.35', 'httplib2>=0.10.3', 'ase>=3.14.1'],
                        'plot': ['matplotlib>=2.0.0']},
        classifiers=['Programming Language :: Python :: 2.7',
                     'Programming Language :: Python :: 3.6',
                     'Development Status :: 4 - Beta',
                     'Intended Audience :: Science/Research',
                     'Intended Audience :: System Administrators',
                     'Intended Audience :: Information Technology',
                     'Operating System :: OS Independent',
                     'Topic :: Other/Nonlisted Topic',
                     'Topic :: Scientific/Engineering'],
        test_suite='nose.collector',
        tests_require=['nose'],
        scripts=[]
        # scripts=[os.path.join('scripts', f) for f in os.listdir(os.path.join(module_dir, 'scripts'))]
    )
