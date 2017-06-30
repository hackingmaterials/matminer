#!/usr/bin/env python

from setuptools import setup, find_packages
import os
import multiprocessing, logging  # AJ: for some reason this is needed to not have "python setup.py test" freak out

module_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    setup(
        name='matminer',
        version='0.1.0',
        description='matminer is a library that contains tools for data mining in Materials Science',
        long_description=open(os.path.join(module_dir, 'README.rst')).read(),
        url='https://github.com/hackingmaterials/matminer',
        author='Anubhav Jain, Saurabh Bajaj',
        author_email='anubhavster@gmail.com, saurabhbajaj2@gmail.com',
        license='modified BSD',
        packages=find_packages(),
        package_data={},
        zip_safe=False,
        install_requires=['pymatgen>=4.0', 'tqdm>=3.7.1', 'pandas>=0.17.1',
                          'unittest2==1.1.0', "pymongo==3.2.2", 'pint'],
        extras_require={'citrine':['citrination-client>=1.3.1']},
        classifiers=['Programming Language :: Python :: 2.7',
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
        #scripts=[os.path.join('scripts', f) for f in os.listdir(os.path.join(module_dir, 'scripts'))]
    )
