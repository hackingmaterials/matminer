#!/usr/bin/env python

import os
import ruamel.yaml as yaml

from setuptools import setup, find_packages


PWD = os.path.dirname(os.path.abspath(__file__))


def parse_requirements(requirement_type):
    """Parses installation, optional, and testing requirements from YAML files.

    The "optional" requirements are required for testing. The installation
    requirements are stored in "requirements.yaml" and the optional/testing
    requirements are stored in "requirements-optional.yaml".

    Args:
        requirement_type: (str) Either "install", "optional", or "testing".

    Returns: (list or dict) Either a list of dependencies (install/testing) or
        a dictionary of dependencies (optional).
    """
    if requirement_type == 'install':

        with open(os.path.join(PWD, "requirements.yaml")) as file:
            requirements = yaml.load(file)
            return [r.replace("==", ">=") for r in requirements]

    else:

        with open(os.path.join(PWD, "requirements-optional.yaml")) as file:
            optional_requirements = yaml.load(file)

            if requirement_type == 'optional':

                for key, val in optional_requirements.items():
                    optional_requirements[key] = [
                        r.replace("==", ">=") for r in val]
                return optional_requirements

            elif requirement_type == 'testing':

                testing_requires = []
                for val in optional_requirements.values():
                    testing_requires.extend(
                        [r.replace("==", ">=") for r in val])
                return testing_requires


if __name__ == "__main__":
    setup(
        name='matminer',
        version='0.5.0',
        description='matminer is a library that contains tools for data mining'
                    ' in Materials Science',
        long_description=open(os.path.join(PWD, 'README.md')).read(),
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
        install_requires=parse_requirements('install'),
        extras_require=parse_requirements('optional'),
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
        tests_require=parse_requirements('testing').append('nose'),
        scripts=[]
        # scripts=[os.path.join('scripts', f) for f in
        #          os.listdir(os.path.join(module_dir, 'scripts'))]
    )
