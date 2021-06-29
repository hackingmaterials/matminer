# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import os
import json
import webbrowser
import requests
from invoke import task
from matminer import __version__
from monty.os import cd


"""
Deployment file to facilitate releases.
"""

__author__ = "Shyue Ping Ong, Anubhav Jain"
__email__ = "ongsp@ucsd.edu"
__date__ = "Sep 1, 2014"


@task
def make_doc(ctx):
    with cd("docs_rst"):
        ctx.run("python featurizer_summary.py > featurizer_summary.rst")
        ctx.run("python dataset_summary.py > dataset_summary.rst")
        ctx.run("sphinx-apidoc -o . -f ../matminer")
        ctx.run("make html")
        ctx.run("cp _static/* ../docs/html/_static")

    with cd("docs"):
        ctx.run("cp -r html/* .")
        ctx.run("rm -r html")
        ctx.run("rm -r doctrees")

        # Avoid the use of jekyll so that _dir works as intended.
        ctx.run("touch .nojekyll")


@task
def update_doc(ctx):
    make_doc(ctx)
    with cd("docs"):
        ctx.run("git add .")
        ctx.run("git commit -a -m \"Update to v{}\"".format(__version__))
        ctx.run("git push")

@task
def publish(ctx):
    ctx.run("twine upload dist/* --verbose")


@task
def release_github(ctx):
    payload = {
        "tag_name": "v" + __version__,
        "target_commitish": "master",
        "name": "v" + __version__,
        "body": "",
        "draft": False,
        "prerelease": False
    }
    # For this to work properly, you need to go to your Github profile, generate
    # a "Personal access token". Then do export GITHUB_RELEASES_TOKEN="xyz1234"
    # (or add it to your bash_profile).
    response = requests.post(
        "https://api.github.com/repos/hackingmaterials/matminer/releases",
        data=json.dumps(payload),
        headers={"Authorization": "token " + os.environ["GITHUB_RELEASES_TOKEN"]})
    print(response.text)


@task
def release(ctx, test=False):
    if test:
        ctx.run("python setup.py test")
    ctx.run("rm -r build dist", warn=True)
    ctx.run("python setup.py sdist bdist_wheel")

    publish(ctx)
    update_doc(ctx)
    release_github(ctx)


@task
def open_doc(ctx):
    pth = os.path.abspath("docs/index.html")
    webbrowser.open("file://" + pth)
