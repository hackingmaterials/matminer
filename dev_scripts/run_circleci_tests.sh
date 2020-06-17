python3 -m venv test_env
. test_env/bin/activate
pip install --quiet --upgrade pip
pip install --quiet --upgrade setuptools
cd ~/matminer
pip install --quiet -r requirements-optional.txt
pip install --quiet -r requirements.txt
pip install --quiet -e .
python setup.py test