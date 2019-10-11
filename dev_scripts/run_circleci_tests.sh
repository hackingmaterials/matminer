export CI=circle
python3 -m venv test_env
. test_env/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools
cd ~/matminer
pip install -r requirements-optional.txt
pip install -r requirements.txt
pip install -e .
python setup.py test