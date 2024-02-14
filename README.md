# Lamp Classification GAN


## check_dataset_py

Checks if all images are the same size. If it would not be truth, we would have to transform then and set all with the same size.
We are luck, all images are of shape (480, 640, 3).

To run this file, the images container folder needs to be passed as argument.


## Issues

### Virtual Environment (venv)

We should consider create a virtual environment (venv) with just what we need.
pip list
pip freeze > requirements.txt
python3 -m pip freeze > requirements.txt
Choose the real requirements
python3 -m venv ~/.venv/gan_lamp
python -m pip install -r requirements.txt



I have decided to work over config 9.

python ganlampid.py --config=9 --label=1
python ganlampid.py --config=9 --label=2
python ganlampid.py --config=9 --label=3
python ganlampid.py --config=9 --label=4
python ganlampid.py --config=9 --label=5
python ganlampid.py --config=9 --label=6
python ganlampid.py --config=9 --label=7
python ganlampid.py --config=9 --label=8
python ganlampid.py --config=9 --label=9
python ganlampid.py --config=9 --label=10


python ganlampid.py --config=9 --label=1 --mode=gen
python ganlampid.py --config=9 --label=2 --mode=gen
python ganlampid.py --config=9 --label=3 --mode=gen
python ganlampid.py --config=9 --label=4 --mode=gen
python ganlampid.py --config=9 --label=5 --mode=gen
python ganlampid.py --config=9 --label=6 --mode=gen
python ganlampid.py --config=9 --label=7 --mode=gen
python ganlampid.py --config=9 --label=8 --mode=gen
python ganlampid.py --config=9 --label=9 --mode=gen
python ganlampid.py --config=9 --label=10 --mode=gen
