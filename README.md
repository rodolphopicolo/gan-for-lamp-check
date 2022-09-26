# Lamp Classification GAN


## check_dataset_py

Checks if all images are the same size. If it would not be truth, we would have to transform then and set all with the same size.
We are look, all images are of shape (480, 640, 3).

To run this file, the images container folder needs to be passaed as argument.


## Issues

### Virtual Environment (venv)

We should consider create a virtual environment (venv) with just what we need.
pip list
pip freeze > requirements.txt
Choose the real requirements
python3 -m venv ~/.venv/gan_lamp
python -m pip install -r requirements.txt
