import os
from pathlib import Path

from setuptools import setup, find_packages
import warnings
warnings.filterwarnings('ignore')

# Metadata
NAME='loan-prediction-model'
DESCRIPTION='ML model package for prediction of Loan'
URL='https://github.com/Harsh-Agarwals/Deep-Learning-Mastery/tree/main/MLOps/Packaging-ML-Model'
EMAIL='harshag.code@gmail.com'
AUTHOR='Harsh Agarwal'
REQUIRES_PYTHON='>=3.7.0'

# getting the present working directory
pwd = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(pwd, 'requirements.txt')) as reqmt:
    requirements = reqmt.readlines()
    
# requirements = [req[:-1] for req in requirements]
requirements = [req.strip() for req in requirements][1:]

ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / 'prediction_model'
about = {}
with open(PACKAGE_DIR / 'VERSION') as vers:
    vx = vers.read().strip()
    about['__version__'] = vx
    

setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(
        exclude=('tests',)
    ),
    package_data={'prediction_model': ['VERSION']},
    install_requires=requirements,
    extras_require={},
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ]
)