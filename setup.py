from io import open

from setuptools import setup

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

# Note: removed all identity information from this file

setup(name='biologistguidesbi',
      version='0.0.1',
      description='Package for RSNL algorithm for \
                   simulation-based inference',
      license='GPL',
      packages=['BVCBM', 'FUCCI'],
      zip_safe=False,
      python_requires='>=3.7',
      install_requires=requirements
      )
