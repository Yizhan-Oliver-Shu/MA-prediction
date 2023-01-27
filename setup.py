from setuptools import setup, find_packages

requirements = [
      'pandas',
      'numpy',
      'wrds',
      'pandas_market_calendars',
      'tqdm',
      'sklearn',
      'statsmodels'
]

setup(name="MA_prediction",
      version="0.1",
      python_requires='>=3.5',
      # packages=find_packages(),
      install_requires=requirements,
      )