# Raptor ARIMA Model Python API

* This API Supports to model time series data using box-jenkins model.

# Python Installation:
* Command for installation: 'pip3 install raptorarima'

## Usage

The ARIMA Model is an implementation from scratch which using MA, AR and differencing to model the Data.
``` python
Define the p,d,q parameters.
q = 1
d = 0
p = 3

m = ARIMA(q, d, p)
pred = m.fit_predict(x)
```


## Installation

``` bash
git clone https://github.com/swaroop9ai9/Fintech/raptor_arima_api.git
cd raptor arima
python setup.py install
```

### Requirements

- pandas
- numpy
- random

### Note

Make sure to use Time series Data Values.

### Steps to create pip package
1) Prepare directory with setup.py, packagename(apiname(__init__.py[with main code]),other py files which supports the __init__.py file.)
2) Upload the package to Git
3) cd packagename
4) chmod +x apiname #makes it an executable
5) python3 setup.py bdist_wheel
  [
  * build: build package information.
  * dist: Contains your .whl file. A WHL file is a package saved in the Wheel format, which is the standard built-package format used for Python distributions. You    can directly install a .whl file using pip install some_package.whl on your system
  * project.egg.info: An egg package contains compiled bytecode, package information, dependency links, and captures the info used by the setup.py test command when running tests.]

6) [Optional] to check install on local machine 
  python -m pip install dist/packagename-0.1-py3-none-any.whl
  
7) python3 -m twine upload dist/* [in order to upload the package] (Will ask your username and password in pypi)

8) Login to pypi account to view the uploaded package

