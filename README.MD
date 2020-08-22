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
