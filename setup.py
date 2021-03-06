import setuptools

with open("README.md","r") as fh:
	long_description = fh.read()
INSTALL_REQUIRES = (
    ['pandas', 'numpy']
)


setuptools.setup(
    name="raptorarima",
    version="0.0.1",
    author="M V D SATYA SWAROOP",
    author_email="swaroop9ai9@gmail.com",
    description="Implementation of ARIMA model (Auto Regressive Integrated Moving Average)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/swaroop9ai9/raptorarima",
    dependency_links=['https://github.com/swaroop9ai9/raptorarima/blob/master/raptorarima/autoregressive.py','https://github.com/swaroop9ai9/raptorarima/blob/master/raptorarima/differencing.py','https://github.com/swaroop9ai9/raptorarima/blob/master/raptorarima/movingaverage.py','https://github.com/swaroop9ai9/raptorarima/blob/master/raptorarima/plots.py'],
    packages=setuptools.find_packages(),
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
