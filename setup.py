# import os
import setuptools

setuptools.setup(
    name='pyseg2',
    version="1.2",
    packages=setuptools.find_packages(),
    install_requires=['numpy' ],
    python_requires=">=3.2",  # because of datetime.timezone, see
    )
