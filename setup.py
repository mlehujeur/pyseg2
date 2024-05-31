# import os
import setuptools

setuptools.setup(
    name='pyseg2',
    version="1.4.2",
    license='GPL3',
    author="Maximilien Lehujeur",
    author_email="maximilien.lehujeur@univ-eiffel.fr",
    decription='Standalone package to read/write seg2 files in python',
    long_description='Standalone package to read/write seg2 files in python',  
    packages=setuptools.find_packages(),
    install_requires=['numpy'],
    python_requires=">=3.2",  # because of datetime.timezone
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        ],
    )
