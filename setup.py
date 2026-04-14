from setuptools import setup

setup(
    name='Doubly-Stochastic-DGP',
    version='1.0',
    author="Hugh Salimbeni",
    author_email="hrs13@ic.ac.uk",
    license="Apache License 2.0",
    packages=["doubly_stochastic_dgp"],
    python_requires=">=3.9,<3.13",
    install_requires=[
        "tensorflow>=2.16,<2.17",
        "tensorflow-probability[tf]>=0.24,<0.25",
        "gpflow>=2.9,<3",
        "numpy>=1.23",
        # GPflow 2.10 imports pkg_resources at module import time.
        "setuptools<81",
    ],
    extras_require={
        "datasets": [
            "openml>=0.14",
            "pandas>=2.0",
            "openpyxl>=3.1",
            "xlrd>=2.0",
        ],
        "demos": [
            "scipy>=1.10",
            "matplotlib>=3.7",
            "scikit-learn>=1.3",
            "openml>=0.14",
            "pandas>=2.0",
            "openpyxl>=3.1",
            "xlrd>=2.0",
        ],
        "test": ["pytest>=8"],
    },
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)
