from setuptools import setup, find_packages

setup(
    name="trading_bot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "tensorflow>=2.7.0",
        "yfinance>=0.1.63",
        "scikit-learn>=0.24.2",
        "ta>=0.7.0",
    ],
    python_requires=">=3.9",
)
