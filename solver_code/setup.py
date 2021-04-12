from setuptools import setup, find_packages


setup(
    name="solver_lasso",
    install_requires=[
        "libsvmdata",
        "numpy>=1.12",
        "numba",
        "seaborn>=0.7",
        "joblib",
        "scipy>=0.18.0",
        "matplotlib>=2.0.0",
        "scikit-learn>=0.23",
        "pandas",
        "ipython",
    ],
    packages=find_packages(),
)
