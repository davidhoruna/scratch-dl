from setuptools import setup, find_packages

setup(
    name="vision-models",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
)
