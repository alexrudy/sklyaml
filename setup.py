from setuptools import find_packages, setup

setup(
    name="skly",
    version="1.0",
    author="Alex Rudy",
    author_email="alex.rudy@bit.ly",
    description="BitlyIQ Data Science Tools",
    packages=find_packages(),
    install_requires=["pyyaml", "scikit-learn", "pathlib;python_version<'3.3'"],
)
