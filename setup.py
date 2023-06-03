from setuptools import find_packages, setup

install_requires = ["torch>=1.10.2"]

setup(
    name="effop",
    author="Matthew Muckley",
    author_email="matt.muckley@gmail.com",
    version="0.0.0",
    packages=find_packages(exclude=["tests", "experiments", "configs" "notebooks"]),
    setup_requires=["wheel"],
    install_requires=install_requires,
)
