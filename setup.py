
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="snovalleyai_piecewise_polynomial_layers",
    version="1.0.0",
    author="John Loverich",
    author_email="john.loverich@gmail.com",
    description="Piecewise polynomial layer for tensorflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jloveric/piecewise-polynomial-layers",
    packages=setuptools.find_packages(),
    install_requires=['numpy','tensorflow'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 
