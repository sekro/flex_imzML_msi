import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="flex_imzML-sekro",
    version="2022.0.12",
    author="Sebastian Krossa",
    author_email="sebastian.krossa@ntnu.no",
    description="A set of tools for working with MS imaging data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sekro/flex_imzML_msi",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)