import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="food-extractor", 
    version="0.9.0",
    author="Charlene Chambliss",
    description="A toolkit for training and evaluating lightweight BERT models for ingredient and product extraction.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chambliss/food-extractor",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.8',
)