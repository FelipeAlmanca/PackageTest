from setuptools import setup, find_packages

with open("README.md", "r") as f:
    page_description = f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Package_test",
    version="0.0.1",
    author="Felipe",
    author_email="e-maill",
    description="Random Forest em Arquivo train.csv",
    long_description=page_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FelipeAlmanca/PackageTest"
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.8',
)
