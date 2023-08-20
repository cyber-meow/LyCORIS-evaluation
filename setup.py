from setuptools import setup, find_packages


# List of requirements
# This could be retrieved from requirements.txt
requirements = []


# Package (minimal) configuration
setup(
    name="lycoris_eval",
    version="0.0.1",
    description="Lycoris evaluation scripts",
    package_dir={"": "."},
    packages=find_packages(),  # __init__.py folders search
    install_requires=requirements
)
