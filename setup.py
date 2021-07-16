from setuptools import find_packages, setup
from my_project import __version__

setup(
    name="my_project",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["wheel"],
    version=__version__,
    description="industrialization of project",
    author="Ange KOUAME Aymen",
)
