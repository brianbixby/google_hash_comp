from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'numpy>=1.16.1',
    'tensorflow>=1.11.0',
    'matplotlib>=3.0.2',
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)