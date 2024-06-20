from setuptools import setup, find_packages

setup(
    name='clf-spatial-uniformity',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        'numpy', 
        'scipy'
    ],
    author='Willian Oliveira',
    description='Spatial uniformity of remote sensing image classification results',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/wivoliveira/clf-spatial-uniformity',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: Apache License 2.0',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
