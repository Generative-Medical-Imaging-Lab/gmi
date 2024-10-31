from setuptools import setup, find_packages

setup(
    name='gmi',
    version='0.0',
    description='Generative Medical Imaging Laboratory (GMI) package',
    author='Your Name',
    author_email='tivnanmatt@gmail.com',
    url='https://github.com/Generative-Medical-Imaging-Lab/gmi',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here, e.g., 'numpy', 'scipy'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)