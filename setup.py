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
        # Dependencies are managed in requirements.txt
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'pytest-mock>=3.10.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'gmi=main:cli',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)