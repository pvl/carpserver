import setuptools

setuptools.setup(
    name="CARPServer",
    version="0.1.0",
    description="Server api for CARP model",
    url="https://github.com/pvl/carpserver",
    packages=setuptools.find_packages(),
    install_requires=open('requirements.txt').readlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
