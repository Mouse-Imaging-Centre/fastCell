import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fastCell",
    version="0.0.1",
    author="Nick Wang",
    author_email="nick.wang@mail.mcgill.ca",
    description="Fast Deep Neural Networks for Cell Image Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mouse-Imaging-Centre/fastCell",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    scripts=[
        'fastCell/deep_segment.py',
        'fastCell/pixelize_segment.py'
    ],
    install_requires=[
        'opencv-python==4.1,==3.4'
        'pandas>=0.25',
        'fastai>=1.0.53'
    ]
)