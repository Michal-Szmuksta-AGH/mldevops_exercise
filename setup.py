from setuptools import setup, find_packages

setup(
    name="autoencoders",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "jupyterlab==4.2.5",
        "notebook==7.2.2",
        "matplotlib==3.9.2",
        "numpy==1.26.4",
        "torch",
        "torchvision",
        "torchaudio"
    ],
    python_requires="==3.10.14",
)
