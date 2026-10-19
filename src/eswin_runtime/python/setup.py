from setuptools import setup, find_packages

setup(
    name="eswin-runtime",
    version="1.0.0",
    description="Python bindings for Open ESWIN NPU Runtime (EIC7700X Dual-Die)",
    author="Antigravity Team",
    packages=find_packages(),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)
