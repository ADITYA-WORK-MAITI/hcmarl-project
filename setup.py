"""HC-MARL: Human-Centric Multi-Agent Reinforcement Learning."""

from setuptools import setup, find_packages

setup(
    name="hcmarl",
    version="0.1.0",
    author="Aditya Maiti",
    author_email="aditya.maiti@ipu.ac.in",
    description=(
        "Human-Centric Multi-Agent Reinforcement Learning for Safe and Fair "
        "Human-Robot Collaboration in Warehouse Environments"
    ),
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "cvxpy>=1.4.0",
        "osqp>=0.6.3",
        "pyyaml>=6.0",
        "torch>=2.0.0",
        "gymnasium>=0.29.0",
        "pettingzoo>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
        "wandb": [
            "wandb>=0.16.0",
        ],
    },
)
