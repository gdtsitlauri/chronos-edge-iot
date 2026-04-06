"""CHRONOS: Causal Hypergraph-Regulated Orchestration of Networked Offloading with Spiking Intelligence."""

from setuptools import setup, find_packages

setup(
    name="chronos",
    version="0.1.0",
    description="Causal Hypergraph Framework for Joint Federated Learning, "
                "Task Scheduling, and Communication Optimization in Edge-IoT Systems",
    author="CHRONOS Research Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "networkx>=3.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "tensorboard>=2.13.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.2.0",
        "pot>=0.9.0",           # Python Optimal Transport
        "causal-learn>=0.1.3",  # Causal discovery
    ],
    extras_require={
        "dev": ["pytest>=7.0", "black", "flake8"],
        "snn": ["snntorch>=0.7.0"],  # Spiking Neural Networks
    },
)
