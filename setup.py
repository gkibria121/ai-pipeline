from setuptools import setup, find_packages

setup(
    name="aasist",
    version="1.0.0",
    description="AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention",
    author="Jee-weon Jung, Hemlata Tak",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "torchaudio>=0.10.0",
        "numpy>=1.20.0",
        "soundfile>=0.10.3",
        "scipy>=1.7.0",
        "tqdm>=4.62.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)