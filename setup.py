from setuptools import setup, find_packages

setup(
    name="adaptive-collapse-dynamics",
    version="1.0.0",
    author="Stanislav Usychenko",
    description="ACE: Adaptive Collapse Eviction for Transformer KV-Cache",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.9",
)