from setuptools import setup, find_packages

setup(
    name="flash-attention-triton",
    version="0.1.0",
    description="Implementation of FlashAttention with bias in Triton",
    author="pengzhangzhi",
    author_email="",  # Add your email if you want
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "triton>=2.0.0",
    ],
    python_requires=">=3.8",
) 