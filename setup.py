from setuptools import setup, find_packages

setup(name='htrocr',
      version='0.1',
      description='Package for reading handwritten documents.',
      author="Linas Einikis",
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU Lesser General Public License',
        'Programming Language :: Python :: 3.9',
        'Topic :: Image Processing :: Linguistic',
      ],
      license='GNU',
      python_requires=">=3.9",
      install_requires=[
        "albumentations>=1.3.0",
        "dependency-injector>=4.41.0",
        "pybind11",
        "fire>=0.5.0",  
        "Levenshtein>=0.20.8",
        "lightning-utilities>=0.8.0",
        "lxml>=4.9.2", 
        "matplotlib>=3.7.1",
        "numpy>=1.23.5",
        "pandas>=1.5.2",
        "pytorch-lightning>=2.0.2",
        "Pillow>=9.3.0", 
        "scikit-image>=0.21.0",
        "scikit-learn>=1.2.2",
        "scipy>=1.9.3",
        "tensorflow>=2.13.0",
        "timm>=0.5.4",
        "torch>=1.13.1",
        "torchaudio>=0.13.1",
        "torchmetrics>=0.11.4",
        "torchvision>=0.14.1",
        "tqdm",
        "transformers>=4.26.0",
        "trdg>=1.8.0",
        "wandb>=0.14.0",
      ],
      packages=find_packages(),
      include_package_data=True,
      )