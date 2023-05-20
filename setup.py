from setuptools import setup,find_packages
setup(     
    name='mypackage',
    author='Me',     
    version='1.0',
    packages=find_packages(),
    url="https://github.com/uonat/SS2023_DI-Lab_Precitaste",     
    install_requires=[         
        'timm',
        'pyyaml==5.1',   
    ],
    dependency_links = ['git+https://github.com/facebookresearch/detectron2.git'],
    # ... more options/metadata
)








"""
!python -m pip install pyyaml==5.1
import distutils.core
# Note: This is a faster way to install detectron2 in Colab, but it does not include all functionalities.
# See https://detectron2.readthedocs.io/tutorials/install.html for full installation instructions
!git clone 'https://github.com/facebookresearch/detectron2'  &> /dev/null
dist = distutils.core.run_setup("./detectron2/setup.py")
!python -m pip install {' '.join([f"'{x}'" for x in dist.install_requires])} &> /dev/null
sys.path.insert(0, os.path.abspath('./detectron2'))

# Properly install detectron2. (Please do not install twice in both ways)
# !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'


"""