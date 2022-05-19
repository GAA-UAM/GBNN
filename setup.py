from setuptools import setup, find_packages
import codecs
import os


here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.2'
DESCRIPTION = 'GBNN'
LONG_DESCRIPTION = 'Gradient Boosted Neural Network.'

setup(name="GBNN",
      version=VERSION,
      author="Seyedsaman Emami, Gonzalo Martínez-Muñoz",
      author_email="emami.seyedsaman@uam.es, gonzalo.martinez@uam.es",
      description=LONG_DESCRIPTION,
      packages=find_packages(),
      install_requires=['numpy', 'scipy', 'sklearn'],
      classifiers=[
          "Development Status :: 1 - Planning",
          "Intended Audience :: Developers",
          "Programming Language :: Python :: 3",
          "Operating System :: Unix",
          "Operating System :: MacOS :: MacOS X",
          "Operating System :: Microsoft :: Windows",
      ]
      )
