import setuptools

setuptools.setup(
    name="GBNN",
    version="0.0.2",
    author="Seyedsaman Emami, Gonzalo Martínez-Muñoz",
    author_email="emami.seyedsaman@uam.es, gonzalo.martinez@uam.es",
    description="Gradient Boosted Neural Network",
    packages=['gbnn'],
    install_requires=['numpy', 'scipy', 'sklearn'],
    classifiers=("Programming Language :: Python :: 3")
)
