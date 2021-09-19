import setuptools

setuptools.setup(
    name="GBNN",
    version="1.0.1",
    author="Seyedsaman Emami, Gonzalo Martínez-Muñoz",
    author_email="emami.seyedsaman@uam.es, gonzalo.martinez@uam.es",
    description="Gradient Boosted Neural Network",
    packages=['gbnn'],
    install_requires=['numpy', 'scipy', 'sklearn'],
    classifiers=("Programming Language :: Python :: 3")
)
