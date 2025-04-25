import setuptools

setuptools.setup(
     name="OrbitTails",
     version="0.1",
     author="Ana Maria Melian",
     author_email="anamaria.melian@yale.edu",
     description="A user-friendly Python package for calculating and comparing ram pressure stripped galaxy tail angles.",
     packages=["OrbitTails"],
     install_requires=["numpy", "matplotlib", "astropy", "gala"],
     package_dir={"": "src"}
)