from setuptools import setup, find_packages
setup(
		name="ChemListem",
		version="0.1.0",
		description="Chemical Named Entity Recognition using Deep Learning",
		packages=find_packages(),
		package_data={'':['*.txt']},
		author="Peter Corbett",
		author_email="corbettp@rsc.org",
		url="https://bitbucket.org/rscapplications/chemlistem",
		license="MIT",
		install_requires=['keras>=2.0.3', 'scikit-learn', 'numpy', 'h5py']
)