from setuptools import setup, find_packages
setup(
		name="ChemListem",
		version="0.0.1",
		description="Chemical Named Entity Recognition using Deep Learning",
		packages=find_packages(),
		package_data={'':['*.txt','*.h5','*.json']},
		author="Peter Corbett",
		author_email="corbettp@rsc.org",
		url="http://www.example.com/chemilistem/",
		download_url="http://www.example.com/chemilistem/tarball/chemlistem.tar.gz",
		license="MIT",
		install_requires=['keras', 'scikit-learn', 'numpy']
)