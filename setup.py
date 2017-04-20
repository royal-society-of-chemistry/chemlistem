from setuptools import setup, find_packages
setup(
		name="ChemListem",
		version="0.0.2",
		description="Chemical Named Entity Recognition using Deep Learning",
		packages=find_packages(),
		package_data={'':['*.txt']},
		author="Peter Corbett",
		author_email="corbettp@rsc.org",
		url="https://bitbucket.org/rscapplications/chemlistem",
		download_url="https://bitbucket.org/rscapplications/chemlistem/downloads/ChemListem-0.0.1-py3-none-any.whl",
		license="MIT",
		install_requires=['keras>=2.0.3', 'scikit-learn', 'numpy', 'tensorflow>=1.0.1']
)