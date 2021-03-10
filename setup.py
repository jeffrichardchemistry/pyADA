from setuptools import setup, find_packages

with open("README.md", 'r') as fr:
	description = fr.read()

setup(
    name='pyADAqsar',
    version='1.0.0',
    url='https://github.com/jeffrichardchemistry/pyADA',
    license='GNU GPL',
    author='Jefferson Richard',
    author_email='jrichardquimica@gmail.com',
    keywords='Cheminformatics, Chemistry, Applicability Domain, QSAR, SAR, Molecular Fingerprint',
    description='A cheminformatics package to perform Applicability Domain of molecular fingerprints based in similarity calculation.',
    long_description = description,
    long_description_content_type = "text/markdown",
    packages=['pyADA'],
    install_requires=['pandas<=1.1.5', 'scipy<=1.5.4','numpy<=1.19.5', 'tqdm<=4.57.0', 'scikit-learn'],
	classifiers = [
		'Intended Audience :: Developers',
		'Intended Audience :: Science/Research',
		'Topic :: Scientific/Engineering :: Chemistry',
		'Topic :: Scientific/Engineering :: Physics',
		'Topic :: Scientific/Engineering :: Bio-Informatics',
		'Topic :: Scientific/Engineering',
		'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
		'Natural Language :: English',
		'Operating System :: Microsoft :: Windows',
		'Operating System :: POSIX :: Linux',
		'Environment :: MacOS X',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3.6',
		'Programming Language :: Python :: 3.7',
		'Programming Language :: Python :: 3.8',]
)
