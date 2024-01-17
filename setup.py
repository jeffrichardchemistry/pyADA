from setuptools import setup, find_packages

with open("README.md", 'r') as fr:
	description = fr.read()

setup(
    name='pyADAqsar',
    version='1.1.1',
    url='https://github.com/jeffrichardchemistry/pyADA',
    license='GNU GPL',
    author='Jefferson Richard',
    author_email='jrichardquimica@gmail.com',
    keywords='Cheminformatics, Chemistry, Applicability Domain, QSAR, SAR, Molecular Fingerprint',
    description='A cheminformatics package to perform Applicability Domain of molecular fingerprints based in similarity calculation.',
    long_description = description,
    long_description_content_type = "text/markdown",
    packages=['pyADA'],
    install_requires=['pandas<=2.0.3', 'scipy<=1.10.1','numpy<=1.24.4', 'tqdm', 'scikit-learn<=1.3.2','plotly'],
    extras_require={'molplotly':['molplotly']},
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
		'Programming Language :: Python :: 3.8',]
)
