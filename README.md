[![DOI](https://zenodo.org/badge/340351316.svg)](https://zenodo.org/badge/latestdoi/340351316)


# pyADA
pyADA (Python Applicability Domain Analyzer) is a cheminformatics package to perform Applicability Domain of molecular fingerprints based in similarity calculation.
In this case, the calculation of the Applicability Domain consists of a scan of similarities of the structures
present in the test set in relation to the training set, the best similarity threshold is the one with the lowest
error and also the lowest number of molecules with similarity below the threshold. 
A notebook file with an example of using this package is present in the directory 'example/example_of_use.ipynb'
### Dependencies
<ul>
<li><b>numpy</b></li>
<li><b>pandas</b></li>
<li><b>tqdm</b></li>
<li><b>scikit-learn</b></li>
<li><b>Tested in python3.6 and python3.8</b></li>
</ul>

## Install
<b>Via pip</b>
```
pip3 install pyADAqsar
```

<b>Via github</b>
```
git clone https://github.com/jeffrichardchemistry/pyADA
cd pyADA
python3 setup.py install
```

## How to use
This package has three classes: Smetrics (perform some statistical parameters like Q2ext R2ext etc), Similarity (realize similarity calculations based in differents metrics ) and ApplicabilityDomain (run a scan of AD with differents thresholds). The line code bellow import all classes.
```
from pyADA import Smetrics, Similarity, ApplicabilityDomain
```
A file containing a jupyter-notebook with a few examples of use is in 'example' folder.
For more information about documentation run the help function of classes.
```
help(Smetrics)
help(Similarity)
help(ApplicabilityDomain)
```

# How to cite
- BibTex
```
@article{dias2023spectrafp,
  title={SpectraFP: a new spectra-based descriptor to aid in cheminformatics, molecular characterization and search algorithm applications},
  author={Dias-Silva, Jefferson R and Oliveira, Vitor M and Sanches-Neto, Fl{\'a}vio O and Wilhelms, Renan Z and J{\'u}nior, Luiz HK Queiroz},
  journal={Physical Chemistry Chemical Physics},
  volume={25},
  number={27},
  pages={18038--18047},
  year={2023},
  publisher={Royal Society of Chemistry}
}
```

- Normal citation

```
Dias-Silva, J. R., Oliveira, V. M., Sanches-Neto, F. O., Wilhelms, R. Z., & Júnior, L. H. Q. (2023). SpectraFP: a new spectra-based descriptor to aid in cheminformatics, molecular characterization and search algorithm applications. Physical Chemistry Chemical Physics, 25(27), 18038-18047.
```

```
DIAS-SILVA, Jefferson R. et al. SpectraFP: a new spectra-based descriptor to aid in cheminformatics, molecular characterization and search algorithm applications. Physical Chemistry Chemical Physics, v. 25, n. 27, p. 18038-18047, 2023.
```

```
Dias-Silva, Jefferson R., et al. "SpectraFP: a new spectra-based descriptor to aid in cheminformatics, molecular characterization and search algorithm applications." Physical Chemistry Chemical Physics 25.27 (2023): 18038-18047.```
