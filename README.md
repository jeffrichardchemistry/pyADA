# pyADA
pyADA is a cheminformatics package to perform Applicability Domain of molecular fingerprints based in similarity calculation.
In this case, the calculation of the Applicability Domain consists of a scan of similarities of the structures
present in the test set in relation to the training set, the best similarity threshold is the one with the lowest
error and also the lowest number of molecules with similarity below the threshold. 
A notebook file with an example of using this package is present in the directory 'example/example_of_use.ipynb'
### Dependencies
<ul>
<li><b>numpy</b></li>
<li><b>pandas</b></li>
<li><b>tqdm</b></li>
</ul>

## Install
<b>Via github</b>
```
git clone https://github.com/jeffrichardchemistry/pyADA
cd pyADA
python3 setup.py install
```
