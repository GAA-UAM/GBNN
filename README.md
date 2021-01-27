


# GBNN
GBNN is a python library for dealing with classification (Binary and multi-class) and regression problems.

# Title
Gradient Boosted Neural Network

## Citation 
If you use this package, please cite it as below.
<br/> **Authors:**
- Seyedsaman Emami 
- Gonzalo Martínez-Muñoz

## Installation
Installing via [Git](https://github.com/) to install GBNN.

```bash
$ python -m pip install git+https://github.com/GAA-UAM/GBNN/GBNN.py
```

## Usage
You can simply use this package with the standards of Scikit-learn.

```python
import GBNN

model = GBNN.GNEGNEClassifier()
model.fit(x_train, y_train)
model.predict(x_test)
```

## Requirements
This package takes advantage of the following libraries:
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [scipy](https://www.scipy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
<br/>
These libraries had already imported to the GBNN package.

### References
"Cite this paper when you run GBNN"
- Type: Article
- Title: Sequential Training of Neural Networks with Gradient Boosting
- Doi:

### License
[GNU General Public License v3.0](https://spdx.org/licenses/GPL-3.0-or-later.html)

### Keywords
Gradient Boosting, Neural Network

### Version 
1.0.1

### Date-released
2017-12-18


