## Table of contents
* [GBNN](#GBNN)
* [Title](#Title)
    * [Citation](#Citation)
        * [Paper](#Paper)
        * [License](#License)
* [Installation](#Installation)
* [Usage](#Usage)
   * [Requirements](#Requirements)
* [Contributing](#Contributing)  
* [Keywords](#Keywords)  
* [Version](#Version)  
* [Date-released](#Date-released)      




# GBNN
GBNN is a python library for dealing with classification (Binary and multi-class) and regression problems.

# Title
Gradient Boosted Neural Network

## Citation 
If you use this package, please cite it as below.

```yaml
References:
    Type: article
    Authors:
      - Seyedsaman Emami
      - Gonzalo Martínez-Muñoz
    Arxiv:
      - https://arxiv.org/abs/1909.12098
    Keywords:
      - Gradient Boosting
      - "Neural Network"
```
### Paper
[Sequential Training of Neural Networks with Gradient Boosting](https://arxiv.org/abs/1909.12098)

### License
The package is licensed under the [GNU General Public License v3.0](https://spdx.org/licenses/GPL-3.0-or-later.html).



# Installation
Installing via [Git](https://github.com/) to install GBNN.

```bash
$ python -m pip install git+https://github.com/GAA-UAM/GBNN/GBNN.py
```

# Usage
If you want to use this library easily after installing it, you could import 
it into your python project. You can use this package with the standards of 
Scikit-learn.
In the following, one can see the example of implementing the algorithm. 

```python
import GBNN

model = GBNN.GNEGNEClassifier()
model.fit(x_train, y_train)
model.predict(x_test)
```

## Requirements
This package takes advantage of the following libraries:
- [numpy](https://numpy.org/) - Numerical Python
- [pandas](https://pandas.pydata.org/) - python data analysis library
- [scipy](https://www.scipy.org/) - Scientific computation in Python
- [scikit-learn](https://scikit-learn.org/stable/) - Machine learning in Python
<br/>
These libraries had already imported to the GBNN package.

# Contributing
All contributions are welcome. You can help this project by creating an issue, 
reporting an improvement or a bug, forking and creating a pull request to the 
development branch.
<br/>
The authors and developers involved in the development of the GBNN package can be found in the [contributor](contributors.txt)'s file.


# Keywords
**`Gradient Boosting`**, **`Neural Network`**

# Version 
1.0.0

# Date-released
2021-01-27
