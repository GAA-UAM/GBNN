
![![alt text](https://github.com/GAA-UAM/GBNN/blob/main/doc/GBNNcover1.png)](https://github.com/GAA-UAM/GBNN/blob/main/doc/GBNNcover1.png)

## Table of contents
* [GBNN](#GBNN)
* [Title](#Title)
    * [Citation](#Citation)
        * [Paper](#Paper)
        * [License](#License)
* [Installation](#Installation)
* [Usage](#Usage)
   * [GBNN method](#GBNN-Example)
   * [cross-validation](#cross-validation-Exapmle)
   * [Requirements](#Requirements)
* [Contributions](#Contributions)  
* [Keywords](#Keywords)  
* [Version](#Version)  
* [Date-released](#Date-released)      




# GBNN
GBNN is a python library for dealing with classification (Binary and multi-class) and regression problems.

# Title
Gradient Boosted Neural Network

## Citation 
If you use this package, please [cite](CITATION.cff) it as below.

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
## GBNN-Example
In the following, one can see the example of implementing the algorithm. 

```python
import GBNN

model = GBNN.GNEGNEClassifier(total_nn=200, num_nn_step=1, eta=1.0, solver='lbfgs',
                     subsample=0.5, tol=0.0, max_iter=200, random_state=None, activation='logistic')
model.fit(x_train, y_train)
model.predict(x_test)
```
The default values of the GBNN's hyper-parameters are, as above code. 
The `total_nn` applies to the number of hidden units. The `total_nn` regards the units per iteration. 
And `activation` introduces the default activation function of the base neural network.


## cross-validation-Exapmle
To implement the GBNN method through the cross-validation processes with K folds, 
you could also consider the [cross-validation.py](crossvalidation.py) and imported as following.
With this file, you would also take advantage of the Grid Search method in order to select the optimized hyper-parameter.

```python
import crossvalidation as gridsearch

model = GBNN.GNEGNEClassifier()
param_grid = {'clf__num_nn_step': [1, 2, 3, 4], 'clf__subsample': [
    0.25, 0.5, 0.75, 1], 'clf__eta': [0.025, 0.05, 0.1, 0.5, 1]}
gridsearch(X, y, model, param_grid, scoring_functions,
                pipeline, best_scoring, random_state, n_cv_general, n_cv_intrain)
```
the `n_cv_general`, refers to the number of cross-validation. the `n_cv_intrain`, refers to the number of within-train cross-validation.

## Requirements
This package takes advantage of the following libraries, which had already imported to the GBNN package.:
- [numpy](https://numpy.org/) - Numerical Python
- [pandas](https://pandas.pydata.org/) - python data analysis library
- [scipy](https://www.scipy.org/) - Scientific computation in Python
- [scikit-learn](https://scikit-learn.org/stable/) - Machine learning in Python


# Contributions
Contributions to the GBNN are welcome! . You can improve this project by creating an issue, 
reporting an improvement or a bug, forking and creating a pull request to the 
development branch. For more information, check the [contributing guidelines](contributing-guidelines.md).
<br/>
The authors and developers involved in the development of the GBNN package can be found in the [contributor](contributors.txt)'s file.


# Keywords
**`Gradient Boosting`**, **`Neural Network`**

# Version 
1.0.0

# Date-released
2021-01-27
