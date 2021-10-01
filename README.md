<div class="row">
  <div class="column">
    <img src="https://github.com/GAA-UAM/GBNN/blob/main/doc/nn_hlt_nlayer_gne_mc.jpg" alt="NN" style="width:30%">
  </div>
  <div class="column">
    <img src="https://github.com/GAA-UAM/GBNN/blob/main/doc/nn_hlt_nlayer_mc.jpg" alt="mc" style="width:30%">
  </div>
</div>


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
   * [Updated](#Updated)
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
The package is licensed under the [GNU Lesser General Public License v2.1](https://github.com/GAA-UAM/GBNN/blob/main/LICENSE).



# Installation
First, clone this project then, install the Python package:

```bash
pip install .
```

# Usage
If you want to use this library easily after installing it, you could import 
it into your python project. You can use this package with the standards of 
Scikit-learn.
Note that this project can run on both Windows and Linux and is tested on both of them.
## GBNN-Example
In the following, one can see the example of implementing the algorithm. 

```python
import gbnn

model = gbnn.GNEGNEClassifier(total_nn=200, num_nn_step=1, eta=1.0, solver='lbfgs',
                     subsample=0.5, tol=0.0, max_iter=200, random_state=None, activation='logistic')
model.fit(x_train, y_train)
model.predict(x_test)
```
The default values of the GBNN's hyper-parameters are, as above code. 
The `total_nn` applies to the number of hidden units. The `total_nn` regards the units per iteration. 
And `activation` introduces the default activation function of the base neural network.
<br/>
Check the [example](https://github.com/GAA-UAM/GBNN/tree/main/examples/examples.ipynb) file to see the GBNN performance over the classification and regression problem.


## cross-validation-Example
To implement the GBNN method through the cross-validation processes with K folds, 
you could also consider the [cross_validation.py](https://github.com/GAA-UAM/GBNN/tree/main/gbnn/cross_validation.py) and imported as following.
With this file, you would also take advantage of the Grid Search method in order to select the optimized hyper-parameter.

```python
from gbnn import cross_validation, GNEGNEClassifier

model = GNEGNEClassifier()
param_grid = {'clf__num_nn_step': [1, 2, 3, 4], 'clf__subsample': [
    0.25, 0.5, 0.75, 1], 'clf__eta': [0.025, 0.05, 0.1, 0.5, 1]}
cross_validation.gridsearch(X, y, model, param_grid, scoring_functions,
                            pipeline, best_scoring, random_state, n_cv_general, n_cv_intrain)
```
the `n_cv_general`, refers to the number of cross-validation. the `n_cv_intrain`, refers to the number of within-train cross-validation.

## Requirements
This package takes advantage of the following libraries, which had already imported to the GBNN package.:
- [numpy](https://numpy.org/) - Numerical Python
- [pandas](https://pandas.pydata.org/) - python data analysis library
- [scipy](https://www.scipy.org/) - Scientific computation in Python
- [scikit-learn](https://scikit-learn.org/stable/) - Machine learning in Python

# How to cite GBNN
If you are using GBNN in your paper, please cite our work [GBNN](https://arxiv.org/abs/1909.12098).
```
@article{emami2019sequential,
  title={Sequential training of neural networks with gradient boosting},
  author={Emami, Seyedsaman and Mart{\'\i}nez-Mu{\~n}oz, Gonzalo},
  journal={arXiv preprint arXiv:1909.12098},
  year={2019}
}
```
# Contributions
Contributions to the GBNN are welcome! . You can improve this project by creating an issue, 
reporting an improvement or a bug, forking and creating a pull request to the 
development branch. For more information, check the [contributing guidelines](contributing-guidelines.md).
<br/>
The authors and developers involved in the development of the GBNN package can be found in the [contributor](contributors.txt)'s file.

## Key members of GBNN

* [Gonzalo Martínez-Muñoz](https://github.com/gmarmu)
* [Seyedsaman Emami](https://github.com/samanemami)

# Keywords
**`Gradient Boosting`**, **`Neural Network`**

# Version 
1.0.1

## Updated
2021-09-19

# Date-released
2021-01-27
