
# stepshift

Stepshift is a package that implements the stepshifting algorithm described in appendix A of  
[Hegre et al. (2020)](https://journals.sagepub.com/doi/full/10.1177/0022343320962157).

## Installation

Stepshift is currently only distributed as a source distribution, which means
that the end user needs a C compiler. This means that OSX users need to have
Xcode installed on their system before proceeding. In addition, the numpy
requirement is quite strict, since stepshift uses the Numpy C API via Cython.

Install by running:

```
pip install stepshift
```

## Usage

Stepshift has a module called `stepshift.views` which contains a class called
`StepshiftedModels`. This class wraps the stepshifting procedure, exposing a
simple, Scikit-Learn-like (but not equivalent) API. The model takes three
arguments: A scikit learn estimator, a list containing integers, which denotes
the steps, and a string variable which is the name of the dependent variable:

```
from sklearn.linear_model import LogisticRegression
from stepshift.views import StepshiftedModels

mdl = StepshiftedModels(LogisticRegression(),[1,2,3,4,5,6,7,8],"outcome")
```

