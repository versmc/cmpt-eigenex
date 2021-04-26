# Some Extensions for Eigen3
## Summary
This repository contains some extensions for [Eigen3](http://eigen.tuxfamily.org/index.php).

## Features (my memo)
- header only
- c++11-like interface (random,initializer_list)
- Eigen-like interface (some solver classes)
- no json interface for Eigen (, which is separated to other repositories)
- no dependence on my "cmpt/util.hpp"


## Contents

- c++11-like random distribution for Vector, Matrix, Tensor in Eigen3
- Lanczos method
- Arnoldi method
- multi-dimensional indices interpreter
- SVD(Singular Values Decomposition) for Tensor
- BlockTensor (tensor class for block sparse tensor)
- numpy-like einsum (under construction)
- etc...

## Dependences
### c++11
for some functions
### Eigen3
For #include "Eigen/...".
The include path must contains unsupported/CXX11 to use Tensor classes.


