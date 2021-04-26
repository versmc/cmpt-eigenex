#pragma once

/// <summary>
/// This file includes all headers in cmpt/eigen_ex/*.hpp
/// 
/// include path must contains Eigen and unsupported in Eigen (for Tensor)
/// 
/// </summary>


/// <summary>
/// headers depending on Eigen
/// </summary>

#include "cmpt/eigen_ex/util.hpp"

#include "cmpt/eigen_ex/random.hpp"

#include "cmpt/eigen_ex/lanczos.hpp"
#include "cmpt/eigen_ex/arnoldi.hpp"
#include "cmpt/eigen_ex/triplets_matrix.hpp"

/// <summary>
/// headers depending on unsupported in Eigen
/// </summary>

#include "cmpt/eigen_ex/tensor_util.hpp"
#include "cmpt/eigen_ex/tensor_random.hpp"
#include "cmpt/eigen_ex/tensor_svd.hpp"
#include "cmpt/eigen_ex/vector_map.hpp"

#include "cmpt/eigen_ex/multi_indices.hpp"
#include "cmpt/eigen_ex/block_tensor.hpp"

#include "cmpt/eigen_ex/tensor_kronecker_product.hpp"
#include "cmpt/eigen_ex/einsum.hpp"








