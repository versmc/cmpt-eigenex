//
// This is an simple sample programs for cmpt::EigenEx::ArnoldiEigenSolver
// The interface is similar to cmpt::EigenEx::LanczosEigenSolver
// 

#include <iostream>
#include "Eigen/Core"
#include "cmpt/eigen_ex/arnoldi.hpp"


int main(){

  // type alias
  using Scalar = std::complex<double>;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  

  // a test for EigenSolver 
  if (1) {
    int n = 50; // matrix height
    int m = 40; // upper limit of subspace dimension for Arnoldi method

    // definition of matrix and matrix-vector multiplication function
    MatrixType A = MatrixType::Random(n, n);
    auto matmul = [n, A](Scalar const* in, Scalar* out) {
      Eigen::Map<const VectorType> m_in(in, n);
      Eigen::Map<VectorType> m_out(out, n);
      m_out = A * m_in;
    };

    // settings of Arnoldi method (There are more settings parameter)
    cmpt::EigenEx::ArnoldiEigenSolver<Scalar> es;
    es.setMatrixMultiplication(matmul, n);
    es.setMaxIterations(m);   // in this case, fix subspace rank
    es.setMinIterations(m);   // in this case, fix subspace rank
    es.setTolerance(1.0e-14); // set error to judge the convergence, but in this case maxIterations has priority to tolerance
    es.setMaxEigenvalues(2);  // compute small 2 eigenvalues and eigenvectors
    
    es.compute(); // computation of Arnoldi method

    MatrixType P = es.eigenvectors();
    MatrixType D = es.eigenvalues().asDiagonal();

    MatrixType AP = A * P;
    MatrixType PD = P * D;

    // shold be zero matrix if m == n
    // Since m < n in this case, AP-PD is approximately zero
    MatrixType zero_AP_PD = AP - PD;

    std::cout<<"AP-PD\n"<<zero_AP_PD<<std::endl; 



  }


  return 0;
}
