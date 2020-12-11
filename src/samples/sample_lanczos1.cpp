//
// This code is a simplest example for lanczos method using cmpt::EigenEx::LanczosEigenSolver
//

#include <iostream>

#include "cmpt/eigen_ex/lanczos.hpp"


int main(){
  
  // definition of matrix to diagonalize
  int n=3;
  Eigen::MatrixXd H(n,n);
  H<< 1.0, 0.5, 0.0,
      0.5, 2.0, 0.5,
      0.0, 0.5, 3.0;

  // definition of matrix-vector multiplication
  auto matmul=[&H](double const* in,double* out){
    Eigen::Map<const Eigen::VectorXd> v_in(in,H.rows());
    Eigen::Map<Eigen::VectorXd> v_out(out,H.rows());
    v_out=H*v_in;
  };

  // set lanczos solver
  cmpt::EigenEx::LanczosEigenSolver<double> lanczos;
  lanczos.setMatrixMultiplication(matmul,n);
  lanczos.setTolerance(1.0e-5); // set tolerance to judge the convergence
  lanczos.setMaxIterations(100); // set max limit of iterations
  
  lanczos.compute();

  Eigen::VectorXd eivals=lanczos.eigenvalues();
  Eigen::MatrixXd eivecs=lanczos.eigenvectors();

  // output results
  std::cout<<"eigen values:\n"<<eivals<<std::endl;
  std::cout<<"eigen vectors:\n"<<eivecs<<std::endl;
  for(auto& log:lanczos.log()){
    std::cout<<log<<std::endl;
  }

	return 0;
}
