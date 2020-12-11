#include <iostream>



#include "cmpt/eigen_ex/arnoldi.hpp"



namespace cmpt{
  namespace workspace{
    inline int main(){
      
        using Scalar = std::complex<double>;
				using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
				using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
				using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
				using MatMulFunction = typename EigenEx::ArnoldiBase<Scalar>::MatMulFunction;
				
				// a test for EigenSolver 
				if (0) {
					int n = 50;
					int m = 40;
					MatrixType A = MatrixType::Random(n, n);
					MatMulFunction matmul = [n, A](Scalar const* in, Scalar* out) {
						Eigen::Map<const VectorType> m_in(in, n);
						Eigen::Map<VectorType> m_out(out, n);
						m_out = A * m_in;
					};

					EigenEx::ArnoldiEigenSolver<Scalar> es;
					es.setMatrixMultiplication(matmul, n);
					es.setMaxIterations(m);
					es.setMinIterations(m);

					es.compute();

					MatrixType P = es.eigenvectors();
					MatrixType D = es.eigenvalues().asDiagonal();

					MatrixType AP = A * P;
					MatrixType PD = P * D;

					MatrixType zero_AP_PD = AP - PD;

          std::cout<<"AP-PD\n"<<zero_AP_PD<<std::endl;



				}


				if (1) {
					using AES = EigenEx::ArnoldiEigenSolver<Scalar>;
					using EES = typename AES::EigenSolverTraits<MatrixType>::Type;

					int n = 4;
					MatrixType A = MatrixType::Random(n, n);
					MatMulFunction matmul = [n, A](Scalar const* in, Scalar* out) {
						Eigen::Map<const VectorType> m_in(in, n);
						Eigen::Map<VectorType> m_out(out, n);
						m_out = A * m_in;
					};

					AES aes;
					aes.setMatrixMultiplication(matmul, n);
					aes.setThreshold(1.0e-14);
					aes.setEigenvalueShift(RealScalar(0.0));
					aes.setMaxIterations(aes.unlimited);
					aes.setMinIterations(aes.unlimited);
					aes.setInitialVector();
					aes.setMaxEigenvalues(5);
					aes.setTolerance(1.0e-10);

					aes.compute();

					MatrixType P = aes.eigenvectors();
					MatrixType D = aes.eigenvalues().asDiagonal();

					MatrixType AP = A * P;
					MatrixType PD = P * D;

					MatrixType zero_AP_PD = AP - PD;

          std::cout<<"AP-PD"<<std::endl;
          std::cout<<zero_AP_PD<<std::endl;

					EES ees;
					ees.compute(A);
					VectorType ees_eigenvalues = ees.eigenvalues();
					MatrixType ees_eigenvectors = ees.eigenvectors();



				}


      return 0;
    }
  }
}

int main(){
  return cmpt::workspace::main();
}
