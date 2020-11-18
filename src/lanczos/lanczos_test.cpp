#include <iostream>



#include "cmpt/eigen_ex/lanczos.hpp"



namespace cmpt{
  namespace workspace{
    inline int main(){

      // sample calculation for LanczosEigenSolver
				if (1) {
					using Scalar = std::complex<double>;
					using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
					using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
					using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
					using RealVectorType = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;

					int n = 50;	// matrix height

					std::function<void(const Scalar*, Scalar*)> matrix_multiple;	// matrix multiplication function
					Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(n, n);	// a Hermite matrix to diagonalize,  H=U D U^t
					Eigen::MatrixXcd U = Eigen::MatrixXcd::Zero(n, n);	// a Unitary matrix which has eigen vectors
					Eigen::MatrixXcd D = Eigen::MatrixXcd::Zero(n, n);	// a Diagonal matrix which has eigen values, D=U^t H U

					//** settings of matrix D, U, H **


					//* settings of D *
					{
						double lambda_min = 0.0;
						double lambda_max = 50.0;

						if (0) {	// sequence 
							for (int i = 0; i < n; ++i) {
								D(i, i) = Scalar(lambda_min + (lambda_max - lambda_min) * i / n);
							}
						}
						if (1) {	// sequence and degeneration
							for (int i = 0; i < n; ++i) {
								D(i, i) = Scalar(lambda_min + (lambda_max - lambda_min) * i / n);
							}
							for (int i = 0; i < n; i += 2) {
								D(i + 1, i + 1) = D(i, i);
							}
						}
						if (0) {	// random
							std::normal_distribution<RealScalar> norm;
							std::mt19937 engine(5);
							for (int i = 0; i < n; ++i) {
								D(i, i) = Scalar(norm(engine), norm(engine));
							}
						}
					}

					//* settings of U *
					if (1) {
						std::mt19937 engine(2);
						EigenEx::OrthogonalMatrixDistribution<MatrixType> mdist(n, n);
						U = mdist(engine);
					}
					if (0) {
						U = MatrixType::Identity(n, n);
					}

					//* settings of H *
					H = U * D * U.adjoint();

					//* settings of matrix multiplication function *
					matrix_multiple = [n, &H](const Scalar* in, Scalar* out) {
						Eigen::Map<VectorType> mout(out, n);
						Eigen::Map<const VectorType> min_(in, n);
						mout = H * min_;
					};

					//** diagonalization **

					// use cmpt::lanczos::EigenSolver
					if (1) {
						//** settings of solver **
						using Solver = cmpt::lanczos::LanczosEigenSolver<Scalar>;

						std::mt19937 random_engine(1);
						Solver es;

						if (1) {
							es.setMatrixMultiplication(matrix_multiple, n);	// set matrix-vector multiplication, necessary
							es.setEigenvalueShift(0.0);											// the default is 0.0
							es.setTolerance(1.0e-10);												// the default is 1.0e-12 for double
							es.setThreshold(1.0e-14);												// the default is 1.0e-12 for double
							es.setMinIterations(Solver::unlimited);					// the default is unlimited(==-1)
							es.setMaxIterations(1000);											// the default is unlimited(==-1)
							es.setComputeEigenvectorsOn(true);							// the default is true
							es.setIndicesForConvergence({ 0 });							// the default is {0}
							es.setInitialVector(es.lanczosBase().makeRandomVector(random_engine, n));	// set initial vector with random engine
							es.setMaxEigenvalues(Solver::unlimited);				// the default unlimited
							es.setOrthogonalizingVectors({});								// the default is {} empty
							es.setReorthogonalizeInterval(1);								// the default is 0
							es.setReserveSize(128);													// reserve size of lanczos vector default is 128
						}

						es.compute();


						RealVectorType  eivals = es.eigenvalues();

						std::cout << "matrix height : " << es.matrixHeight() << std::endl;
						std::cout << "iterations : " << es.iterations() << std::endl;
						std::cout << "subspace rank : " << es.lanczosvectors().size() << std::endl;

						std::cout << "eigen values : \n" << eivals << std::endl;
						std::cout << "convergenceLog :\n";


						for (auto& edges_ : es.convergenceLog()) {
							auto& idx = edges_.first;
							auto& edges = edges_.second;
							std::cout << "## " << idx << " ##" << std::endl;
							for (auto& edge : edges) {
								std::cout << edge << std::endl;
							}
							std::cout << std::endl;
						}
						std::cout << "log :\n";
						for (auto& str : es.log()) {
							std::cout << str << std::endl;
						}
					}

					// time evolving
					if (0) {
						//** settings of solver **
						using Solver = cmpt::lanczos::LanczosEigenSolver<Scalar>;
						using ExpSolver = cmpt::lanczos::ExponentialSolver<Scalar>;

						std::mt19937 random_engine(1);
						Solver es;

						if (1) {
							es.setMatrixMultiplication(matrix_multiple, n);
							es.setEigenvalueShift(0.0);
							es.setTolerance(1.0e-10);
							es.setThreshold(1.0e-14);
							es.setMinIterations(Solver::unlimited);
							es.setMaxIterations(1000);
							es.setComputeEigenvectorsOn(true);
							es.setIndicesForConvergence({ 0 });
							es.setInitialVector(es.lanczosBase().makeRandomVector(random_engine, n));
							es.setMaxEigenvalues(Solver::unlimited);
							es.setOrthogonalizingVectors({});
							es.setReorthogonalizeInterval(1);
							es.setReserveSize(128);
						}

						es.compute();

						if (es.eigenvalues().size() < 2) { return 0; }
						RealScalar radius = (std::max)(
							std::abs(es.eigenvalues()[0]),
							std::abs(es.eigenvalues()[es.eigenvalues().size() - 1])
							);

						Scalar start(0.0, 0.0);
						Scalar end(0.0, -0.01);
						int div = 20;
						Scalar dx = (end - start) * (1.0 / div);
						VectorType v = es.makeRandomVector(random_engine, n);
						VectorType v_l = v;
						VectorType v_t = v;

						std::vector<Scalar> xs;
						std::vector<Scalar> diffs;
						std::vector<Scalar> norm_l;
						std::vector<Scalar> norm_t;


						for (int i = 0; i < div; ++i) {
							Scalar x = start + (end - start) * (static_cast<RealScalar>(i) / div);
							xs.push_back(x);
							VectorType v_temp;
							ExpSolver::solveWithTaylorAutoDivision(
								dx,
								matrix_multiple,
								n,
								radius,
								v_t,
								v_temp,
								1.0e-10,
								-1
							);
							v_t = v_temp;

							es.setInitialVector(std::move(v_l));
							ExpSolver::solveWithLanczos(
								dx,
								es,
								v_l
							);

							diffs.push_back((v_t - v_l).norm());
							norm_t.push_back(v_t.norm());
							norm_l.push_back(v_l.norm());

						}

						std::cout << "diffs  norm_t  norm_l" << std::endl;
						for (int i = 0, ni = diffs.size(); i < ni; ++i) {
							std::cout
								<< diffs[i] << "  "
								<< norm_t[i] << "  "
								<< norm_l[i] << "  "
								<< std::endl;
						}

					}


				}


      return 0;
    }
  }
}

int main(){
  return cmpt::workspace::main();
}
