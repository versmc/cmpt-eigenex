//
// This code is a practical example for cmpt::EigenEx::LanczosEigenSolver
//

#include <iostream>
#include <complex>

#include "Eigen/Sparse"
#include "cmpt/eigen_ex/lanczos.hpp"

int main(){
	// type alias
	using Scalar = std::complex<double>;
	using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
	using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
	using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
	using RealVectorType = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;

	int n = 200; // matrix height

	std::vector<Eigen::Triplet<Scalar>> triplets;
	for(int i=0;i<n-1;++i){
		triplets.push_back(Eigen::Triplet<Scalar>(i,i+1,Scalar(0.0,-1.0)));
		triplets.push_back(Eigen::Triplet<Scalar>(i+1,i,Scalar(0.0,+1.0)));
	}

	Eigen::SparseMatrix<Scalar> H(n,n);
	H.setFromTriplets(triplets.begin(),triplets.end());

	auto matmul=[&H](Scalar const* in,Scalar* out){
		Eigen::Map<const VectorType> v_in(in,H.rows());
		Eigen::Map<VectorType> v_out(out,H.cols());
		v_out=H*v_in;
	};


	//* settings of solver *
	using Solver = cmpt::EigenEx::LanczosEigenSolver<Scalar>;

	std::mt19937 random_engine(1);
	Solver es;

	if (1){
		es.setMatrixMultiplication(matmul, n);																		// set matrix-vector multiplication, necessary
		es.setEigenvalueShift(0.0);																								// the default is 0.0
		es.setTolerance(1.0e-7);																									// the default is 1.0e-12 for double
		es.setThreshold(1.0e-14);																									// the default is 1.0e-12 for double
		es.setMinIterations(Solver::unlimited);																		// the default is unlimited(==-1)
		es.setMaxIterations(1000);																								// the default is unlimited(==-1)
		es.setComputeEigenvectorsOn(true);																				// the default is true
		es.setIndicesForConvergence({0});																					// the default is {0}
		es.setInitialVector(es.lanczosBase().makeRandomVector(random_engine, n)); // set initial vector with random engine
		es.setMaxEigenvalues(10);																									// compute smallest 10 eigenvalues and eigenvectors. the default unlimited 
		es.setOrthogonalizingVectors({});																					// the default is {} empty
		es.setReorthogonalizeInterval(1);																					// the default is 1
		es.setReserveSize(128);																										// reserve size of lanczos vector default is 128
	}

	es.compute();

	// output results

	RealVectorType eivals = es.eigenvalues();

	std::cout << "matrix height : " << es.matrixHeight() << std::endl;
	std::cout << "iterations : " << es.iterations() << std::endl;
	std::cout << "subspace rank : " << es.lanczosvectors().size() << std::endl;

	std::cout << "eigen values : \n"
						<< eivals << std::endl;
	std::cout << "log :\n";
	for (auto &str : es.log()){
		std::cout << str << std::endl;
	}

	// output convergence log
	if(0){
		std::cout << "convergenceLog :\n";
		for (auto &edges_ : es.convergenceLog()){
			auto &idx = edges_.first;
			auto &edges = edges_.second;
			std::cout << "## " << idx << " ##" << std::endl;
			for (auto &edge : edges){
				std::cout << edge << std::endl;
			}
			std::cout << std::endl;
		}
	}
	
	

	return 0;
}
