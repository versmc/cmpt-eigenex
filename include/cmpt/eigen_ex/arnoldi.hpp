#pragma once


#include <limits>
#include <cmath>
#include <random>
#include <complex>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <functional>


#include "Eigen/Core"
#include "Eigen/Eigenvalues"

#include "cmpt/eigen_ex/random.hpp"
#include "cmpt/eigen_ex/lanczos.hpp"		// for type ailiases of DefaultTolerance and ArnoldiException



namespace cmpt {
	namespace EigenEx {

		

		using ArnoldiException = EigenEx::LanczosException;


		/// <summary>
		/// This class generates the basis of Krylov subspace,
		/// h_ij						// matrix elements in Krylov subspace
		/// arnoldivectors	// the orthogonal basis
		/// </summary>
		template<class Scalar_>
		class ArnoldiBase {
		public:
			// type alias
			using Index = Eigen::Index;
			using Scalar = Scalar_;
			using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
			using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
			using RealVectorType = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
			using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
			using ScalarDistribution = typename NormalDistributionGen<Scalar>::Type;
			using VectorDistribution = typename EigenEx::VectorDistribution<ScalarDistribution>;
			using MatMulFunction = std::function<void(const Scalar*, Scalar*)>;






			/// <summary>
			/// make random normalized vector
			/// you can use this to set initial vector
			/// </summary>
			template<class URBG>
			static VectorType makeRandomVector(URBG& g, Index size) {

				VectorDistribution vdist(
					ScalarDistribution(),	// normal distribution for each value
					size,					// size of vector
					true					// normalize ON
				);

				return vdist(g);

			}


		protected:

			/// <summary>
			/// Schmidt orthogonalization
			/// v_ortho must be normalized
			/// </summary>
			static void orthogonalize(VectorType& v_target, const VectorType& v_ortho) {
				Scalar temp = v_ortho.dot(v_target);
				v_target -= temp * v_ortho;
			}


		protected:

			Index reserveSize_;		// initial reserved size of arnoldi_vectors, eigen_vectors

			std::vector<VectorType> orthogonalizingVectors_;	// vector for reorthogonalizing in addition to arnoldivectors, this vectors must be orthogonalized and normalized
			MatMulFunction matrixMultiplication_;		// operation that matrix multiplication to vector, opetate(in,out) <--> |out> = H |in> 
			Scalar eigenvalueShift_;
			Index matrixHeight_;
			VectorType initialVector_;			// initial vector, this is not orthogonalized by vectors_for_ortho, and is not normalized
			RealScalar threshold_;

		public:
			// accessors for parameters of settings of arnoldi computing

			Index reserveSize()const { return reserveSize_; }
			ArnoldiBase& setReserveSize(Index resSize) { reserveSize_ = resSize; return *this; }

			const std::vector<VectorType>& orthogonalizingVectors()const { return orthogonalizingVectors_; }
			std::vector<VectorType>& refOrthogonalizingVectors() { return orthogonalizingVectors_; }
			ArnoldiBase& setOrthogonalizingVectors(const std::vector<VectorType>& orthoVec) {
				orthogonalizingVectors_ = orthoVec;
				return *this;
			}
			ArnoldiBase& setOrthogonalizingVectors(std::vector<VectorType>&& orthoVec) {
				orthogonalizingVectors_.swap(orthoVec);
				return *this;
			}

			const MatMulFunction& matrixMultiplication()const { return matrixMultiplication_; }
			ArnoldiBase& setMatrixMultiplication(const MatMulFunction& matmul, Index height) {
				matrixMultiplication_ = matmul;
				matrixHeight_ = height;
				return *this;
			}
			ArnoldiBase& setMatrixMultiplication(MatMulFunction&& matmul, Index height) {
				std::swap(matrixMultiplication_, matmul);
				matrixHeight_ = height;
				return *this;
			}
			Index matrixHeight()const { return matrixHeight_; }

			Scalar eigenvalueShift()const { return eigenvalueShift_; }
			ArnoldiBase& setEigenvalueShift(Scalar eishift) { eigenvalueShift_ = eishift; return *this; }



			const VectorType& initialVector()const {
				return initialVector_;
			}
			ArnoldiBase& setInitialVector(const VectorType& inivec) {
				initialVector_ = inivec;
				return *this;
			}
			ArnoldiBase& setInitialVector(VectorType&& inivec) {
				initialVector_ = std::move(inivec);
				return *this;
			}


			// set  initial vector whose size is rows and contents are random (fixed seed)
			ArnoldiBase& setInitialVector() {
				std::mt19937 rengine;
				setInitialVector(makeRandomVector(rengine, matrixHeight_));
				return *this;
			}

			RealScalar threshold()const {
				return threshold_;
			}
			ArnoldiBase& setThreshold(RealScalar thre) {
				threshold_ = thre;
				return *this;
			}



		protected:
			// variables of computed datas

			Index iterations_;		// height of tridiagonal Matrix == the number of gained eigenvectors

			std::vector<VectorType> arnoldivectors_;	// arnoldi vector, [0] is setted with set_initial_vector(...); and the other is setted with compute(...)

			VectorType v_;						// vectors to compute next arnoldivector
			RealScalar residue_;			// 
			std::vector<std::vector<Scalar>> h_;

		public:
			// accessors for computed datas

			Index iterations()const { return iterations_; }
			const std::vector<VectorType>& arnoldivectors()const { return arnoldivectors_; }
			const std::vector<std::vector<Scalar>>& h()const { return h_; }

			RealScalar residue()const {
				return residue_;
			}

		public:
			ArnoldiBase() { setAllSettingsDefault(); }


			/// <summary>
			/// set default settings of arnoldi computation
			/// this function DOESN'T clear computed data
			/// </summary>
			ArnoldiBase& setAllSettingsDefault() {

				setReserveSize(128);
				setOrthogonalizingVectors(std::vector<VectorType>());
				setMatrixMultiplication([](const Scalar* in, Scalar* out) {}, 0);
				setEigenvalueShift(0.0);
				setInitialVector();
				setThreshold(DefaultTolerance<RealScalar>::value());

				return *this;
			}

			/// <summary>
			/// this function clears arnoldivectors, alpha, beta, and so on
			/// not clears settings
			/// </summary>
			void clearArnoldiSteps() {
				iterations_ = 0;
				arnoldivectors_.clear();
				h_.clear();
				v_.resize(0);
			}


			/// <summary>
			/// clear all data, and set all settings default
			/// </summary>
			void clear() {
				clearArnoldiSteps();
				setAllSettingsDefault();
			}


			/// <summary>
			/// 初期ベクトルから arnoldivectors[0] を設定する
			/// ノルムがゼロになることで初期化できない場合は設定されない
			/// </summary>
			void setInitialArnoldivector() {
				if (matrixHeight_ < 0) {
					throw ArnoldiException("matrixHeight_ < 0");
				}
				if (matrixHeight_ != initialVector_.size()) {
					setInitialVector();
				}

				// memory allocation
				arnoldivectors_.resize(1);

				// set first arnoldi vectors
				arnoldivectors_[0] = initialVector_;
				for (auto& v_o : orthogonalizingVectors_) {
					orthogonalize(arnoldivectors_.back(), v_o);
				}
				RealScalar nrm = arnoldivectors_.back().norm();
				if (nrm < threshold_) {
					arnoldivectors_.resize(0);
				}
				else {
					arnoldivectors_.back().normalize();
				}

			}

		public:


			/// <summary>
			/// Krylov 部分空間の次元が上限に達しているかどうかを判定する
			/// </summary>
			bool arnoldiStepIsUtmost()const {
				if (arnoldivectors_.size() == 0) {
					return false;
				}
				if (arnoldivectors_.size() == static_cast<std::size_t>(matrixHeight_)) {
					return true;
				}
				if (residue_ <= threshold_) {
					return true;
				}
				return false;
			}



			/// <summary>
			/// arnoldi ステップを1回おこなう。
			/// 
			/// 初回は
			/// arnoldivector_[0]
			/// 
			/// h_[0] (hess(0,0), hess(1,0))
			/// arnoldivector_[1]
			/// 
			/// 次回以降は
			/// h_[0] (hess(0,k), hess(1,k), ..., hess(k+1,k))
			/// arnoldivector_[k+1]
			/// 
			/// が設定される
			/// 
			/// 初回時初期ベクトルのサイズが不正である場合初期ベクトルが乱数で設定される。
			/// arnoldi ステップをこれ以上行うことができない場合(直交空間が残っていない場合) ステップは行われず false を返す
			/// その他の場合 true を返す
			/// この仕様は while(...) で使うためのもの
			/// </summary>
			bool updateArnoldiSteps() {
				if (matrixHeight_ <= 0) {
					return false;
				}
				if (!matrixMultiplication_) {
					return false;
				}


				if (arnoldivectors_.size() == 0) {
					// 初回は arnoldivectors_[0]、h(0,0), v_ を設定

					// arnoldivector[0] を設定
					setInitialArnoldivector();	// set arnoldivectors_[0]
					if (arnoldivectors_.size() == 0) {
						return false;
					}


					// v_=A*q[0]
					v_.resize(matrixHeight_);
					matrixMultiplication_(arnoldivectors_[0].data(), v_.data());
					if (eigenvalueShift_ != RealScalar(0.0)) {
						v_ += eigenvalueShift_ * arnoldivectors_[0];
					}
					for (auto& v_ortho : orthogonalizingVectors_) {
						orthogonalize(v_, v_ortho);
					}

					// set h(0,0)={q[0]|v_}, v_=v_-h(0,0)|q[0]}
					h_.resize(1);
					h_[0].resize(2);
					h_[0][0] = (arnoldivectors_[0].adjoint() * v_)(0, 0);
					v_ = v_ - h_[0][0] * arnoldivectors_[0];
					h_[0][1] = Scalar(0.0);

					residue_ = v_.norm();

					// iteration and return true
					++iterations_;
					return true;
				}
				else {
					// 2回目以降は arnoldivectors_[k]、h(k,k-1),h(0:k,k), v_ を設定

					if (arnoldiStepIsUtmost()) {
						return false;
					}

					Index k = arnoldivectors_.size();
					h_[k - 1].resize(k + 1);	// 念の為
					h_[k - 1][k] = Scalar(residue_);
					arnoldivectors_.resize(k + 1);
					arnoldivectors_[k] = Scalar(1.0 / h_[k - 1][k]) * v_;
					v_.resize(matrixHeight_);	// 念の為

					// v_=A*q[k]
					matrixMultiplication_(arnoldivectors_[k].data(), v_.data());
					if (eigenvalueShift_ != RealScalar(0.0)) {
						v_ += eigenvalueShift_ * arnoldivectors_[k];
					}
					for (auto& v_ortho : orthogonalizingVectors_) {
						orthogonalize(v_, v_ortho);
					}

					// set h(0,k),...,h(k,k), and orhtogonalize v_
					h_.resize(k + 1);
					h_[k].resize(k + 2);
					for (Index i = 0; i <= k; ++i) {
						h_[k][i] = (arnoldivectors_[i].adjoint() * v_)(0, 0);
						v_ = v_ - h_[k][i] * arnoldivectors_[i];
					}
					h_[k][k + 1] = Scalar(0.0);
					residue_ = v_.norm();

					++iterations_;
					return true;
				}

				return false;	// never called
			}

			/// <summary>
			/// arnoldivectors によって構成される行列を生成する
			/// 形状は (matrixHeight,height_of_hessenberg_matrix)
			/// </summary>
			MatrixType makeArnoldiMatrix()const {
				Index nr = matrixHeight_;
				Index nc = h_.size();
				if (nc > nr) {
					nc = nr;
				}
				MatrixType V(nr, nc);
				for (Index c = 0, nc = V.cols(); c < nc; ++c) {
					V.col(c) = arnoldivectors_[c];
				}
				return V;
			}


			/// <summary>
			/// 現在の arnoldi step で得られている Hessenberg matrix を構成して返す
			/// </summary>
			MatrixType makeHessenbergMatrix()const {
				Index hsize = h_.size();
				if (hsize > matrixHeight_) {
					hsize = matrixHeight_;
				}
				MatrixType hess = MatrixType::Zero(hsize, hsize);
				for (Index c = 0, nc = hess.cols(); c < nc; ++c) {
					Index nr = hess.rows();
					Index nr_ = h_[c].size();
					if (nr_ < nr) {
						nr = nr_;
					}
					for (Index r = 0; r < nr; ++r) {
						hess(r, c) = h_[c][r];
					}
				}
				return hess;
			}





		};


		/// <summary>
		/// eigen solver with Arnoldi
		/// </summary>
		template<class Scalar_>
		class ArnoldiEigenSolver {
		public:

			// type aliases
			using Index = Eigen::Index;
			using Scalar = Scalar_;
			using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
			using ComplexScalar = std::complex<RealScalar>;

			using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
			using RealVectorType = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
			using ComplexVectorType = Eigen::Matrix<ComplexScalar, Eigen::Dynamic, 1>;

			using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
			using RealMatrixType = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
			using ComplexMatrixType = Eigen::Matrix<ComplexScalar, Eigen::Dynamic, Eigen::Dynamic>;

			using ScalarDistribution = typename NormalDistributionGen<Scalar>::Type;
			using VectorDistribution = typename EigenEx::VectorDistribution<ScalarDistribution>;
			using MatMulFunction = std::function<void(const Scalar*, Scalar*)>;


			/// <summary>
			/// This class give the type of eigensolver
			/// Scalar is
			/// double or std::complex{double}
			/// </summary>
			template<class MatrixType_>
			class EigenSolverTraits {
			public:
				using MatrixType = MatrixType_;
				using Scalar = typename MatrixType::Scalar;
				using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
				using ComplexScalar = std::complex<RealScalar>;
				using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
				using RealVectorType = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
				using ComplexVectorType= Eigen::Matrix<ComplexScalar, Eigen::Dynamic, 1>;



				template<class T, class AlwaysBool>
				class Dummy {
				public:
					using Type = Eigen::EigenSolver<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
				};

				template<class T, class AlwaysBool>
				class Dummy<std::complex<T>, AlwaysBool> {
				public:
					using Type = Eigen::ComplexEigenSolver<Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic>>;
				};

				using Type = typename Dummy<Scalar, bool>::Type;

			};

			using DenseEigenSolver = typename EigenSolverTraits<MatrixType>::Type;	// eigensolver of hessenberg matrix 


			// header of log generation
			static std::string headERROR() { return std::string("ERROR     "); }
			static std::string headWARN() { return std::string("WARN      "); }
			static std::string headINFO() { return std::string("INFO      "); }
			static std::string headDEBUG() { return std::string("DEBUG     "); }


			// Index value for special case of minIterations, maxIterations
			static constexpr Index unlimited = -1;

			/// <summary>
			/// make random normalized vector
			/// you can use this to set initial vector
			/// </summary>
			template<class URBG>
			static VectorType makeRandomVector(URBG& g, Index size) {
				return ArnoldiBase<Scalar>::makeRandomVector(g, size);
			}

		protected:

			Index minIterations_;													// lower limit of arnoldi step, default is unlimited
			Index maxIterations_;													// upper limit of arnoldi step, default is unlimited
			RealScalar tolerance_;											// error to judge convergence
			std::vector<Index> indicesForConvergence_;		// indices of eigenvalue to judge its convergence
																									// the order of eigenvalue is descending by its abs


			Index maxEigenvalues_;						// the max number of eigenvalues and eigenvectors to compute
			bool computeEigenvectorsOn_;		// false means only eigenvalues will be computed, not eigenvectors, default true

		public:

			Index minIterations()const { return minIterations_; }
			ArnoldiEigenSolver& setMinIterations(Index miniter) {
				minIterations_ = miniter;
				return *this;
			}


			Index maxIterations()const { return maxIterations_; }
			ArnoldiEigenSolver& setMaxIterations(Index maxiter) {
				maxIterations_ = maxiter;
				return *this;
			}

			RealScalar tolerance()const { return tolerance_; }
			ArnoldiEigenSolver& setTolerance(RealScalar toler) {
				tolerance_ = toler;
				return *this;
			}


			const std::vector<Index>& indicesForConvergence()const {
				return indicesForConvergence_;
			}
			ArnoldiEigenSolver& setIndicesForConvergence(const std::vector<Index>& iCovs) {
				indicesForConvergence_ = iCovs;
				return *this;
			}

			Index maxEigenvalues()const { return maxEigenvalues_; }
			ArnoldiEigenSolver& setMaxEigenvalues(Index maxeivals) {
				maxEigenvalues_ = maxeivals;
				return *this;
			}

			Index computeEigenvectorsOn()const { return computeEigenvectorsOn_; }
			ArnoldiEigenSolver& setComputeEigenvectorsOn(bool cEivecOn) {
				computeEigenvectorsOn_ = cEivecOn;
				return *this;
			}



		protected:
			ArnoldiBase<Scalar> arnoldiBase_;

		public:	// transparent accessor for arnoldiBase

			const ArnoldiBase<Scalar>& arnoldiBase()const { return arnoldiBase_; }

			Index reserveSize()const { return arnoldiBase_.reserveSize(); }
			ArnoldiEigenSolver& setReserveSize(Index resSize) { arnoldiBase_.setReserveSize(resSize); return *this; }

			const std::vector<VectorType>& orthogonalizingVectors()const { return arnoldiBase_.orthogonalizingVectors(); }
			std::vector<VectorType>& refOrthogonalizingVectors() { return arnoldiBase_.refOrthogonalizingVectors(); }
			ArnoldiEigenSolver& setOrthogonalizingVectors(const std::vector<VectorType>& orthoVec) {
				arnoldiBase_.setOrthogonalizingVectors(orthoVec);
				return *this;
			}
			ArnoldiEigenSolver& setOrthogonalizingVectors(std::vector<VectorType>&& orthoVec) {
				arnoldiBase_.setOrthogonalizingVectors(std::move(orthoVec));
				return *this;
			}

			const MatMulFunction& matrixMultiplication()const { return arnoldiBase_.matrixMultiplication(); }
			ArnoldiEigenSolver& setMatrixMultiplication(const MatMulFunction& matmul, Index height) {
				arnoldiBase_.setMatrixMultiplication(matmul, height);
				return *this;
			}
			ArnoldiEigenSolver& setMatrixMultiplication(MatMulFunction&& matmul, Index height) {
				arnoldiBase_.setMatrixMultiplication(std::move(matmul), height);
				return *this;
			}
			Index matrixHeight()const { return arnoldiBase_.matrixHeight(); }

			Scalar eigenvalueShift()const { return arnoldiBase_.eigenvalueShift(); }
			ArnoldiEigenSolver& setEigenvalueShift(Scalar eishift) { arnoldiBase_.setEigenvalueShift(eishift); return *this; }



			const VectorType& initialVector()const {
				return arnoldiBase_.initialVector();
			}
			ArnoldiEigenSolver& setInitialVector(const VectorType& inivec) {
				arnoldiBase_.setInitialVector(inivec);
				return *this;
			}
			ArnoldiEigenSolver& setInitialVector(VectorType&& inivec) {
				arnoldiBase_.setInitialVector(std::move(inivec));
				return *this;
			}

			// set  initial vector whose size is rows and contents are random (fixed seed)
			ArnoldiEigenSolver& setInitialVector() {
				arnoldiBase_.setInitialVector();
				return *this;
			}

			RealScalar threshold()const { return arnoldiBase_.threshold(); }
			ArnoldiEigenSolver& setThreshold(RealScalar thre) {
				arnoldiBase_.setThreshold(thre);
				return *this;
			}


			Index iterations()const { return arnoldiBase_.iterations(); }
			const std::vector<VectorType>& arnoldivectors()const { return arnoldiBase_.arnoldivectors(); }



		protected:
			// computed datas

			ComplexVectorType eigenvalues_;	// finally, set as eigenvalues of matrixMultiplication
																// the order is descending order of its absolute value, also used for convergence judgement
																// during calculation, this variable is also used as eigenvalues of hessenberg matrix 
																// and used as judgement of convergence
			ComplexMatrixType eigenvectors_;					// eigenvectors of matrixMultiplication
			ComplexMatrixType eigenvectors_h_;				// eigenvectors of hessenberg matrix
			std::vector<std::string> log_;		// log of computing with header
			MatrixType hessenbergMatrix_;
			DenseEigenSolver des_;						// 現在 eigenvalues eigenvectors は別途メンバにあるのでこれはメンバとして残しておく必要があるのか?

			std::map<Index, std::vector<ComplexScalar>> convergenceLog_;	// 

		public:


			const ComplexVectorType& eigenvalues()const { return eigenvalues_; }
			const ComplexMatrixType& eigenvectors()const { return eigenvectors_; }
			const ComplexMatrixType& eigenvectors_h()const { return eigenvectors_h; }
			const std::vector<std::string>& log()const { return log_; }
			const MatrixType& hessenbergMatrix()const { return hessenbergMatrix_; }

			const DenseEigenSolver& des()const { return des_; }
			const std::map<Index, std::vector<RealScalar>>& convergenceLog()const { return convergenceLog_; };

		public:

			ArnoldiEigenSolver() { setAllSettingsDefault(); }

			/// <summary>
			/// set default settings of arnoldi computation
			/// this function DOESN'T clear computed data
			/// </summary>
			ArnoldiEigenSolver& setAllSettingsDefault() {
				setMinIterations(1);
				setMaxIterations(unlimited);
				setTolerance(DefaultTolerance<RealScalar>::value());
				setIndicesForConvergence(std::vector<Index>{0});
				setMaxEigenvalues(unlimited);
				setComputeEigenvectorsOn(true);

				arnoldiBase_.setAllSettingsDefault();

				return *this;
			}


			/// <summary>
			/// clear computed datas
			/// this function DOESN'T clear settings for computing
			/// </summary>
			ArnoldiEigenSolver& clearComputedData() {
				arnoldiBase_.clearArnoldiSteps();
				eigenvalues_.resize(0);
				eigenvectors_.resize(0, 0);
				log_.clear();
				convergenceLog_.clear();
				return *this;
			}


			/// <summary>
			/// initialize this class
			/// clear computed data and set all settings default
			/// </summary>
			ArnoldiEigenSolver& clear() {
				clearComputedData();
				setAllSettingsDefault();
				return *this;
			}


			/// <summary>
			/// this function execute arnoldi calculation from this state
			/// you can change settings and start arnoldi steps from current step
			/// matrixMultiplication_ must have not be changed before calling this function
			/// </summary>
			Index continueToCompute() {
				log_.push_back(headINFO() + "ArnoldiEigenSolver<ScalarType>::continueToCompute(...) was called");
				if (arnoldiBase_.arnoldivectors().size() == 0) {
					return compute();
				}

				// main calculation
				Index ret = mainCalculation_();
				log_.push_back(headINFO() + "ArnoldiEigenSolver<ScalarType>::compute(...) finish computing");
				return ret;

			}

			/// <summary>
			/// compute matrix diagonalization
			/// </summary>
			Index compute() {

				log_.push_back(headINFO() + "ArnoldiEigenSolver<ScalarType>::compute(...) was called");

				clearComputedData();

				// set initialVector if initial_vector is empty or invalid
				if (initialVector().size() != matrixHeight()) {
					log_.push_back(headINFO() + "in compute(), initial_vector is empty or invalid, then set at random");
					setInitialVector();	// set initial vector at random
				}


				// main calculation

				Index ret = mainCalculation_();

				log_.push_back(headINFO() + "ArnoldiEigenSolver<ScalarType>::compute(...) finish computing");
				return ret;
			}



			Index mainCalculation_() {
				// do arnoldi steps until convergence or achieve utmost state
				Index set_initialvector_is_fail = false;
				while (true) {

					// check convergence and utmost state
					updateConvergenceLog_();


					// judge convergence and break
					{
						if (set_initialvector_is_fail) {
							log_.push_back(headINFO() + "initial arnoldivector generation fail");
							break;
						}
						if (arnoldiBase_.arnoldiStepIsUtmost()) {
							log_.push_back(headINFO() + "arnoldi steps finished with threshold");
							log_.push_back(headINFO() + "arnoldi steps achieved full of Krylov subspace");
							break;
						}
						if (arnoldiBase_.iterations() >= minIterations()) {
							if (arnoldiBase_.iterations() == maxIterations()) {
								log_.push_back(headWARN() + "arnoldi steps achieved maxIterations");
								break;
							}
							if (isConverged_()) {
								log_.push_back(headINFO() + "arnoldi steps converged with tolerance");
								break;
							}
						}
					}

					// arnoldi step
					arnoldiBase_.updateArnoldiSteps();
					if (arnoldivectors().size() == 0) {
						set_initialvector_is_fail = true;
					}


					// diagonalize tridiagonal matrix

					hessenbergMatrix_ = arnoldiBase_.makeHessenbergMatrix();
					if (hessenbergMatrix_.rows() == 0) {
						eigenvalues_.resize(0);
						eigenvectors_h_.resize(0, 0);
					}
					else {
						des_.compute(hessenbergMatrix_);
						eigenvalues_ = des_.eigenvalues();
						auto order = compute_sorted_indices(
							eigenvalues_.data(),
							eigenvalues_.data() + eigenvalues_.size(),
							[](const ComplexScalar& a, const ComplexScalar& b)->bool {
								return std::abs(a) > std::abs(b);
							}
						);
						EigenEx::cwiseShuffle(eigenvalues_, order);
						eigenvectors_h_ = des_.eigenvectors();
						EigenEx::rowwiseShuffle(eigenvectors_h_, order);
					}
				}


				// back eigen value to original one
				Index eivalsize = eigenvalues_.size();
				if (maxEigenvalues_ != unlimited) {
					if (maxEigenvalues_ < eivalsize) {
						eivalsize = maxEigenvalues_;
					}
				}
				ComplexVectorType eivals_temp = eigenvalues_;
				eigenvalues_.resize(eivalsize);
				for (Index k = 0; k < eivalsize; ++k) {
					eigenvalues_[k] = eivals_temp[k] - arnoldiBase_.eigenvalueShift();
				}

				// get eigen_vector[M][N] from eigen_vector_h[M][M]
				if (computeEigenvectorsOn_) {
					MatrixType arnoldiMatrix = arnoldiBase_.makeArnoldiMatrix();

					eigenvectors_ = ComplexMatrixType::Zero(matrixHeight(), eivalsize);
					for (Index r = 0, nr = eigenvectors_.rows(); r < nr; ++r) {
						for (Index c = 0, nc = eigenvectors_.cols(); c < nc; ++c) {
							for (Index j = 0, nj = eigenvectors_h_.rows(); j < nj; ++j) {
								eigenvectors_(r, c) += arnoldiMatrix(r, j) * eigenvectors_h_(j, c);
							}
						}
					}
				}
				else {
					eigenvectors_.resize(0, 0);
				}
				return 0;
			}





		protected:
			// helper functions for arnoldi calculation

			/// <summary>
			/// this function returns index for eigenvalues or eigenvectors in range [0,n)
			/// negative i count from end index
			/// when i is invalid, returns -1
			/// </summary>
			static Index getFormalIndex(Index i, Index n) {
				if (-n <= i && i < 0) {
					return n - (-i - 1) % n - 1;
				}
				else if (0 <= i && i < n) {
					return i % n;
				}
				else {
					return -1;
				}
			}

			/// <summary>
			/// append convergence log with eigenvalues_
			/// eigenvalues_ must be computed before this function call
			/// </summary>
			void updateConvergenceLog_() {
				for (auto& indexForConvergence : indicesForConvergence_) {
					Index i = getFormalIndex(indexForConvergence, eigenvalues_.size());
					if (i < 0) {
						continue;
					}
					convergenceLog_[indexForConvergence].push_back(
						eigenvalues_[i]
					);
				}
			}

			/// <summary>
			/// this function judge convergence with current convergenceLog_
			/// </summary>
			bool isConverged_() {
				bool is_converged = true;
				if (eigenvalues_.size() < 2) {
					is_converged = false;
					return is_converged;
				}
				RealScalar scale = std::abs(eigenvalues_[0] - eigenvalues_[eigenvalues_.size() - 1]);
				for (auto& idxFroConvergence : indicesForConvergence_) {
					auto itr = convergenceLog_.find(idxFroConvergence);
					if (itr == convergenceLog_.end()) {
						is_converged = false;

						break;
					}
					auto& edge = itr->second;
					if (edge.size() < 2) {
						is_converged = false;
						break;
					}
					ComplexScalar cur = edge[edge.size() - 1];
					ComplexScalar old = edge[edge.size() - 2];
					if (std::abs((cur - old) / scale) > tolerance_) {
						is_converged = false;
						break;
					}
				}
				return is_converged;
			}


		public:


			// return the number of error log
			Index hasERROR()const {
				Index count = 0;
				for (const auto& str : log_) {
					if (str.find(headERROR()) == 0) {
						++count;
					}
				}
				return count;
			}

			// return the number of warning log
			Index hasWARN()const {
				Index count = 0;
				for (const auto& str : log_) {
					if (str.find(headWARN()) == 0) {
						++count;
					}
				}
				return count;
			}




		};


	}
}








