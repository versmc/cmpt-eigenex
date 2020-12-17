#pragma once


/// <summary>
/// This header defines classes for lanczos method
/// 
/// # Policy
/// 
/// ## Coding style
/// header only
/// Eigen-like
/// 
/// ## Dependency
/// c++11					lambda, random, constexpr
/// Eigen3				Matrix class, SelfAdjointEigenSolver
/// 
/// # TODO
/// DegenerateEigenSolver: eigensolver for Degeneracy
/// 
/// </summary>



#include <limits>
#include <cmath>
#include <random>
#include <complex>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <functional>


#include <array>						// for sample code
#include <iostream>					// for sample code
#include <iomanip>					// for sample code

#include "Eigen/Core"
#include "Eigen/Eigenvalues"


#include "cmpt/eigen_ex/random.hpp"		// for random distribution for objects in Eigen






namespace cmpt {

	namespace EigenEx {

		/// <summary>
		/// this class gives a default value to judge the covergence
		/// 
		/// Scalar is double -> 1.0e-12
		/// Scalar is float  -> 1.0e-4
		/// others           -> 1.0e-12
		/// 
		/// </summary>
		template<class Scalar>
		class DefaultTolerance {
		protected:
			template<class S, class AlwaysBool>
			struct Dummy {
				static constexpr S value() { return 1.0e-12; }
			};

			template<class AlwaysBool>
			struct Dummy<float, AlwaysBool> {
				static constexpr float value() { return 1.0e-4; }
			};

			template<class AlwaysBool>
			struct Dummy<double, AlwaysBool> {
				static constexpr double value() { return 1.0e-12; }
			};


		public:
			static constexpr Scalar value() { return Dummy<Scalar, bool>::value(); }
		};



		/// <summary>
		/// exception class used in lanczos
		/// </summary>
		class LanczosException :public std::runtime_error {
		public:
			LanczosException(const char* _Message)
				: runtime_error(_Message)
			{}
		};


		/// <summary>
		/// This class generate the basis of Krylov subspace
		/// Lanczos vector	(orthogonal basis of Krylov subspace)
		/// alpha						(diagonal elements)
		/// beta						(sub-diagonal elements)
		/// </summary>
		template<class Scalar_>
		class LanczosBase {
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

			Index reserveSize_;		// initial reserved size of lanczos_vectors, eigen_vectors

			std::vector<VectorType> orthogonalizingVectors_;	// vector for reorthogonalizing in addition to u[0]...u[k-2], this vectors must be orthogonalized and normalized
			MatMulFunction matrixMultiplication_;		// operation that matrix multiplication to vector, opetate(in,out) <--> |out> = H |in> 
			RealScalar eigenvalueShift_;
			Index matrixHeight_;
			Index reorthogonalizeInterval_;			// reorthogonalize Lanczos vectors u[k+1] by old Lanczos vector u[0],...,u[k] and vectors_for_ortho[]
			VectorType initialVector_;			// initial vector, this is not orthogonalized by vectors_for_ortho, and is not normalized
			RealScalar threshold_;

		public:
			// accessors for parameters of settings of lanczos computing

			Index reserveSize()const { return reserveSize_; }
			LanczosBase& setReserveSize(Index resSize) { reserveSize_ = resSize; return *this; }

			const std::vector<VectorType>& orthogonalizingVectors()const { return orthogonalizingVectors_; }
			std::vector<VectorType>& refOrthogonalizingVectors() { return orthogonalizingVectors_; }
			LanczosBase& setOrthogonalizingVectors(const std::vector<VectorType>& orthoVec) {
				orthogonalizingVectors_ = orthoVec;
				return *this;
			}
			LanczosBase& setOrthogonalizingVectors(std::vector<VectorType>&& orthoVec) {
				orthogonalizingVectors_.swap(orthoVec);
				return *this;
			}

			const MatMulFunction& matrixMultiplication()const { return matrixMultiplication_; }
			LanczosBase& setMatrixMultiplication(const MatMulFunction& matmul, Index height) {
				matrixMultiplication_ = matmul;
				matrixHeight_ = height;
				return *this;
			}
			LanczosBase& setMatrixMultiplication(MatMulFunction&& matmul, Index height) {
				std::swap(matrixMultiplication_, matmul);
				matrixHeight_ = height;
				return *this;
			}
			Index matrixHeight()const { return matrixHeight_; }

			RealScalar eigenvalueShift()const { return eigenvalueShift_; }
			LanczosBase& setEigenvalueShift(RealScalar eishift) { eigenvalueShift_ = eishift; return *this; }

			bool reorthogonalizeInterval()const { return reorthogonalizeInterval_; }
			LanczosBase& setReorthogonalizeInterval(Index reorthoInterval) {
				reorthogonalizeInterval_ = reorthoInterval;
				return *this;
			}

			const VectorType& initialVector()const {
				return initialVector_;
			}
			LanczosBase& setInitialVector(const VectorType& inivec) {
				initialVector_ = inivec;
				return *this;
			}
			LanczosBase& setInitialVector(VectorType&& inivec) {
				initialVector_ = std::move(inivec);
				return *this;
			}


			// set  initial vector whose size is rows and contents are random (fixed seed)
			LanczosBase& setInitialVector() {
				std::mt19937 rengine;
				setInitialVector(makeRandomVector(rengine, matrixHeight_));
				return *this;
			}

			RealScalar threshold()const {
				return threshold_;
			}
			LanczosBase& setThreshold(RealScalar thre) {
				threshold_ = thre;
				return *this;
			}



		protected:
			// variables of computed datas

			Index iterations_;		// height of tridiagonal Matrix == the number of gained eigenvectors

			std::vector<VectorType> lanczosvectors_;	// Lanczos vector, [0] is setted with set_initial_vector(...); and the other is setted with compute(...)

			VectorType v_;						// used as temporal array[N]
			std::vector<RealScalar> alpha_;		// diagonal element of tridiagonal matrix
			std::vector<RealScalar> beta_;		// non-diagonal element of tridiagonal matrix


		public:
			// accessors for computed datas

			Index iterations()const { return iterations_; }
			const std::vector<VectorType>& lanczosvectors()const { return lanczosvectors_; }
			const std::vector<RealScalar>& alpha()const { return alpha_; }
			const std::vector<RealScalar>& beta()const { return beta_; }



		public:
			LanczosBase() { setAllSettingsDefault(); }


			/// <summary>
			/// set default settings of lanczos computation
			/// this function DOESN'T clear computed data
			/// </summary>
			LanczosBase& setAllSettingsDefault() {

				setReserveSize(128);
				setOrthogonalizingVectors(std::vector<VectorType>());
				setMatrixMultiplication([](const Scalar* in, Scalar* out) {}, 0);
				setEigenvalueShift(0.0);
				setReorthogonalizeInterval(1);
				setInitialVector();
				setThreshold(DefaultTolerance<RealScalar>::value());

				return *this;
			}

			/// <summary>
			/// this function clears lanczosvectors, alpha, beta, and so on
			/// not clears settings
			/// </summary>
			void clearLanczosSteps() {
				iterations_ = 0;
				lanczosvectors_.clear();
				alpha_.clear();
				beta_.clear();
				v_.resize(0);
			}


			/// <summary>
			/// clear all data, and set all settings default
			/// </summary>
			void clear() {
				clearLanczosSteps();
				setAllSettingsDefault();
			}


			/// <summary>
			/// initialize lanczosVectors_[0] initialVector
			/// if the norm initialVector is zero, lanczosVectors_[0] will not be initialized
			/// </summary>
			void setInitialLanczosvector() {
				if (matrixHeight_ < 0) {
					throw LanczosException("matrixHeight_ < 0");
				}
				if (matrixHeight_ != initialVector_.size()) {
					setInitialVector();
				}

				// memory allocation
				lanczosvectors_.resize(1);

				// set first lanczos vectors
				lanczosvectors_[0] = initialVector_;
				for (auto& v_o : orthogonalizingVectors_) {
					orthogonalize(lanczosvectors_.back(), v_o);
				}
				RealScalar nrm = lanczosvectors_.back().norm();
				if (nrm < threshold_) {
					lanczosvectors_.resize(0);
				}
				else {
					lanczosvectors_.back().normalize();
				}

			}

		public:


			/// <summary>
			/// judge the dimension of Krylov subspace is utmost or not
			/// </summary>
			bool lanczosStepIsUtmost()const {
				if (lanczosvectors_.size() == static_cast<std::size_t>(matrixHeight_)) {
					return true;
				}
				else if (beta_.size() > 0) {
					if (beta_.back() <= threshold_) {
						return true;
					}
					else {
						return false;
					}
				}

				else {
					return false;
				}
			}



			/// <summary>
			/// do Lanczos step one time
			/// 
			/// 
			/// At first, this function will set
			/// lanczosvector_[0]
			/// alpha_[0]
			/// v_=H*lanczosvector_[0]
			/// 
			/// from second, this function will set 
			/// beta_[k-1]
			/// lanczosvector_[k]
			/// alpha_[k]
			/// v_=H*lanczosvector_[k]
			/// 
			/// 
			/// If initial vector is invalid, lanczosVector[0] set with random number
			/// If Lanczos step is utmost, no step is done and this function return false, otherwise, this function return true.
			/// this is useful with while() syntax
			/// </summary>
			bool updateLanczosSteps() {
				if (matrixHeight_ <= 0) {
					return false;
				}
				if (!matrixMultiplication_) {
					return false;
				}
				if (lanczosvectors_.size() == 0) {
					// lanczosvector[0], alpha[0], v を設定

					setInitialLanczosvector();	// set lanczosvectors_[0]
					if (lanczosvectors_.size() == 0) {
						return false;
					}

					v_.resize(matrixHeight_);

					// |v> = (H+eigenvalue_shift)|u[0]>
					matrixMultiplication_(lanczosvectors_[0].data(), v_.data());
					if (eigenvalueShift_ != RealScalar(0.0)) {
						v_ += eigenvalueShift_ * lanczosvectors_[0];
					}

					// alpha[0] = <u[0]|v>
					alpha_.push_back(std::real(Scalar(lanczosvectors_[0].dot(v_)))); // 何故か一時変数を作らないとキャストできない

					return true;
				}
				else {
					Index k = lanczosvectors_.size() - 1;
					// set u[k+1]
					lanczosvectors_.push_back(VectorType(matrixHeight_));
					if (k == 0) {
						lanczosvectors_[k + 1] = v_ - alpha_[k] * lanczosvectors_[k];
					}
					else {
						lanczosvectors_[k + 1] = v_ - alpha_[k] * lanczosvectors_[k] - beta_[k - 1] * lanczosvectors_[k - 1];
					}

					//reorthogonalize u[k+1]
					if (reorthogonalizeInterval_ > 0) {
						Index kmod = (lanczosvectors_.size() - 1) % reorthogonalizeInterval_;
						Index nk = lanczosvectors_.size();

						// orthogonalize by old lanczosvectors
						for (Index kk = kmod, nkk = lanczosvectors_.size() - 1; kk < nkk; kk += reorthogonalizeInterval_) {
							orthogonalize(lanczosvectors_.back(), lanczosvectors_[kk]);
						}

						// orthogonalize by vectors for orthogonalizing
						if (kmod == 0) {
							for (auto& v_o : orthogonalizingVectors_) {
								orthogonalize(lanczosvectors_.back(), v_o);
							}
						}
					}

					//set beta[k] = |u[k+1]|
					beta_.push_back(lanczosvectors_[k + 1].norm());



					if (beta_[k] <= threshold_) {
						//beta_.pop_back();
						lanczosvectors_.pop_back();
						return false;
					}
					else {
						lanczosvectors_[k + 1] /= beta_[k];

						// |v> = (H+eigenvalue_shift)|u[k]>
						matrixMultiplication_(lanczosvectors_[k + 1].data(), v_.data());
						if (eigenvalueShift_ != RealScalar(0.0)) {
							v_ += eigenvalueShift_ * lanczosvectors_[k + 1];
						}

						// alpha[k] = <u[k]|v>
						alpha_.push_back(std::real(lanczosvectors_[k + 1].dot(v_)));

						++iterations_;
						return true;
					}


				}

			}



		};



		/// <summary>
		/// eigen solver with Lanczos method
		/// </summary>
		template<class Scalar_>
		class LanczosEigenSolver {
		public:

			// type aliases
			using Index = Eigen::Index;
			using Scalar = Scalar_;
			using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
			using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
			using RealVectorType = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
			using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
			using RealMatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
			using ScalarDistribution = typename NormalDistributionGen<Scalar>::Type;
			using VectorDistribution = typename EigenEx::VectorDistribution<ScalarDistribution>;
			using MatMulFunction = std::function<void(const Scalar*, Scalar*)>;


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
				return LanczosBase<Scalar>::makeRandomVector(g, size);
			}

		protected:

			Index minIterations_;													// lower limit of Lanczos step, default is unlimited
			Index maxIterations_;													// upper limit of Lanczos step, default is unlimited
			RealScalar tolerance_;											// error to judge convergence
			std::vector<Index> indicesForConvergence_;		// indices of eigenvalue to judge its convergence
																									// negative value means count from maximum eignvalues

			Index maxEigenvalues_;						// the max number of eigenvalues and eigenvectors to compute
			bool computeEigenvectorsOn_;		// false means only eigenvalues will be computed, not eigenvectors, default true

		public:

			Index minIterations()const { return minIterations_; }
			LanczosEigenSolver& setMinIterations(Index miniter) {
				minIterations_ = miniter;
				return *this;
			}


			Index maxIterations()const { return maxIterations_; }
			LanczosEigenSolver& setMaxIterations(Index maxiter) {
				maxIterations_ = maxiter;
				return *this;
			}

			RealScalar tolerance()const { return tolerance_; }
			LanczosEigenSolver& setTolerance(RealScalar toler) {
				tolerance_ = toler;
				return *this;
			}


			const std::vector<Index>& indicesForConvergence()const {
				return indicesForConvergence_;
			}
			LanczosEigenSolver& setIndicesForConvergence(const std::vector<Index>& iCovs) {
				indicesForConvergence_ = iCovs;
				return *this;
			}

			Index maxEigenvalues()const { return maxEigenvalues_; }
			LanczosEigenSolver& setMaxEigenvalues(Index maxeivals) {
				maxEigenvalues_ = maxeivals;
				return *this;
			}

			Index computeEigenvectorsOn()const { return computeEigenvectorsOn_; }
			LanczosEigenSolver& setComputeEigenvectorsOn(bool cEivecOn) {
				computeEigenvectorsOn_ = cEivecOn;
				return *this;
			}



		protected:
			LanczosBase<Scalar> lanczosBase_;

		public:	// transparent accessor for lanczosBase

			const LanczosBase<Scalar>& lanczosBase()const { return lanczosBase_; }

			Index reserveSize()const { return lanczosBase_.reserveSize(); }
			LanczosEigenSolver& setReserveSize(Index resSize) { lanczosBase_.setReserveSize(resSize); return *this; }

			const std::vector<VectorType>& orthogonalizingVectors()const { return lanczosBase_.orthogonalizingVectors(); }
			std::vector<VectorType>& refOrthogonalizingVectors() { return lanczosBase_.refOrthogonalizingVectors(); }
			LanczosEigenSolver& setOrthogonalizingVectors(const std::vector<VectorType>& orthoVec) {
				lanczosBase_.setOrthogonalizingVectors(orthoVec);
				return *this;
			}
			LanczosEigenSolver& setOrthogonalizingVectors(std::vector<VectorType>&& orthoVec) {
				lanczosBase_.setOrthogonalizingVectors(std::move(orthoVec));
				return *this;
			}

			const MatMulFunction& matrixMultiplication()const { return lanczosBase_.matrixMultiplication(); }
			LanczosEigenSolver& setMatrixMultiplication(const MatMulFunction& matmul, Index height) {
				lanczosBase_.setMatrixMultiplication(matmul, height);
				return *this;
			}
			LanczosEigenSolver& setMatrixMultiplication(MatMulFunction&& matmul, Index height) {
				lanczosBase_.setMatrixMultiplication(std::move(matmul), height);
				return *this;
			}
			Index matrixHeight()const { return lanczosBase_.matrixHeight(); }

			RealScalar eigenvalueShift()const { return lanczosBase_.eigenvalueShift(); }
			LanczosEigenSolver& setEigenvalueShift(RealScalar eishift) { lanczosBase_.setEigenvalueShift(eishift); return *this; }

			bool reorthogonalizeInterval()const { return lanczosBase_.reorthogonalizeInterval(); }
			LanczosEigenSolver& setReorthogonalizeInterval(Index reorthoInterval) {
				lanczosBase_.setReorthogonalizeInterval(reorthoInterval);
				return *this;
			}

			const VectorType& initialVector()const {
				return lanczosBase_.initialVector();
			}
			LanczosEigenSolver& setInitialVector(const VectorType& inivec) {
				lanczosBase_.setInitialVector(inivec);
				return *this;
			}
			LanczosEigenSolver& setInitialVector(VectorType&& inivec) {
				lanczosBase_.setInitialVector(std::move(inivec));
				return *this;
			}

			// set  initial vector whose size is rows and contents are random (fixed seed)
			LanczosEigenSolver& setInitialVector() {
				lanczosBase_.setInitialVector();
				return *this;
			}

			RealScalar threshold()const { return lanczosBase_.threshold(); }
			LanczosEigenSolver& setThreshold(RealScalar thre) {
				lanczosBase_.setThreshold(thre);
				return *this;
			}


			Index iterations()const { return lanczosBase_.iterations(); }
			const std::vector<VectorType>& lanczosvectors()const { return lanczosBase_.lanczosvectors(); }
			const std::vector<RealScalar>& alpha()const { return lanczosBase_.alpha(); }
			const std::vector<RealScalar>& beta()const { return lanczosBase_.beta(); }


		protected:
			// computed datas

			RealVectorType eigenvalues_;
			MatrixType eigenvectors_;
			std::vector<std::string> log_;													// log of computing with header
			Eigen::SelfAdjointEigenSolver<RealMatrixType> es_tri_;	// to compute tridiagonal matrix
			std::map<Index, std::vector<RealScalar>> convergenceLog_;	// 

		public:


			const RealVectorType& eigenvalues()const { return eigenvalues_; }
			const MatrixType& eigenvectors()const { return eigenvectors_; }
			const std::vector<std::string>& log()const { return log_; }
			const Eigen::SelfAdjointEigenSolver<RealMatrixType>& es_tri()const { return es_tri_; }
			const std::map<Index, std::vector<RealScalar>>& convergenceLog()const { return convergenceLog_; };

		public:

			LanczosEigenSolver() { setAllSettingsDefault(); }

			/// <summary>
			/// set default settings of lanczos computation
			/// this function DOESN'T clear computed data
			/// </summary>
			LanczosEigenSolver& setAllSettingsDefault() {
				setMinIterations(1);
				setMaxIterations(unlimited);
				setTolerance(DefaultTolerance<RealScalar>::value());
				setIndicesForConvergence(std::vector<Index>{0});
				setMaxEigenvalues(unlimited);
				setComputeEigenvectorsOn(true);

				lanczosBase_.setAllSettingsDefault();

				return *this;
			}


			/// <summary>
			/// clear computed datas
			/// this function DOESN'T clear settings for computing
			/// </summary>
			LanczosEigenSolver& clearComputedData() {
				lanczosBase_.clearLanczosSteps();
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
			LanczosEigenSolver& clear() {
				clearComputedData();
				setAllSettingsDefault();
				return *this;
			}


			/// <summary>
			/// this function execute Lanczos calculation from this state
			/// you can change settings and start lanczos steps from current step
			/// matrixMultiplication_ must have not be changed before calling this function
			/// </summary>
			Index continueToCompute() {
				log_.push_back(headINFO() + "EigenSolver<ScalarType>::continueToCompute(...) was called");
				if (lanczosBase_.lanczosvectors().size() == 0) {
					return compute();
				}

				// main calculation
				Index ret = mainCalculation_();
				log_.push_back(headINFO() + "EigenSolver<ScalarType>::compute(...) finish computing");
				return ret;

			}

			/// <summary>
			/// compute matrix diagonalization
			/// </summary>
			Index compute() {

				log_.push_back(headINFO() + "EigenSolver<ScalarType>::compute(...) was called");

				clearComputedData();

				// set initialVector if initial_vector is empty or invalid
				if (initialVector().size() != matrixHeight()) {
					log_.push_back(headINFO() + "in compute(), initial_vector is empty or invalid, then set at random");
					setInitialVector();	// set initial vector at random
				}


				// main calculation

				Index ret = mainCalculation_();

				log_.push_back(headINFO() + "EigenSolver<ScalarType>::compute(...) finish computing");
				return ret;
			}



			Index mainCalculation_() {
				es_tri_.computeFromTridiagonal(RealVectorType(0), RealVectorType(0));
				// do lanczos steps until convergence or achieve utmost state
				Index set_initialvector_is_fail = false;
				while (true) {

					// check convergence and utmost state
					updateConvergenceLog_();

					{
						if (set_initialvector_is_fail) {
							log_.push_back(headINFO() + "initial lanczosvector generation fail");
							break;
						}
						if (lanczosBase_.lanczosStepIsUtmost()) {
							log_.push_back(headINFO() + "lanczos steps finished with threshold");
							log_.push_back(headINFO() + "lanczos steps achieved full of Krylov subspace");
							break;
						}
						if (lanczosBase_.iterations() >= minIterations()) {
							if (lanczosBase_.iterations() == maxIterations()) {
								log_.push_back(headWARN() + "lanczos steps achieved maxIterations");
								break;
							}
							if (isConverged_()) {
								log_.push_back(headINFO() + "lanczos steps converged with tolerance");
								break;
							}
						}
					}

					// lanczos step
					lanczosBase_.updateLanczosSteps();
					if (lanczosvectors().size() == 0) {
						set_initialvector_is_fail = true;
					}


					// diagonalize tridiagonal matrix
					Eigen::Map<const RealVectorType> e_alpha(lanczosBase_.alpha().data(), lanczosBase_.alpha().size());
					Eigen::Map<const RealVectorType> e_beta(lanczosBase_.beta().data(), lanczosBase_.beta().size());
					es_tri_.computeFromTridiagonal(e_alpha, e_beta);
				}


				// back eigen value to original one
				Index eivalsize = es_tri_.eigenvalues().size();
				if (maxEigenvalues_ != unlimited) {
					if (maxEigenvalues_ < eivalsize) {
						eivalsize = maxEigenvalues_;
					}
				}
				eigenvalues_.resize(eivalsize);
				for (Index k = 0; k < eivalsize; ++k) {
					eigenvalues_[k] = es_tri_.eigenvalues()[k] - lanczosBase_.eigenvalueShift();
				}

				// get eigen_vector[M][N] from eigen_vector_tdm[M][M]
				if (computeEigenvectorsOn_) {
					eigenvectors_.resize(matrixHeight(), eivalsize);
					for (Index kk = 0; kk < eivalsize; ++kk) {
						eigenvectors_.col(kk) = VectorType::Zero(lanczosBase_.matrixHeight());
						for (Index m = 0, nm = es_tri_.eigenvectors().rows(); m < nm; ++m) {
							eigenvectors_.col(kk) += es_tri_.eigenvectors()(m, kk) * lanczosBase_.lanczosvectors()[m];
						}

						// phase_factor fixing for eigenvectors
						Scalar phase_factor = 1.0;
						for (Index i = 0, ni = eigenvectors_.rows(); i < ni; ++i) {
							Scalar value = eigenvectors_(i, kk);
							RealScalar abs_i = std::abs(value);
							if (abs_i > RealScalar(0.0)) {
								phase_factor =  value/ abs_i;
								break;
							}
						}
						eigenvectors_.col(kk) = VectorType((1.0 / phase_factor) * (eigenvectors_.col(kk).normalized()));
					}
				}
				else {
					eigenvectors_.resize(0, 0);
				}
				return 0;
			}





		protected:
			// helper functions for lanczos calculation

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
			/// append convergence log with current es_tri_
			/// es_tri_ must be computed before this function call
			/// </summary>
			void updateConvergenceLog_() {
				const RealVectorType& trieivals = es_tri_.eigenvalues();
				for (auto& indexForConvergence : indicesForConvergence_) {
					Index i = getFormalIndex(indexForConvergence, trieivals.size());
					if (i < 0) {
						continue;
					}
					convergenceLog_[indexForConvergence].push_back(
						trieivals[i]
					);
				}
			}

			/// <summary>
			/// this function judge convergence with current convergenceLog_
			/// </summary>
			bool isConverged_() {
				bool is_converged = true;
				if (es_tri_.eigenvalues().size() < 2) {
					is_converged = false;
					return is_converged;
				}
				RealScalar scale = es_tri_.eigenvalues()[0] - es_tri_.eigenvalues()[es_tri_.eigenvalues().size() - 1];
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
					RealScalar cur = edge[edge.size() - 1];
					RealScalar old = edge[edge.size() - 2];
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


		/// <summary>
		/// This class solves
		/// f(H)|ket>,
		/// where H is Hermite
		/// 
		/// This class is static class
		/// </summary>
		template<class Scalar_>
		class LanczosFunctionSolver {
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

			using Solver = LanczosEigenSolver<Scalar>;

			/// <summary>
			/// calculate with eigenvalues and eigenvectors 
			/// </summary>
			static void solve(
				const std::function<Scalar(Scalar)>& func,
				const RealVectorType& eivals,
				const MatrixType& eivecs,
				const VectorType& in,
				VectorType& out
			) {
				Index max = eivals.size();
				if (max > eivecs.size()) {
					max = eivecs.size();
				}

				out = VectorType::Zero(in.size());

				VectorType flambda(eivals.size());
				for (Index i = 0, ni = eivals.size(); i < ni; ++i) {
					flambda[i] = f(Scalar(flambda[i]));
				}
				out = eivecs * flambda.asDiagonal() * eivecs.adjoint() * in;

			}


			/// <summary>
			/// calculate with already-computed Lanczos solver
			/// </summary>
			static VectorType solve(
				const std::function<Scalar(Scalar)>& func,
				const Solver& solved_es
			) {
				VectorType out;
				solve(func, solved_es.eigenvalues(), solved_es.eigenvectors(), solved_es.initialVector(), out);
				return out;
			}

		};





		/// <summary>
		/// This class computes
		/// exp(xA)|ket}
		/// with Lancozs method,
		/// where x is Scalar and H is a Hermite matrix.
		/// This class also has a solver using Taylor expansion
		/// </summary>
		template<typename Scalar_>
		class LanczosExponentialSolver {
		public:
			// type aliases
			using Index = Eigen::Index;
			using Solver = LanczosEigenSolver<Scalar_>;

			using Scalar = typename Solver::Scalar;
			using RealScalar = typename Solver::RealScalar;
			using VectorType = typename Solver::VectorType;
			using RealVectorType = typename Solver::RealVectorType;
			using MatrixType = typename Solver::MatrixType;
			using RealMatrixType = typename Solver::RealMatrixType;
			using MatMulFunction = typename Solver::MatMulFunction;

			static constexpr Index unlimited = Solver::unlimited;

			/// <summary>
			/// exp(xA) を固有値 固有ベクトルで展開して計算する
			/// </summary>
			static void solveWithEigens(
				const Scalar x,
				const RealVectorType& eivals,
				const MatrixType& eivecs,
				Index max_expand,
				const VectorType& in,
				VectorType& out
			) {
				Index max = max_expand;
				if (eivals.size() - max_expand < 0) {
					max = eivals.size();
				}
				if (eivecs.cols() - max_expand < 0) {
					max = eivecs.cols();
				}

				out = VectorType::Zero(in.size());

				// 固有値で展開して加算する
				// exp(x*E_n) が小さい項から足し合わせる
				for (Index n_ = 0; n_ < max; ++n_) {
					Index n = n_;
					if (std::real(x) < 0.0) {
						n = max - n_ - 1;
					}
					Scalar inner = (eivecs.col(n).adjoint() * in)(0, 0);
					Scalar c = std::exp(x * eivals[n]) * inner;
					out += c * eivecs.col(n);
				}

			}


			/// <summary>
			/// EigenSolver を指定する場合
			/// input ベクトルは EigenSolver の initialVector が使われる
			/// </summary>
			static void solveWithLanczos(
				const Scalar& x,
				Solver& es,
				VectorType& out
			) {
				es.compute();
				solveWithEigens(
					x,
					es.eigenvalues(),
					es.eigenvectors(),
					es.eigenvalues().size(),
					es.initialVector(),
					out
				);
			}





			/// <summary>
			/// テイラー展開で計算
			/// シンプルにテイラー展開するのみ
			/// </summary>
			static void solveWithTaylorNoDivision(
				Scalar x,
				const MatMulFunction& matmul,
				Index matrix_height,
				RealScalar matrix_radius,
				const VectorType& in,
				VectorType& out,
				RealScalar error = 1.0e-14,
				Index max_expansion = unlimited
			) {

				// k == 0
				out = in;

				// k == 1
				Scalar c_k = Scalar(1.0);
				RealScalar radius_k = RealScalar(1.0);
				Index k = 1;
				VectorType ket_k;
				ket_k.resize(matrix_height);
				c_k *= x / static_cast<double>(k);
				radius_k *= matrix_radius;
				matmul(in.data(), ket_k.data());
				out += c_k * ket_k;
				if (max_expansion == 1) {
					return;
				}

				// k >= 2
				VectorType ket_pre;
				ket_pre.resize(matrix_height);
				ket_pre.swap(ket_k);
				for (k = 2; k != max_expansion; ++k) {
					c_k *= x / (double)k;
					radius_k *= matrix_radius;
					matmul(ket_pre.data(), ket_k.data());
					out += c_k * ket_k;
					ket_k.swap(ket_pre);
					if (std::abs(c_k * radius_k) < error) {
						break;
					}
				}
			}



			/// <summary>
			/// テイラー展開で計算
			/// 並進の距離が大きく精度の低下が予想される場合には並進幅を分割して計算する
			/// </summary>
			static void solveWithTaylorAutoDivision(
				Scalar x,
				const MatMulFunction& matmul,
				Index matrix_height,
				RealScalar matrix_radius,
				const VectorType& in,
				VectorType& out,
				RealScalar error = 1.0e-14,
				Index max_expansion = unlimited
			) {
				RealScalar rad = std::abs(x * matrix_radius);
				Index div = static_cast<Index>(rad + 1.0);
				for (Index i = 0; i < div; ++i) {
					Scalar x_ = static_cast<RealScalar>(1.0 / div) * x;
					solveWithTaylorNoDivision(
						x_,
						matmul,
						matrix_height,
						matrix_radius,
						in,
						out,
						error,
						max_expansion
					);
				}

			}


		};







	}


}





















