#pragma once


#include <random>

#include "Eigen/Core"
#include "Eigen/CXX11/Tensor"	// for Eigen::Tensor

#include "cmpt/eigen_ex/util.hpp"



/// <summary>
/// 乱数関連
/// </summary>
namespace cmpt {
	
	namespace EigenEx {

		/// <summary>
		/// 乱数の Eigen::Matrix を取得するための分布関数を表現するクラス
		/// 返す乱数の型はテンプレート引数 Distribution から推論する
		/// すなわち Eigen::Matrix{Distribution::result_type,Eigen::Dynamic,Eigen::Dynamic}
		/// </summary>
		template<class Distribution>
		class MatrixDistribution {

		public:
			using ScalarType = typename Distribution::result_type;
			using result_type = Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>;
			using RealScalarType = typename Eigen::NumTraits<ScalarType>::Real;


			Distribution dist;
			int rows;
			int cols;
			bool normalize_on;

			MatrixDistribution(const Distribution& dist_ = Distribution(), int rows_ = 0, int cols_ = 0, bool normalize_on_ = true) :
				dist(dist_), rows(rows_), cols(cols_), normalize_on(normalize_on_)
			{}


			template<class URBG>
			result_type operator()(URBG& g) {
				result_type mat;
				mat.resize(rows, cols);
				auto norm = RealScalarType(0);
				for (int r = 0; r < rows; ++r) {
					for (int c = 0; c < cols; ++c) {
						mat(r, c) = dist(g);
					}
				}

				if (normalize_on) {
					mat.normalize();
				}
				return mat;
			}


		};


		/// <summary>
		/// 乱数の Eigen::Vector を取得するための分布関数を表現するクラス
		/// 返す乱数の型はテンプレート引数 Distribution から推論する
		/// すなわち Eigen::Vector{Distribution::result_type,Eigen::Dynamic,Eigen::Dynamic}
		/// </summary>
		template<class Distribution>
		class VectorDistribution {
		public:
			using ScalarType = typename Distribution::result_type;
			using result_type = Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>;
			using RealScalarType = typename Eigen::NumTraits<ScalarType>::Real;


			Distribution dist;
			int size;
			bool normalize_on;

			VectorDistribution(const Distribution& dist_ = Distribution(), int size_ = 0, bool normalize_on_ = true) :
				dist(dist_), size(size_), normalize_on(normalize_on_)
			{}

			template<class URBG>
			result_type operator()(URBG& g) {
				result_type vec;
				vec.resize(size);
				auto norm = RealScalarType(0);
				for (int i = 0; i < size; ++i) {
					vec[i] = dist(g);
				}
				if (normalize_on) {
					vec.normalize();
				}
				return vec;
			}


		};



		/// <summary>
		/// ランダムな直交行列(ユニタリ行列)を生成する分布関数を表現するクラス
		/// 
		/// 各列が直行するノルム1のベクトルで構成される行列を生成する
		/// rows<cols の場合には余剰な列はゼロ埋めされる
		/// </summary>
		template<class MatrixType>
		class OrthogonalMatrixDistribution {
		public:
			using result_type = MatrixType;
			using ScalarType = typename result_type::Scalar;
			using VectorType = Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>;
			using XDist = typename NormalDistributionGen<ScalarType>::Type;
			using VDist = VectorDistribution<XDist>;

			int rows;
			int cols;
			VDist dist;


			OrthogonalMatrixDistribution(int rows_, int cols_) :
				rows(rows_),
				cols(cols_),
				dist(
					XDist(),
					rows,
					true
				)
			{}

			template<class URBG>
			result_type operator()(URBG& g) {
				result_type mat;
				int min_rc = (std::min)(rows, cols);
				mat.resize(rows, cols);

				for (int c = 0, nc = min_rc; c < nc; ++c) {
					VectorType vec = dist(g);
					for (int cc = 0; cc < c; ++cc) {
						schmidt_orthogonalize(vec, mat.col(cc), false);
					}
					mat.col(c) = vec.normalized();
				}
				for (int c = min_rc, nc = mat.cols(); c < nc; ++c) {
					mat.col(c) = VectorType::Zero(mat.rows());
				}
				return mat;
			}



		};



		/// <summary>
		/// 乱数の Eigen::Tensor{Scalar,NumDims} を取得するための分布関数を表現するクラス
		/// 返す乱数の型はテンプレート引数 Distribution から推論する
		/// すなわち Eigen::Matrix{Distribution::result_type,Eigen::Dynamic,Eigen::Dynamic}
		/// </summary>
		template<class Distribution, int NumDims>
		class TensorDistribution {

		public:
			using Scalar = typename Distribution::result_type;
			using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
			using result_type = Eigen::Tensor<Scalar, NumDims>;
			//using Dimensions=typename result_type::Dimensions;
			using Dimensions = typename Eigen::array<int, NumDims>;

			Distribution dist;
			Dimensions dimensions;


			TensorDistribution(
				const Distribution& dist_ = Distribution(),
				const Dimensions& dims_ = Dimensions()
			) :
				dist(dist_),
				dimensions(dims_)
			{}


			template<class URBG>
			result_type operator()(URBG& g) {
				result_type t;
				t.resize(dimensions);
				Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> mt(t.data(), t.size());
				for (int i = 0, ni = mt.size(); i < ni; ++i) {
					mt[i] = dist(g);
				}
				return t;
			}


		};




	}

}

