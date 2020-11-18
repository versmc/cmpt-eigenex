#pragma once

#include <random>
#include "Eigen/CXX11/Tensor"


namespace cmpt {
	namespace eigen_ex {

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
