#pragma once

#include "Eigen/KroneckerProduct"
#include "cmpt/eigen_ex/multi_indices.hpp"


namespace cmpt {
	/// <summary>
	/// TensorKroneckerProduct の実装
	/// </summary>
	namespace EigenEx {

		/// <summary>
		/// Eigen::kroneckerProduct をテンソルに拡張したもの
		/// 本クラスの生成は主に EigenEx::tensorKroneckerProduct(tA,tB) を通じて行う
		/// インデックスは密
		/// </summary>
		template<class TensorLHS_, class TensorRHS_>
		class TensorKroneckerProduct {
		public:
			using Axis = integer_type::Axis;
			using Index = integer_type::Index;

			using TensorLHS = TensorLHS_;
			using TensorRHS = TensorRHS_;

			using ScalarLHS = typename TensorLHS::Scalar;
			using ScalarRHS = typename TensorRHS::Scalar;
			using Scalar = typename std::remove_reference<decltype(ScalarLHS()* ScalarRHS())>::type;


			static constexpr Axis NLHS = EigenEx::TensorTraits<TensorLHS>::NumDimensions;
			static constexpr Axis NRHS = EigenEx::TensorTraits<TensorRHS>::NumDimensions;
			static constexpr Axis NumDimensions = NLHS + NRHS;


			using VectorLHS = Eigen::Matrix<ScalarLHS, Eigen::Dynamic, 1>;
			using VectorRHS = Eigen::Matrix<ScalarRHS, Eigen::Dynamic, 1>;

			using KroneckerProductType = Eigen::KroneckerProduct<Eigen::Map<const VectorLHS>, Eigen::Map<const VectorRHS>>;


			const TensorLHS& tL;
			const TensorRHS& tR;
			ProductIndices<NumDimensions> pi;
			ProductIndices<2> pi2;


			const ProductIndices<NumDimensions>& productIndices()const { return pi; }




			TensorKroneckerProduct(
				const TensorLHS& tL_a,
				const TensorRHS& tR_a
			) : tL(tL_a), tR(tR_a) {
				Index dimL = 1;
				Index dimR = 1;
				std::array<Index, NLHS + NRHS> dims;
				for (Axis d = 0; d < NLHS; ++d) {
					dims[d] = tL.dimension(d);
					dimL *= tL.dimension(d);
				}
				for (Axis d = 0; d < NRHS; ++d) {
					dims[NLHS + d] = tR.dimension(d);
					dimR *= tR.dimension(d);
				}
				pi = ProductIndices<NLHS + NRHS>(dims);
				pi2 = ProductIndices<2>(dimL, dimR);
			}

			/// <summary>
			/// 絶対インデックスでアクセス
			/// 
			/// 絶対インデックスは kroneckerProduct のインデックスであり
			/// ストレージ上で密になる
			/// 
			/// 必ずしも密、昇順ではないことに注意
			/// </summary>
			Scalar coeff(Index i)const {
				auto i2 = pi2.indices(i);
				return tL.data()[i2[0]] * tR.data()[i2[1]];
			}

			/// <summary>
			/// 各インデックスでアクセス
			/// 各インデックスは productIndices を利用して絶対インデックスに翻訳される
			/// </summary>
			Scalar coeff(const std::array<Index, NLHS + NRHS>& indices) const {
				return coeff(pi.absoluteIndex(indices));
			}

			Scalar operator()(const std::array<Index, NLHS + NRHS>& indices)const {
				return coeff(indices);
			}


			std::array<Index, NumDimensions> dimensions()const { return productIndices().dimensions(); }
			Index dimensions(Axis d)const { return productIndices().dimension(d); }



			Eigen::Tensor<Scalar, NumDimensions> makeDenseTensor()const {
				Eigen::Tensor<Scalar, NumDimensions> t(dimensions());
				auto pi_ = ProductIndices<NumDimensions>(t.dimensions());
				for (Index i = 0, ni = t.size(); i < ni; ++i) {
					auto indices = pi_.indices(i);
					t.data()[i] = coeff(indices);
				}
				return t;
			}


		};


		template<class TensorLHS, class TensorRHS>
		inline TensorKroneckerProduct<TensorLHS, TensorRHS> tensorKroneckerProduct(
			const TensorLHS& lhs,
			const TensorRHS& rhs
		) {
			return TensorKroneckerProduct<TensorLHS, TensorRHS>(lhs, rhs);
		}


	}
}





