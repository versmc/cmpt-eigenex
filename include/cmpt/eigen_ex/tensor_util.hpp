#pragma once

/// <summary>
/// This file defines some functions to extends the functions of Eigen::Tensor
/// 
/// include paths must provide the path for Eigen/CXX11/Tensor
/// </summary>

#include <array>
#include <vector>
#include "Eigen/CXX11/Tensor"

namespace cmpt {
	

	/// <summary>
	///  TensorTraits
	/// </summary>
	namespace EigenEx {

		/// <summary>
		/// Eigen::Tensor 等に関するコンパイル時の情報を取得するクラス
		/// 
		/// 特に
		/// NumDimensions を constexpr で取得
		/// ネストした std::vector との相互変換
		/// を行う
		/// 
		/// 
		/// テンプレート引数 TensorType_ は以下を持つ必要がある
		/// 
		/// Eigen::Tensor{Scalar,NumDims} 型
		/// もしくは
		/// メンバ型 Scalar と static constexpr int NumDimensions を持つ型
		/// </summary>
		template<class TensorType_>
		class TensorTraits {
		public:
			using TensorType = TensorType_;
			using Scalar = typename TensorType::Scalar;
			using Axis = int;

		protected:
			template<class TensorType_ND>
			struct NumDimensionsGetter {

				template<class T>
				struct Dummy {
					static constexpr Axis NumDimensions = T::NumDimensions;
				};

				template<class Scalar, Axis NumDims>
				struct Dummy<Eigen::Tensor<Scalar, NumDims>> {
					static constexpr Axis NumDimensions = NumDims;
				};

				static constexpr Axis value = Dummy<TensorType_ND>::NumDimensions;
			};

			template<class Scalar_, Axis NumDims_>
			struct NestedVectorGetter {

				template<class S, Axis N>
				struct Dummy {
					using Type = typename std::vector<typename NestedVectorGetter<S, N - 1>::Type>;
				};

				template<class S>
				struct Dummy<S, 0> {
					using Type = S;
				};


				using Type = typename Dummy<Scalar_, NumDims_>::Type;

			};

		public:
			//static constexpr Axis NumDimensions = NumDimensionsGetter<TensorType>::value;
			static constexpr Axis NumDimensions = NumDimensionsGetter<TensorType>::value;
			using NestedVectorType = typename NestedVectorGetter<Scalar, NumDimensions>::Type;
			using IndicesType = Eigen::array<int, NumDimensions>;

			static NestedVectorType TensorToNestedVector(const TensorType& t) {
				NestedVectorType nv;
				IndicesType indices;
				for (auto& i : indices) { i = 0; }
				Dummy<TensorType, NestedVectorType, 0>::set_nv_impl(t, nv, indices);
				return nv;
			}

			static IndicesType getNestedVectorShape(const NestedVectorType& nv) {
				IndicesType shape;
				Dummy<TensorType, NestedVectorType, 0>::set_shape_impl(nv, shape);
				return shape;
			}

			static TensorType NestedVectorToTensor(const NestedVectorType& nv) {
				IndicesType shape = getNestedVectorShape(nv);
				TensorType t;
				t.resize(shape);
				t.setZero();
				IndicesType indices;
				Dummy<TensorType, NestedVectorType, 0>::set_tensor_impl(nv, t, indices);
				return t;
			}

		protected:
			template<class T, class NV, int M>
			class Dummy {
			public:
				inline static void set_nv_impl(const TensorType& t, NV& nv, IndicesType& indices) {
					nv.resize(t.dimension(M));
					for (int i = 0, ni = nv.size(); i < ni; ++i) {
						indices[M] = i;
						Dummy<T, typename NV::value_type, M + 1>::set_nv_impl(t, nv[i], indices);
					}
				}

				inline static void set_shape_impl(const NV& nv, IndicesType& shape) {
					int s = nv.size();
					shape[M] = s;
					if (s > 0) {
						Dummy<T, typename NV::value_type, M + 1>::set_shape_impl(nv[0], shape);
					}
					else {
						for (int m = M, nm = shape.size(); m < nm; ++m) {
							shape[m] = 0;
						}
					}
				}

				inline static void set_tensor_impl(const NV& nv, TensorType& t, IndicesType& indices) {
					for (int i = 0, ni = nv.size(); i < ni; ++i) {
						indices[M] = i;
						Dummy<T, typename NV::value_type, M + 1>::set_tensor_impl(nv[i], t, indices);
					}
				}
			};

			template<class T, class NV>
			class Dummy<T, NV, NumDimensions> {
			public:
				inline static void set_nv_impl(const TensorType& t, NV& nv, IndicesType& indices) {
					nv = t(indices);
				}

				inline static void set_shape_impl(const NV& nv, IndicesType& shape) {}

				inline static void set_tensor_impl(const NV& nv, TensorType& t, IndicesType& indices) {
					t(indices) = nv;
				}

			};


		};



	}





	/// <summary>
	/// Eigen の Tensor 関連の拡張関数群
	/// </summary>
	namespace EigenEx {

		/// <summary>
		/// テンソルの縮約に使うコントラクションをネストした初期化子リストのような書き方で指定するための関数
		/// 例) t1.contract(t2,makeContractions({{0,1},...,{1,2}}));
		/// gcc の場合に {}内の整数型のキャストを自動でやってくれないため注意が必要
		/// </summary>
		template<std::size_t N>
		Eigen::array<Eigen::IndexPair<std::size_t>, N> makeContractions(const Eigen::IndexPair<std::size_t>(&pairs)[N]) {
			Eigen::array<Eigen::IndexPair<std::size_t>, N> pairs_;
			for (std::size_t i = 0; i < N; ++i) {
				pairs_[i] = pairs[i];
			}
			return pairs_;
		}


		/// <summary>
		/// テンソルを resize するが
		/// 余分な部分は削除し
		/// 足りない部分はゼロで埋める
		/// </summary>
		template<class Scalar, std::size_t N>
		static Eigen::Tensor<Scalar, N> zerowiselyResized(
			const Eigen::Tensor<Scalar, N>& t,
			const Eigen::array<Eigen::Index, N> shape
		) {
			Eigen::array<Eigen::Index, N> starts;
			for (auto& s : starts) {
				s = 0;
			}
			Eigen::array<Eigen::Index, N> extents;
			for (int i = 0; i < N; ++i) {
				extents[i] = min_value(static_cast<Eigen::Index>(t.dimension(i)), shape[i]);
			}
			Eigen::array<std::pair<Eigen::Index, Eigen::Index>, N> paddings;
			for (int i = 0; i < N; ++i) {
				paddings[i] = std::pair<Eigen::Index, Eigen::Index>{ 0,shape[i] - extents[i] };
			}
			return Eigen::Tensor<Scalar, N>(t.slice(starts, extents)).pad(paddings);
		}


		/// <summary>
		/// 行列 resize するが
		/// 余分な部分は削除し
		/// 足りない部分はゼロで埋める
		/// </summary>
		template<class Scalar, std::size_t N>
		static Eigen::Tensor<Scalar, N> zerowiselyResized(
			const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& m,
			Eigen::Index rows,
			Eigen::Index cols
		) {
			using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
			MatrixType m_ = MatrixType::Zero(rows, cols);
			int nr = min_value(m.rows(), m_.rows());
			int nc = min_value(m.cols(), m_.cols());
			m_.block(0, 0, nr, nc) = m.block(0, 0, nr, nc);
			return m_;
		}


		/// <summary>
		/// ベクトルを resize するが
		/// 余分な部分は削除し
		/// 足りない部分はゼロで埋める
		/// </summary>
		template<class Scalar, std::size_t N>
		static Eigen::Tensor<Scalar, N> zerowiselyResized(
			const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& v,
			int size
		) {
			using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
			VectorType v_ = VectorType::Zero(size);
			Eigen::Index n = min_value(v.size(), v_.size());
			v_.block(0, 0, n, 1) = v.block(0, 0, n, 1);
			return v_;
		}



		/// <summary>
		/// ベクトルを対角テンソルと解釈してテンソルとの contraction をとる
		/// テンソルの添字は引数の添字をそのまま引き継ぐ
		/// より一般に拡張できるかもしれない
		/// </summary>
		template<class Scalar, class ScalarV, std::size_t N>
		static Eigen::Tensor<Scalar, N> contractVectorAsDiagonal(
			const Eigen::Tensor<Scalar, N>& t,
			const Eigen::Matrix<ScalarV, Eigen::Dynamic, 1>& v,
			std::size_t contract_axis
		) {
			Eigen::Index less = 1;
			for (Eigen::Index i = 0; i < contract_axis; ++i) {
				less *= t.dimension(i);
			}
			Eigen::Index more = 1;
			for (Eigen::Index i = contract_axis + 1; i < N; ++i) {
				more *= t.dimension(i);
			}

			//auto idx = mi::ProductIndices<3, mi::ColMajor>(less, t.dimension(contract_axis), more);
			Eigen::array<Eigen::Index, 3> idx_shape{ less,static_cast<int>(t.dimension(contract_axis)) ,more };

			Eigen::TensorMap<Eigen::Tensor<const Scalar, 3>> mt(t.data(), idx_shape);
			Eigen::Tensor<Scalar, N> t_out(t.dimensions());
			Eigen::TensorMap<Eigen::Tensor<Scalar, 3>> mt_out(t_out.data(), idx_shape);
			for (Eigen::Index i = 0, ni = idx_shape[0]; i < ni; ++i) {
				for (Eigen::Index j = 0, nj = idx_shape[2]; j < nj; ++j) {
					for (Eigen::Index k = 0, nk = v.size(); k < nk; ++k) {
						mt_out(i, k, j) = mt(i, k, j) * v(k);
					}
				}
			}
			return t_out;
		}



		/// <summary>
		/// テンソルの指定した添字を行列により変換する
		/// テンソルの添字は引数の添字をそのまま引き継ぐ(ソートしない)
		/// 例)  T_{i,j',k} = M_{j',j}*T_{i,j,k}
		/// </summary>
		template<class Scalar, class ScalarM, std::size_t N>
		static Eigen::Tensor<Scalar, N> transformTensorWithMatrix(
			const Eigen::Tensor<Scalar, N>& t,
			const Eigen::TensorMap<Eigen::Tensor<const ScalarM, 2>>& m,
			Eigen::Index t_axis
		) {



			Eigen::array<std::size_t, N> shuffle;
			for (std::size_t i = 0; i < t_axis; ++i) {
				shuffle[i] = i;
			}
			shuffle[t_axis] = N - 1;
			for (std::size_t i = t_axis + 1; i < N; ++i) {
				shuffle[i] = i - 1;
			}
			Eigen::Tensor<Scalar, N> t_r =
				t.contract(
					m,
					Eigen::array<Eigen::IndexPair<std::size_t>, 1>{Eigen::IndexPair<std::size_t>{t_axis, 1}}
			).shuffle(shuffle);

			return t_r;
		}


		/// <summary>
		/// テンソルの指定した添字を行列により変換する
		/// テンソルの添字は引数の添字をそのまま引き継ぐ(ソートしない)
		/// 例)  T_{i,j',k} = M_{j',j}*T_{i,j,k}
		/// </summary>
		template<class Scalar, class ScalarM, std::size_t N>
		static Eigen::Tensor<Scalar, N> transformTensorWithMatrix(
			const Eigen::Tensor<Scalar, N>& t,
			const Eigen::Matrix<ScalarM, Eigen::Dynamic, Eigen::Dynamic>& m,
			std::size_t t_axis
		) {
			Eigen::TensorMap<Eigen::Tensor<const ScalarM, 2>> tm(m.data(), m.rows(), m.cols());
			return transformTensorWithMatrix(t, tm, t_axis);
		}



	}


}
