/// <summary>
/// このファイルでは汎用的な Eigen の拡張を記述する
/// </summary>

#pragma once



#include <array>
#include <vector>
#include <initializer_list>

#include "Eigen/Core"
#include "Eigen/Sparse"	// Eigen::Triplet<...>
#include "Eigen/Eigenvalues"
#include "Eigen/CXX11/Tensor"



/// <summary>
/// もともと cmpt/util.hpp にあった機能を EigenEx に移植したもの
/// </summary>
namespace cmpt {

	// random
	namespace EigenEx {

		/// <summary>
		/// compplex version of random_distribution
		/// 
		/// random complex values have uniform distribution on their abs and arg (NOT UNIFORM ON COMPLEX PLANE)
		/// </summary>
		template<class RealScalarType = double>
		class ComplexUniformDistribution {
		public:
			using ComplexType = std::complex<RealScalarType>;
			using result_type = ComplexType;

			std::uniform_real_distribution<RealScalarType> abs_dist;
			std::uniform_real_distribution<RealScalarType> arg_dist;

			ComplexUniformDistribution(RealScalarType abs_min, RealScalarType abs_max, RealScalarType arg_min = 0.0, RealScalarType arg_max = 0.0) :
				abs_dist(std::real(abs_min), std::real(abs_max)), arg_dist(std::imag(arg_min), std::imag(arg_max))
			{}

			template<class URBG>
			ComplexType operator()(URBG& g) {
				RealScalarType abs = abs_dist(g);
				RealScalarType arg = arg_dist(g);
				return ComplexType(abs * std::exp(ComplexType(0, arg)));
			}
		};




		/// <summary>
		/// this class give complex random distribution
		/// specifications are similar to random in c++11
		/// </summary>
		template<class RealScalarType>
		class ComplexNormalDistribution {
		public:

			using ComplexType = std::complex<RealScalarType>;
			using result_type = ComplexType;

			std::normal_distribution<RealScalarType> norm;

			ComplexNormalDistribution(RealScalarType mean = 0.0, RealScalarType stddev = 1.0) :norm(mean, stddev) {}


			template<class URBG>
			ComplexType operator()(URBG& g) {
				RealScalarType real = norm(g);
				RealScalarType imag = norm(g);
				return ComplexType(real, imag);
			}

			void reset() { norm.reset(); }

		};



		/// <summary>
		/// template to get the uniform distribution by its Scalar
		/// 
		/// Scalar is supported for
		/// int, double, std::complex{double}
		/// 
		/// add when needed
		/// </summary>
		template<class Scalar_>
		struct UniformDistributionGen {


			template<class S>
			struct Dummy {
				using Type = std::uniform_real_distribution<S>;
			};

			template<class RS>
			struct Dummy<std::complex<RS>> {
				using Type = ComplexUniformDistribution<RS>;
			};

			using Scalar = Scalar_;
			using Type = typename Dummy<Scalar>::Type;

		};


		/// <summary>
		/// template to get the normal distribution distribution by its Scalar 
		/// </summary>
		template<class Scalar_>
		struct NormalDistributionGen {

			template<class S>
			struct Dummy {
				using Type = std::normal_distribution<S>;
			};

			template<class RS>
			struct Dummy<std::complex<RS>> {
				using Type = ComplexNormalDistribution<RS>;
			};

			using Scalar = Scalar_;
			using Type = typename Dummy<Scalar>::Type;

		};


	}

	// exception
	namespace EigenEx {


		/// <summary>
		/// exception class for my code
		/// derive this class if necessary
		/// </summary>
		class RuntimeException :public std::runtime_error {
		public:
			RuntimeException(const char* _Message)
				: runtime_error(_Message)
				// message can be got with what()
			{}
		};
	}

	// CRTP
	namespace EigenEx{




		/// <summary>
		/// base class for CRTP (Curiously Recuring Template Pattern)
		/// </summary>
		template<class Derived_>
		class CRTPBase {
		public:
			using Derived = Derived_;

			Derived& derived() {
				return *static_cast<Derived*>(this);
			}

			const Derived& derived()const {
				return *static_cast<Derived const*>(this);
			}

		};


		

	}
}




namespace cmpt {

	/// <summary>
	/// Eigen の型を構築するためのクラス
	/// 初期化子リストから構築可能
	/// </summary>
	namespace EigenEx {

		template<class MatrixType>
		class Generate {
		};

		template<class ScalarType, int Rows, int Cols>
		class Generate<Eigen::Matrix<ScalarType, Rows, Cols>> {
		};

		template<class ScalarType>
		class Generate<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>> {
		public:
			using MatrixType = Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>;

			/// <summary>
			/// ネストした initializer_list から行列を構築
			/// 行優先
			/// {
			///		{0,1},
			///		{2,3}
			/// }
			/// -> | 0 1 |
			///    | 2 3 |
			/// </summary>
			template<class T>
			static MatrixType from(const std::initializer_list<std::initializer_list<T>>& ill) {
				MatrixType m;
				int rows = ill.size();
				if (ill.size() == 0) {
					return MatrixType();
				}
				int cols = ill.begin()->size();
				if (cols == 0) {
					return MatrixType();
				}
				m.resize(rows, cols);
				{
					int r = 0;
					for (const auto& il : ill) {
						int c = 0;
						for (const auto& x : il) {
							m(r, c) = x;
							++c;
						}
						++r;
					}
				}
				return m;
			}

		};

		template<class ScalarType>
		class Generate<Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>> {
		public:
			using MatrixType = Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>;

			template<class T>
			static MatrixType from(const std::initializer_list<T>& il) {
				MatrixType m;
				int rows = il.size();
				if (il.size() == 0) {
					return MatrixType();
				}

				m.resize(rows);
				{
					int r = 0;
					for (const auto& x : il) {
						m[r] = x;
						++r;
					}
				}
				return m;
			}

		};


	}



	/// <summary>
	/// 線形代数関連
	/// </summary>
	namespace EigenEx {



		/// <summary>
		/// |out} = exp(x*A)|in}
		/// を計算する
		/// </summary>
		template<class Scalar_>
		class OperateAsExp {
		public:
			using Scalar = Scalar_;
			using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
			using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;


			using MatMulFunction = std::function<void(Scalar const*, Scalar*)>;

			/// <summary>
			/// |out} = exp(x*A)|in}
			/// を計算する
			/// 計算にはテイラー展開を用いる
			/// 計算終了条件として
			/// 最大展開数 k < max_expansion (max_expansion == -1 のとき制限なし)
			/// 展開係数のエラー x^k/k! < error
			/// を指定可能
			/// どちらかが満たされたとき展開を終了する
			/// xや線形作用に大きい値を指定するとテイラー展開が激しく振動することにより桁落ちが大きくなるため注意
			/// x*(線形作用の絶対値最大固有値) < 1.0 でやるのが望ましい
			/// 数値誤差を避けるために高次から足し上げるべきだがメモリ節約のためここでは低次から足し上げている
			/// </summary>
			inline static void compute(
				Scalar const* in,
				Scalar* out,
				Scalar x,
				const MatMulFunction& matmul,
				int matrix_height,
				RealScalar error = 1.0e-14,
				int max_expansion = -1
			) {
				Eigen::Map<const VectorType> m_in(in, matrix_height);
				Eigen::Map<VectorType> m_out(out, matrix_height);

				// k == 0
				m_out = m_in;

				// k == 1
				Scalar c_k = Scalar(1.0);
				int k = 1;
				VectorType ket_k;
				ket_k.resize(matrix_height);
				c_k *= x / static_cast<double>(k);
				matmul(m_in.data(), ket_k.data());
				m_out += c_k * ket_k;
				if (max_expansion == 1) {
					return;
				}

				// k >= 2
				VectorType ket_pre;
				ket_pre.resize(matrix_height);
				ket_pre.swap(ket_k);
				for (k = 2; k != max_expansion; ++k) {
					c_k *= x / (double)k;
					matmul(ket_pre.data(), ket_k.data());
					m_out += c_k * ket_k;
					ket_k.swap(ket_pre);
					if (std::abs(c_k) < error) {
						break;
					}
				}
			}

			inline static void compute_inplace(
				Scalar* pv,
				Scalar x,
				const MatMulFunction& matmul,
				int matrix_height,
				RealScalar error = 1.0e-14,
				int max_expansion = -1
			) {
				Eigen::Map<VectorType> m_in(pv, matrix_height);
				VectorType vout;
				vout.resize(matrix_height);
				compute(m_in.data(), vout.data(), x, matmul, matrix_height, error, max_expansion);
				m_in = vout;
			}




		};







		/// <summary>
		/// シュミット直交化
		/// v を u に対して直交化させる
		/// </summary>
		template<class VectorType1, class VectorType2>
		inline void schmidt_orthogonalize(
			VectorType1& v,
			const VectorType2& u,
			bool u_is_normalized = true
		) {
			using RealScalarType = typename Eigen::NumTraits<typename VectorType1::Scalar>::Real;
			RealScalarType weight = 1.0;
			if (!u_is_normalized) {
				weight = 1.0 / u.norm();
			}
			v -= weight * weight * u.dot(v) * u;
		}



		/// <summary>
		/// 直交補空間の基底を取得するためのクラス
		/// </summary>
		template<class MatrixType>
		class OrthogonalSpace {
		public:
			using Scalar = typename MatrixType::Scalar;
			using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

			MatrixType basis;	// 直交化された正規直交基底


			/// <summary>
			/// 指定した基底と直行する基底を構築
			/// 指定する基底は独立性、正規性、直交性を問わない
			/// 構築される基底の数は error により変動する
			/// 構築される基底は以下の条件を満たす
			/// 1. 規格直交基底
			/// 2. 基底は辞書式に並べたときの逆順で最も早いとり方で選ばれソートされている
			/// </summary>
			OrthogonalSpace(
				const MatrixType& target_basis,
				double error = 1.0e-12
			) {
				int rs = target_basis.rows();
				int cs = target_basis.cols();

				std::vector<VectorType> o_basis;
				MatrixType id = MatrixType::Identity(rs, rs);
				int cc = 0;	// index of current orhogonal baisis
				for (int i = 0; i < rs; ++i) {
					VectorType v = id.col(i);
					for (int c = 0; c < cs; ++c) {
						schmidt_orthogonalize(v, target_basis.col(c), false);
					}
					for (auto& u : o_basis) {
						schmidt_orthogonalize(v, u, false);
					}
					if (v.norm() < error) {
						continue;
					}
					v.normalize();
					o_basis.push_back(v);
				}
				basis.resize(target_basis.rows(), o_basis.size());

				for (int c = 0, cs = basis.cols(); c < cs; ++c) {
					basis.col(c) = o_basis[c];
				}
			}

		};


		/// <summary>
		/// OrthogonalSpace のデバッグ用に作ったクラス
		/// </summary>
		template<class MatrixType>
		class OrthogonalSpaceDebug :public OrthogonalSpace<MatrixType> {
		public:
			using Super = OrthogonalSpace<MatrixType>;

			using Scalar = typename MatrixType::Scalar;
			using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;


			MatrixType target_basis;		// ターゲットの基底
			double error;					// 使用したエラー
			MatrixType ortho_check_basis;	// 基底状態の直交化チェック
			MatrixType ortho_check_target;	// ターゲットの直交化チェック

			OrthogonalSpaceDebug(
				const MatrixType& target_basis_,
				double error_ = 1.0e-12
			) :
				Super(target_basis_, error_),
				target_basis(target_basis_),
				error(error_)
			{
				this->ortho_check_basis = this->basis.adjoint() * this->basis;
				this->ortho_check_target = this->target_basis.adjoint() * this->target_basis;
			}

		};







		/// <summary>
		/// スパースな作用関連
		/// そのうちパウリ行列の作用を取り入れたい
		/// </summary>


			/// <summary>
			/// triplets で表現されたスパース行列を mat に左から作用させる
			/// </summary>
		template<class ScalarType, class MatrixType>
		inline void operate_triplets(const std::vector<Eigen::Triplet<ScalarType>>& triplets, MatrixType& mat) {
			std::vector<Eigen::Triplet<ScalarType>> vals;
			for (int c = 0, cs = mat.cols(); c < cs; ++c) {
				for (auto& tri : triplets) {
					vals.push_back(
						Eigen::Triplet<ScalarType>(
							tri.row(),
							c,
							tri.value() * mat(tri.col(), c)
							)
					);
				}
			}
			for (auto& vtri : vals) {
				mat(vtri.row(), vtri.col()) = ScalarType(0.0);
			}
			for (auto& vtri : vals) {
				mat(vtri.row(), vtri.col()) += vtri.value();
			}
		}


		/// <summary>
		/// triplets で表現されたスパース行列を mat に右から作用させる
		/// </summary>
		template<class ScalarType, class MatrixType>
		inline void operate_triplets(MatrixType& mat, const std::vector<Eigen::Triplet<ScalarType>>& triplets) {
			std::vector<Eigen::Triplet<ScalarType>> vals;
			for (int r = 0, rs = mat.rows(); r < rs; ++r) {
				for (auto& tri : triplets) {
					vals.push_back(
						Eigen::Triplet<ScalarType>(
							r,
							tri.col(),
							tri.value() * mat(tri.row(), r)
							)
					);
				}
			}
			for (auto& vtri : vals) {
				mat(vtri.row(), vtri.col()) = ScalarType(0.0);
			}
			for (auto& vtri : vals) {
				mat(vtri.row(), vtri.col()) += vtri.value();
			}
		}


		/// <summary>
		/// 行列に回転行列を左から作用させる
		/// </summary>
		template<class MatrixType>
		inline void rotate_from_left(int ix, int iy, double theta, MatrixType& mat) {
			std::vector<Eigen::Triplet<double>> tri;
			tri.push_back(Eigen::Triplet<double>(ix, ix, std::cos(theta)));
			tri.push_back(Eigen::Triplet<double>(ix, iy, -std::sin(theta)));
			tri.push_back(Eigen::Triplet<double>(iy, ix, std::sin(theta)));
			tri.push_back(Eigen::Triplet<double>(iy, iy, std::cos(theta)));
			operate_triplets(tri, mat);
		}

		/// <summary>
		/// 行列に回転行列を右から作用させる(adjoint を適当に取る)
		/// </summary>
		template<class MatrixType>
		inline void rotate_from_right(int ix, int iy, double theta, MatrixType& mat) {
			std::vector<Eigen::Triplet<double>> tri;
			tri.push_back(Eigen::Triplet<double>(ix, ix, std::cos(theta)));
			tri.push_back(Eigen::Triplet<double>(ix, iy, std::sin(theta)));
			tri.push_back(Eigen::Triplet<double>(iy, ix, -std::sin(theta)));
			tri.push_back(Eigen::Triplet<double>(iy, iy, std::cos(theta)));
			operate_by_triplets(mat, tri);
		}







	}




	/// <summary>
	/// make_indices(...)
	/// </summary>
	namespace EigenEx {

		/// <summary>
		/// make_indices の実装部分
		/// </summary>
		namespace {
			template <std::size_t N, std::size_t M>
			void make_indices_impl(Eigen::array<int, N>& indices) {}

			template <int N, int M, class Head, class... Args>
			void make_indices_impl(Eigen::array<int, N>& indices, const Head& head, const Args&... args) {
				indices[M] = static_cast<int>(head);
				make_indices_impl<N, M + 1, Args...>(indices, args...);
			}
		}


		/// <summary>
		/// make_indices(...)
		/// 引数から Eigen::array{int,N} を生成する
		/// static_cast{int}() で int にキャストできるものはキャストする
		/// キャストの仕様(特に gcc における)のために初期化子リストには意図的に対応していない
		/// </summary>
		template <class... Args>
		auto makeIndices(Args... args)->Eigen::array<int, sizeof...(args)>{
			using Indices = Eigen::array<int, sizeof...(args)>;
			Indices indices;
			make_indices_impl<sizeof...(Args), 0, Args...>(indices, args...);
			return indices;
		}

	}



	/// <summary>
	/// rowwiseShuffle
	/// colwiseShuffle
	/// cwiseShuffle
	/// </summary>
	namespace EigenEx {

		/// <summary>
		/// row (列) ごとにインデックスを並び替える
		/// 例
		/// [0,2,1] で並び替えた場合
		/// 1 2 3    1 3 2
		/// 4 5 6 -> 4 6 5
		/// 7 8 9    7 9 8
		/// </summary>
		template<class Derived, class Indices>
		void rowwiseShuffle(Eigen::DenseBase<Derived>& db, const Indices& shuffle) {
			using Scalar = typename Eigen::DenseBase<Derived>::Scalar;
			using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
			std::vector<VectorType> vdb;
			for (int c = 0, nc = db.cols(); c < nc; ++c) {
				vdb.push_back(db.col(c));
			}
			for (int c = 0, nc = db.cols(); c < nc; ++c) {
				db.col(c) = vdb[shuffle[c]];
			}
		}


		/// <summary>
		/// col (行) ごとにインデックスを並び替える
		/// 例
		/// [0,2,1] で並び替えた場合
		/// 1 2 3    1 2 3
		/// 4 5 6 -> 7 8 9
		/// 7 8 9    4 5 6
		/// </summary>
		template<class Derived, class Indices>
		void colwiseShuffle(Eigen::DenseBase<Derived>& db, const Indices& shuffle) {
			Eigen::DenseBase<Derived> db_ = db.transepose();
			rowwiseShuffle(db_, shuffle);
			db = db_.transepose();
		}

		/// <summary>
		/// 要素ごとにシャッフルする
		/// インデックスは 要素のストレージオーダーによる
		/// </summary>
		template<class Derived, class Indices>
		void cwiseShuffle(Eigen::PlainObjectBase<Derived>& db, const Indices& shuffle) {
			using Scalar = typename Eigen::PlainObjectBase<Derived>::Scalar;
			using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
			Eigen::Map<VectorType> mdb(db.data(), db.size());
			VectorType vdb = mdb;
			for (int i = 0, ni = mdb.size(); i < ni; ++i) {
				mdb[i] = vdb[shuffle[i]];
			}
		}


	}



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
	/// EigenSolverTraits
	/// </summary>
	namespace EigenEx {
		/// <summary>
		/// 固有値ソルバを行列型ごとに習得するクラステンプレート
		/// EigenSolver と ComplexEigenSolver を呼び分ける
		/// </summary>
		template<class MatrixType_>
		class EigenSolverTraits {
		public:
			using MatrixType = MatrixType_;
			using Scalar = typename MatrixType::Scalar;
			using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
			using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
			using RealVectorType = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;


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
		template<class Scalar, int N>
		static Eigen::Tensor<Scalar, N> zerowiselyResized(
			const Eigen::Tensor<Scalar, N>& t,
			const Eigen::array<int, N> shape
		) {
			Eigen::array<int, N> starts;
			for (auto& s : starts) {
				s = 0;
			}
			Eigen::array<int, N> extents;
			for (int i = 0; i < N; ++i) {
				extents[i] = min_value(static_cast<int>(t.dimension(i)), shape[i]);
			}
			Eigen::array<std::pair<int, int>, N> paddings;
			for (int i = 0; i < N; ++i) {
				paddings[i] = std::pair<int, int>{ 0,shape[i] - extents[i] };
			}
			return Eigen::Tensor<Scalar, N>(t.slice(starts, extents)).pad(paddings);
		}


		/// <summary>
		/// 行列 resize するが
		/// 余分な部分は削除し
		/// 足りない部分はゼロで埋める
		/// </summary>
		template<class Scalar, int N>
		static Eigen::Tensor<Scalar, N> zerowiselyResized(
			const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& m,
			int rows,
			int cols
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
		template<class Scalar, int N>
		static Eigen::Tensor<Scalar, N> zerowiselyResized(
			const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& v,
			int size
		) {
			using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
			VectorType v_ = VectorType::Zero(size);
			int n = min_value(v.size(), v_.size());
			v_.block(0, 0, n, 1) = v.block(0, 0, n, 1);
			return v_;
		}



		/// <summary>
		/// ベクトルを対角テンソルと解釈してテンソルとの contraction をとる
		/// テンソルの添字は引数の添字をそのまま引き継ぐ
		/// より一般に拡張できるかもしれない
		/// </summary>
		template<class Scalar, class ScalarV, int N>
		static Eigen::Tensor<Scalar, N> contractVectorAsDiagonal(
			const Eigen::Tensor<Scalar, N>& t,
			const Eigen::Matrix<ScalarV, Eigen::Dynamic, 1>& v,
			int contract_axis
		) {
			int less = 1;
			for (int i = 0; i < contract_axis; ++i) {
				less *= t.dimension(i);
			}
			int more = 1;
			for (int i = contract_axis + 1; i < N; ++i) {
				more *= t.dimension(i);
			}

			//auto idx = mi::ProductIndices<3, mi::ColMajor>(less, t.dimension(contract_axis), more);
			Eigen::array<int, 3> idx_shape{ less,static_cast<int>(t.dimension(contract_axis)) ,more };

			Eigen::TensorMap<Eigen::Tensor<const Scalar, 3>> mt(t.data(), idx_shape);
			Eigen::Tensor<Scalar, N> t_out(t.dimensions());
			Eigen::TensorMap<Eigen::Tensor<Scalar, 3>> mt_out(t_out.data(), idx_shape);
			for (int i = 0, ni = idx_shape[0]; i < ni; ++i) {
				for (int j = 0, nj = idx_shape[2]; j < nj; ++j) {
					for (int k = 0, nk = v.size(); k < nk; ++k) {
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
		template<class Scalar, class ScalarM, int N>
		static Eigen::Tensor<Scalar, N> transformTensorWithMatrix(
			const Eigen::Tensor<Scalar, N>& t,
			const Eigen::TensorMap<Eigen::Tensor<const ScalarM, 2>>& m,
			int t_axis
		) {



			Eigen::array<int, N> shuffle;
			for (int i = 0; i < t_axis; ++i) {
				shuffle[i] = i;
			}
			shuffle[t_axis] = N - 1;
			for (int i = t_axis + 1; i < N; ++i) {
				shuffle[i] = i - 1;
			}
			Eigen::Tensor<Scalar, N> t_r =
				t.contract(
					m,
					Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>{t_axis, 1}}
			).shuffle(shuffle);

			return t_r;
		}


		/// <summary>
		/// テンソルの指定した添字を行列により変換する
		/// テンソルの添字は引数の添字をそのまま引き継ぐ(ソートしない)
		/// 例)  T_{i,j',k} = M_{j',j}*T_{i,j,k}
		/// </summary>
		template<class Scalar, class ScalarM, int N>
		static Eigen::Tensor<Scalar, N> transformTensorWithMatrix(
			const Eigen::Tensor<Scalar, N>& t,
			const Eigen::Matrix<ScalarM, Eigen::Dynamic, Eigen::Dynamic>& m,
			int t_axis
		) {
			Eigen::TensorMap<Eigen::Tensor<const ScalarM, 2>> tm(m.data(), m.rows(), m.cols());
			return transformTensorWithMatrix(t, tm, t_axis);
		}







	}




}
