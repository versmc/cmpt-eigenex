#pragma once

#include "Eigen/Core"
#include "Eigen/SVD"
#include "Eigen/CXX11/Tensor"





/// <summary>
/// tensorSVD(...)
/// 広報互換性のために残しているが現在は TensorSVD クラスを推奨する
/// </summary>
namespace cmpt {
	namespace EigenEx {

		/// <summary>
		/// 軸がソートされているテンソルを分解し、軸がソートされているテンソルを返す
		/// 例1)
		/// t(i0,i1,i2,i3) == L(i0,i1,is) S(is,is) R(i2,i3,is)
		/// 
		/// 例2)
		/// t(i0,i1,i2,i3) == L(i0,i1,i2,is) S(is,is) R(i3,is)
		/// </summary>
		template<class Scalar, int N, int L>
		static void tensorSVD(
			const Eigen::Tensor<Scalar, N>& t,
			Eigen::Tensor<Scalar, L>& tL,
			Eigen::Tensor<Scalar, N - L + 2>& tR,
			Eigen::Matrix<typename Eigen::NumTraits<Scalar>::Real, Eigen::Dynamic, 1>& s,
			unsigned int computationOptions
		) {
			
			using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
			using SVDSolver = Eigen::BDCSVD<MatrixType>;

			constexpr int R = N - L + 2;

			// 行列のサイズ取得
			int mrows = 1;
			int mcols = 1;
			for (int i = 0; i < L - 1; ++i) {
				mrows *= t.dimension(i);
			}
			for (int i = L - 1; i < N; ++i) {
				mcols *= t.dimension(i);
			}

			// 適当に行列にマップ
			Eigen::Map<const MatrixType> mt(t.data(), mrows, mcols);

			// SVD 実行
			SVDSolver svd(mt, computationOptions);
			const MatrixType& U = svd.matrixU();
			const MatrixType& V = svd.matrixV();
			s = svd.singularValues();

			// 分解後のテンソルの形状取得
			Eigen::array<int, L> shapeL;
			Eigen::array<int, R> shapeR;
			for (int i = 0; i < L - 1; ++i) {
				shapeL[i] = t.dimension(i);
			}
			shapeL[L - 1] = U.cols();

			for (int i = 0; i < R - 1; ++i) {
				shapeR[i] = t.dimension(L - 1 + i);
			}
			shapeR[R - 1] = V.cols();

			// 分解後のテンソルを格納
			tL.resize(shapeL);
			tR.resize(shapeR);
			Eigen::Map<MatrixType> mL(tL.data(), U.rows(), U.cols());
			Eigen::Map<MatrixType> mR(tR.data(), V.rows(), V.cols());
			mL = U;
			mR = V.conjugate();
		}


		/// <summary>
		/// 特異値のサイズを指定してTensor を特異値分解する
		/// その際に切り捨てた truncation_error を習得する
		/// truncation_error は切り捨てた特異値の2乗和の平方根で定義される
		/// 絶対スケールで取得する
		/// </summary>
		template<class Scalar, int N, int L>
		static void tensorSVD(
			const Eigen::Tensor<Scalar, N>& t,
			int singular_value_size,
			Eigen::Tensor<Scalar, L>& tL,
			Eigen::Tensor<Scalar, N - L + 2>& tR,
			Eigen::Matrix<typename Eigen::NumTraits<Scalar>::Real, Eigen::Dynamic, 1>& s,
			typename Eigen::NumTraits<Scalar>::Real& truncation_error
		) {
			using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
			using RealVectorType = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;


			constexpr int R = N - L + 2;

			Eigen::Tensor<Scalar, L> tL_;
			Eigen::Tensor<Scalar, N - L + 2> tR_;
			Eigen::Matrix<Scalar, Eigen::Dynamic, 1> s_;

			compute(t, tL_, tR_, s_);

			Eigen::array<int, L> shapeL = tL_.dimensions();
			shapeL[L - 1] = singular_value_size;
			tL = zerowiselyResized(tL_, shapeL);
			Eigen::array<int, L> shapeR = tR_.dimensions();
			shapeR[R - 1] = singular_value_size;
			tR = zerowiselyResized(tR_, shapeR);

			s = RealVectorType::Zero(singular_value_size);
			for (int i = 0, ni = min_value(singular_value_size, s_.size()); i < ni; ++i) {
				s[i] = s_[i];
			}
			truncation_error = RealScalar(0.0);
			for (int i = min_value(singular_value_size, s_.size()), ni = s_.size(); i < ni; ++i) {
				truncation_error += std::real(s_[i] * std::conj(s_[i]));
			}
			truncation_error = std::sqrt(truncation_error);
		}



		/// <summary>
		/// 特異値のサイズを指定してTensor を特異値分解する
		/// </summary>
		template<class Scalar, int N, int L>
		static void tensorSVD(
			const Eigen::Tensor<Scalar, N>& t,
			int singular_value_size,
			Eigen::Tensor<Scalar, L>& tL,
			Eigen::Tensor<Scalar, N - L + 2>& tR,
			Eigen::Matrix<typename Eigen::NumTraits<Scalar>::Real, Eigen::Dynamic, 1>& s
		) {
			typename Eigen::NumTraits<Scalar>::Real truncation_error = 0.0;
			compute(
				t, singular_value_size, tL, tR, s, truncation_error
			);
		}

	}
}

/// <summary>
/// TensorSVD
/// </summary>
namespace cmpt {
	namespace EigenEx {

		/// <summary>
		/// Eigen::Tensor に対する SVD を扱う
		/// 軸がソートされているテンソルを分解し、軸がソートされているテンソルを返す
		/// 仕様は基本的には Eigen::BDCSVD に従う
		/// しかし、threshold 周りは変更している
		/// また、V のテンソルの定義が少し違う(adjoint をとらない)
		///  
		/// テンプレート引数の例
		/// 例1) TensorSVD{Eigen::Tensor{Scalar,4},2,2}
		/// T(i0,i1,i2,i3) == U(i0,i1,is) S(is,is) V(i2,i3,is)
		/// 
		/// 例2) TensorSVD{Eigen::Tensor{Scalar,4},3,1}
		/// t(i0,i1,i2,i3) == U(i0,i1,i2,is) S(is,is) V(i3,is)
		/// </summary>
		template<class TensorT_, int Urow_, int Vrow_>
		class TensorSVD {
		public:	// タイプエイリアス、コンパイル時変数

			using Index = int;
			using Axis = int;
			template<Index N_>
			using Indices = Eigen::array<Axis, N_>;

			static constexpr Axis NT = EigenEx::TensorTraits<TensorT_>::NumDimensions;
			static constexpr Axis NU = Urow_ + 1;
			static constexpr Axis NV = Vrow_ + 1;

			using TensorT = TensorT_;
			using Scalar = typename TensorT::Scalar;
			using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
			using TensorU = Eigen::Tensor<Scalar, NU>;
			using TensorV = Eigen::Tensor<Scalar, NV>;

			using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
			using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
			using RealVectorType = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
			using SingularValuesType = RealVectorType;

			using SVDSolver = Eigen::BDCSVD<MatrixType>;

			// static assert
			static_assert(NT == Urow_ + Vrow_, "in TensorSVD<TensorT_,Urow_,Vrow_>, NT == Urow_+Vrow_ is violated");

		protected:	// メンバ変数
			unsigned int computationOptions_;
			bool isComputed_;

			TensorU tU_;
			TensorV tV_;
			SingularValuesType sv_;

		public:	// アクセッサ

			bool isComputed()const { return isComputed_; }

			const TensorU& tensorU()const { return tU_; }
			const TensorV& tensorV()const { return tV_; }
			const SingularValuesType& singularValues()const { return sv_; }

			/// <summary>
			/// 計算データを破棄して初期化する
			/// </summary>
			TensorSVD& setAllSettingsDefault() {
				computationOptions_ = 0;
				isComputed_ = false;
				tU_ = TensorU();
				tV_ = TensorV();
				sv_ = SingularValuesType();
				return *this;
			}



		public:	// コンストラクタ

			TensorSVD() {
				setAllSettingsDefault();
			}

			TensorSVD(
				const TensorT& tT,
				unsigned int computationOptions = 0
			) {
				setAllSettingsDefault();
				compute(tT, computationOptions);
			}


			/// <summary>
			/// computationOptions を指定して SVD を行う
			/// computationOptions は Eigen::SVDBase の場合と同じようにやる
			/// 多くの場合 computationOptions = Eigen::ComputeThinU|Eigen::ComputeThinV
			/// </summary>
			TensorSVD& compute(
				const TensorT& tT,
				unsigned int computationOptions
			) {
				computationOptions_ = computationOptions;
				compute(tT);
				return *this;
			}


			TensorSVD& compute(
				const TensorT& tT
			) {

				// 行列のサイズ取得
				int mrows = 1;
				int mcols = 1;
				for (int i = 0; i < NU - 1; ++i) {
					mrows *= tT.dimension(i);
				}
				for (int i = NU - 1; i < NT; ++i) {
					mcols *= tT.dimension(i);
				}

				// 適当に行列にマップ
				Eigen::Map<const MatrixType> mtT(tT.data(), mrows, mcols);

				// SVD 実行
				if (computationOptions_ != 0) {
					SVDSolver svd(mtT, computationOptions_);
					const MatrixType& U = svd.matrixU();
					const MatrixType& V = svd.matrixV();
					sv_ = svd.singularValues();

					// 分解後のテンソルの形状取得
					Indices<NU> shapeL;
					Indices<NV> shapeR;
					for (int i = 0; i < NU - 1; ++i) {
						shapeL[i] = tT.dimension(i);
					}
					shapeL[NU - 1] = U.cols();

					for (int i = 0; i < NV - 1; ++i) {
						shapeR[i] = tT.dimension(NU - 1 + i);
					}
					shapeR[NV - 1] = V.cols();

					// 分解後のテンソルを格納
					tU_.resize(shapeL);
					tV_.resize(shapeR);
					Eigen::Map<MatrixType> mL(tU_.data(), U.rows(), U.cols());
					Eigen::Map<MatrixType> mR(tV_.data(), V.rows(), V.cols());
					mL = U;
					mR = V.conjugate();

					isComputed_ = true;
				}
				else {
					isComputed_ = false;
				}

				return *this;
			}


			/// <summary>
			/// threshold 以上の値を持つ SingularValues の数を返す
			/// </summary>
			Index getRank(RealScalar threshold)const {
				if (isComputed_) {
					for (Index i = 0, ni = sv_.size(); i < ni; ++i) {
						if (std::abs(sv_[i]) <= threshold) {
							return i;
						}
					}
					return sv_.size();
				}
				else {
					return 0;
				}
			}

			/// <summary>
			/// threshold 以上の特異値に対応する tensorU を返す
			/// コピーコストが生じる
			/// </summary>
			TensorU getTruncatedTensorU(RealScalar threshold)const {
				return getTruncatedTensorU(getRank(threshold));
			}

			/// <summary>
			/// threshold 以上の特異値に対応する tensorV を返す
			/// コピーコストが生じる
			/// </summary>
			TensorV getTruncatedTensorV(RealScalar threshold)const {
				return getTruncatedTensorV(getRank(threshold));
			}

			/// <summary>
			/// threshold 以上の特異値に対応する SingularValues を返す
			/// コピーコストが生じる
			/// </summary>
			SingularValuesType getTruncatedSingularValues(RealScalar threshold)const {
				return getTruncatedSingularValues(getRank(threshold));
			}


			/// <summary>
			/// 特異値の数を指定して tensorU を返す
			/// 特異値が計算されたテンソルより大きい場合にはゼロ埋めされる
			/// コピーコストが生じる
			/// </summary>
			TensorU getTruncatedTensorU(Index rank)const {
				Indices<NU> shapeU;
				for (int i = 0, ni = NU; i < ni; ++i) {
					shapeU[i] = static_cast<int>(tU_.dimension(i));
				}
				shapeU[NU - 1] = rank;
				TensorU ttU = EigenEx::zerowiselyResized<Scalar, NU>(tU_, shapeU);
				return ttU;
			}

			/// <summary>
			/// 特異値の数を指定して tensorV を返す
			/// 特異値が計算されたテンソルより大きい場合にはゼロ埋めされる
			/// コピーコストが生じる
			/// </summary>
			TensorV getTruncatedTensorV(Index rank)const {
				Indices<NV> shapeV;
				for (int i = 0, ni = NU; i < ni; ++i) {
					shapeV[i] = static_cast<int>(tV_.dimension(i));
				}
				shapeV[NV - 1] = rank;
				TensorU ttV = EigenEx::zerowiselyResized<Scalar, NV>(tV_, shapeV);
				return ttV;
			}

			/// <summary>
			/// 特異値の数を指定して singularValues を返す
			/// 特異値が計算されたテンソルより大きい場合にはゼロ埋めされる
			/// コピーコストが生じる
			/// </summary>
			SingularValuesType getTruncatedSingularValues(Index rank)const {
				SingularValuesType sv(rank);
				Index mi = min_value(sv.size(), sv_.size());
				for (Index i = 0; i < mi; ++i) {
					sv[i] = sv_[i];
				}
				for (Index i = mi, ni = sv.size(); i < ni; ++i) {
					sv[i] = RealScalar(0.0);
				}
				return sv;
			}



		};
	}

}
