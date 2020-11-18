#pragma once

#include <array>
#include <vector>

#include "cmpt/eigen_ex/multi_indices.hpp"
#include "cmpt/eigen_ex/tensor_kronecker_product.hpp"


namespace cmpt {

	/// <summary>
	/// EigenEx::contract(...).from(...).to(...)
	/// の実装
	/// 
	/// 現在 2テンソルの単純な縮約のみ実装している
	/// 
	/// (おそらく) einsum より速い場合があるが einsum より一般性が小さい
	/// 
	/// つぎの縮約は追って実装する
	/// 
	/// また、例外などチェックしていない
	/// 
	/// </summary>
	namespace EigenEx {


		// テンソルの添字を識別するための型
		using IIndex = std::string;


		/// <summary>
		/// 2テンソルの単純な縮約の実装
		/// 3テンソル以上の縮約や縮約以外の和には対応していない
		/// 例
		/// contract(A,B).from({i,j,k},{l,j}).to(i,l,k)
		/// 
		/// </summary>
		template<class TensorA_, class TensorB_, class IIndicesR_>
		class TwoTensorPureContraction {
		public:
			// タイプエイリアス

			using TensorA = TensorA_;
			using TensorB = TensorB_;

			using ScalarA = typename TensorA::Scalar;
			using ScalarB = typename TensorB::Scalar;

			static constexpr int NA = EigenEx::TensorTraits<TensorA>::NumDimensions;
			static constexpr int NB = EigenEx::TensorTraits<TensorB>::NumDimensions;
			static constexpr int NR = std::tuple_size<IIndicesR_>::value;
			static constexpr int NP = (NA + NB - NR) / 2;

			using ScalarR = typename std::remove_reference<decltype(ScalarA()* ScalarB())>::type;
			using TensorR = Eigen::Tensor<ScalarR, NR>;

			using IIndicesA = std::array<IIndex, NA>;
			using IIndicesB = std::array<IIndex, NB>;
			using IIndicesR = std::array<IIndex, NR>;  // ==ToIIndices

			using IndicesA = std::array<int, NA>;
			using IndicesB = std::array<int, NB>;
			using IndicesR = std::array<int, NR>;



			const TensorA& tensorA;
			const TensorB& tensorB;
			const IIndicesA& iiA;
			const IIndicesB& iiB;
			const IIndicesR& iiR;




			/// <summary>
			/// IIndicesX 型の情報を string 形式で取得するための関数
			/// </summary>
			template<class IIndices>
			static std::string iIndicesToSting(const IIndices& iis) {
				std::string str = "";
				int n = iis.size();
				str += "[";
				for (int i = 0; i < n - 1; ++i) {
					str += iis[i];
					str += ",";
				}
				str += iis[n - 1];
				str += "]";
				return str;
			}


			/// <summary>
			/// インデックスの情報の最小限を string で取得する関数
			/// </summary>
			/// <returns></returns>
			std::string makeIIndicesInfo() {
				std::string str;
				str += "from(";
				str += iIndicesToSting(iiA);
				str += ",";
				str += iIndicesToSting(iiB);
				str += ").to(";
				str += iIndicesToSting(iiR);
				str += ")";
				return str;
			}

			template<class IIndices>
			static std::map<IIndex, int> makeNumII(const IIndices& iiX) {
				std::map<IIndex, int> numII;
				for (auto& ii : iiX) {
					auto itr = numII.find(ii);
					if (itr == numII.end()) {
						numII[ii] = 1;
					}
					else {
						numII[ii] += 1;
					}
				}
				return numII;
			}


			/// <summary>
			/// [numIIA,numIIB,numIIR,numIIP]
			/// を作成して返す
			/// </summary>
			static std::array<std::map<IIndex, int>, 4> makeNumIIs(
				const IIndicesA& iiA,
				const IIndicesB& iiB,
				const IIndicesR& iiR
			) {
				auto numIIA = makeNumII(iiA);
				auto numIIB = makeNumII(iiB);
				auto numIIR = makeNumII(iiR);
				auto numIIP = [&numIIA, &numIIB, &numIIR]()->std::map<IIndex, int> {
					std::map<IIndex, int> numII;
					for (auto& iia_ : numIIA) {
						IIndex ii = iia_.first;
						int n = iia_.second;
						auto itr = numII.find(ii);
						if (itr == numII.end()) {
							numII[ii] = n;
						}
						else {
							numII[ii] += n;
						}
					}
					for (auto& iib_ : numIIB) {
						IIndex ii = iib_.first;
						int n = iib_.second;
						auto itr = numII.find(ii);
						if (itr == numII.end()) {
							numII[ii] = n;
						}
						else {
							numII[ii] += n;
						}
					}
					for (auto& iir_ : numIIR) {
						IIndex ii = iir_.first;
						int n = iir_.second;
						auto itr = numII.find(ii);
						if (itr == numII.end()) {
							numII[ii] = -n;
						}
						else {
							numII[ii] += -n;
						}
					}
					return numII;
				}();

				return { numIIA,numIIB,numIIR,numIIP };
			}

			std::array<std::map<IIndex, int>, 4> makeNumIIs()const {
				return makeNumIIs(iiA, iiB, iiR);
			}


			// 2テンソルの単純なコントラクションであるか判定する
			static bool isValid(
				const IIndicesA& iiA,
				const IIndicesB& iiB,
				const IIndicesR& iiR
			) {
				auto numIIs = makeNumIIs(iiA, iiB, iiR);


				auto numIIA = numIIs[0];
				auto numIIB = numIIs[1];
				auto numIIR = numIIs[2];
				auto numIIP = numIIs[3];


				bool isValid = [&numIIP]() {
					bool isVal = true;
					for (auto& n_ : numIIP) {
						auto& ii = n_.first;
						auto& n = n_.second;
						if (n != 2 && n != 0) {
							isVal = false;
							break;
						}
					}
					return isVal;
				}();

				return isValid;
			}


			bool isValid()const {
				return isValid(iiA, iiB, iiR);
			}



			Eigen::array<Eigen::IndexPair<int>, NP> makePairs()const {

				std::array<std::map<IIndex, int>, 4> numIIs = makeNumIIs();
				std::map<IIndex, int> numIIA = numIIs[0];
				std::map<IIndex, int> numIIB = numIIs[1];
				std::map<IIndex, int> numIIR = numIIs[2];
				std::map<IIndex, int> numIIP = numIIs[3];


				Eigen::array<Eigen::IndexPair<int>, NP> prs;
				int i = 0;
				for (auto& ii_ : numIIP) {
					auto& ii = ii_.first;
					auto& n = ii_.second;
					if (n == 2) {
						auto itrA = std::find(iiA.begin(), iiA.end(), ii);
						int iA = std::distance(iiA.begin(), itrA);
						auto itrB = std::find(iiB.begin(), iiB.end(), ii);
						int iB = std::distance(iiB.begin(), itrB);
						prs[i] = Eigen::IndexPair<int>{ iA,iB };
						++i;
						if (i == prs.size()) {
							break;
						}
					}
				}
				return prs;
			}


			Eigen::array<int, NR> makeShuffle()const {

				std::array<std::map<IIndex, int>, 4> numIIs = makeNumIIs();
				std::map<IIndex, int> numIIA = numIIs[0];
				std::map<IIndex, int> numIIB = numIIs[1];
				std::map<IIndex, int> numIIR = numIIs[2];
				std::map<IIndex, int> numIIP = numIIs[3];



				std::array<IIndex, NR> iiR_raw;	// shuffle しない場合のインデックスの情報を作成
				{
					int i = 0;
					for (int iA = 0; iA < NA; ++iA) {
						auto itr = numIIP.find(iiA[iA]);
						if (itr != numIIP.end()) {
							IIndex ii = itr->first;
							int n = itr->second;
							if (n == 0) {
								iiR_raw[i] = iiA[iA];
								++i;
								if (i >= NR) { break; }
							}
						}
					}
					for (int iB = 0; iB < NB; ++iB) {
						auto itr = numIIP.find(iiB[iB]);
						if (itr != numIIP.end()) {
							IIndex ii = itr->first;
							int n = itr->second;
							if (n == 0) {
								iiR_raw[i] = iiB[iB];
								++i;
								if (i >= NR) { break; }
							}
						}
					}












					for (auto& iiR_ : numIIR) {
						auto& ii = iiR_.first;
						auto iA = std::distance(iiA.begin(), std::find(iiA.begin(), iiA.end(), ii));
						auto iB = std::distance(iiB.begin(), std::find(iiB.begin(), iiB.end(), ii));


						if (iA < NA) {

						}

					}
				}

				Eigen::array<int, NR> shfl;
				for (int i = 0; i < NR; ++i) {
					shfl[i] = std::distance(iiR_raw.begin(), std::find(iiR_raw.begin(), iiR_raw.end(), iiR[i]));
				}

				return shfl;

			}


			TensorR compute() {

				// 例外チェック
				if (!isValid()) {
					std::string str;
					str += "In EigenEx::contract(...).from(...).to(...), invalid indices \n";
					str += "IndicesInfo:\n";
					str += makeIIndicesInfo();
					str += "\n";
					throw RuntimeException(str.c_str());
					return TensorR();
				}


				auto pairs = makePairs();

				auto shuffle = makeShuffle();

				return tensorA.contract(tensorB, pairs).shuffle(shuffle);

			}



		};



		/// <summary>
		/// contraction の情報を保持するクラス
		/// 現在 2テンソルの単純な縮約の実装 のみしている
		/// </summary>
		template<class TensorA_, class TensorB_, class IIndicesR_>
		class TwoToTensorRef {
		public:

			using TensorA = TensorA_;
			using TensorB = TensorB_;

			using ScalarA = typename TensorA::Scalar;
			using ScalarB = typename TensorB::Scalar;

			static constexpr int NA = EigenEx::TensorTraits<TensorA>::NumDimensions;
			static constexpr int NB = EigenEx::TensorTraits<TensorB>::NumDimensions;
			static constexpr int NR = std::tuple_size<IIndicesR_>::value;
			static constexpr int NP = (NA + NB - NR) / 2;

			using ScalarR = typename std::remove_reference<decltype(ScalarA()* ScalarB())>::type;
			using TensorR = Eigen::Tensor<ScalarR, NR>;

			using IIndicesA = std::array<IIndex, NA>;
			using IIndicesB = std::array<IIndex, NB>;
			using IIndicesR = std::array<IIndex, NR>;  // ==ToIIndices

			using IndicesA = std::array<IIndex, NA>;
			using IndicesB = std::array<IIndex, NB>;
			using IndicesR = std::array<IIndex, NR>;


			const TensorA& tensorA;
			const TensorB& tensorB;
			const IIndicesA& iiA;
			const IIndicesB& iiB;
			const IIndicesR& iiR;


			template<class IIndices>
			static std::string iIndicesToSting(const IIndices& iis) {
				std::string str = "";
				int n = iis.size();
				str += "[";
				for (int i = 0; i < n - 1; ++i) {
					str += iis[i];
					str += ",";
				}
				str += iis[n - 1];
				str += "]";
				return str;
			}

			std::string makeIIndicesInfo() {
				std::string str;
				str += "from(";
				str += iIndicesToSting(iiA);
				str += ",";
				str += iIndicesToSting(iiB);
				str += ").to(";
				str += iIndicesToSting(iiR);
				str += ")";
				return str;
			}




			TensorR compute() {
				if (
					TwoTensorPureContraction<TensorA, TensorB, IIndicesR>{tensorA, tensorB, iiA, iiB, iiR}.isValid()
					) {
					return TwoTensorPureContraction<TensorA, TensorB, IIndicesR>{tensorA, tensorB, iiA, iiB, iiR}.compute();
				}
				else {
					std::string str;
					str += "In EigenEx::contract(...).from(...).to(...), invalid indices \n";
					str += "IndicesInfo:\n";
					str += makeIIndicesInfo();
					str += "\n";
					throw RuntimeException(str.c_str());
					return TensorR();
				}

			}


		};

		/// <summary>
		/// contraction の情報を構成するための中間クラス
		/// </summary>
		template<class TensorA_, class TensorB_>
		class TwoFromTensorRef {
		public:


			using TensorA = TensorA_;
			using TensorB = TensorB_;


			static constexpr int NA = EigenEx::TensorTraits<TensorA>::NumDimensions;
			static constexpr int NB = EigenEx::TensorTraits<TensorB>::NumDimensions;
			using IIndicesA = typename std::array<IIndex, NA>;
			using IIndicesB = typename std::array<IIndex, NB>;


			const TensorA& tensorA;
			const TensorB& tensorB;

			const IIndicesA& iiA;
			const IIndicesB& iiB;




			template<class Str, int NR>
			typename TwoToTensorRef<TensorA, TensorB, std::array<IIndex, NR>>::TensorR to(const Str(&ar)[NR])const {
				std::array<IIndex, NR> iiR;
				for (int i = 0; i < NR; ++i) {
					iiR[i] = ar[i];
				}
				return TwoToTensorRef<TensorA, TensorB, std::array<IIndex, NR>>{tensorA, tensorB, iiA, iiB, iiR}.compute();
			}




		};

		/// <summary>
		/// /// contraction の情報を構成するための中間クラス
		/// </summary>
		template<class TensorA_, class TensorB_>
		class TwoContractedTensorRef {
		public:

			using TensorA = TensorA_;
			using TensorB = TensorB_;
			static constexpr int NA = EigenEx::TensorTraits<TensorA>::NumDimensions;
			static constexpr int NB = EigenEx::TensorTraits<TensorB>::NumDimensions;
			using IIndicesA = std::array<IIndex, NA>;
			using IIndicesB = std::array<IIndex, NB>;

			const TensorA& tensorA;
			const TensorB& tensorB;

			TwoFromTensorRef<TensorA, TensorB> from(const IIndicesA& ia, const IIndicesB& ib) {
				return TwoFromTensorRef<TensorA, TensorB>{tensorA, tensorB, ia, ib };
			}


		};



		/// <summary>
		/// contraction を書くための関数
		/// 
		/// 使い方としては
		/// contract(A,B).from({"i","j"},{"j","k"}).to({"i","k"})
		/// のような使い方を想定する
		/// 
		/// 
		/// </summary>
		template<class TensorA, class TensorB>
		TwoContractedTensorRef<TensorA, TensorB> contract(const TensorA& ta, const TensorB& tb) {
			return TwoContractedTensorRef<TensorA, TensorB>{ ta, tb};
		}




	}






	/// <summary>
	/// einsum の実装
	/// einsum のインターフェースは tuple にして一般的にする
	/// 必要な拡張は必要になったときにする
	/// </summary>
	namespace EigenEx {



		/// <summary>
		/// einsum の情報の型伝搬の実装
		/// 
		/// 現在 1テンソルもしくは2テンソルの縮約のみ実装している
		/// 
		/// (example)
		/// einsum(A,B).from({i,j},{i}).to({i,j})
		/// 
		/// </summary>
		namespace einsum_impl {
			using Index = integer_type::Index;
			using Axis = integer_type::Axis;
			using IIndex = std::string;

			/// <summary>
			/// IIndices を string に変換する関数
			/// </summary>
			template<Axis N>
			inline std::string IIndicesToString(const std::array<IIndex, N>& iiT) {
				std::stringstream ss;
				ss << "[";
				for (Axis d = 0; d < N - 1; ++d) {
					ss << iiT[d] << ",";
				}
				if (N - 1 >= 0) {
					ss << iiT[N - 1];
				}
				ss << "]";
				return ss.str();
			}


			/// <summary>
			/// IIndices から numII (各IInddex の数)を取得する関数
			/// </summary>
			template<class IIndicesX>
			static std::map<IIndex, Index> makeNumII(const IIndicesX& iiX) {
				std::map<IIndex, Index> numII;
				for (auto& ii : iiX) {
					auto itr = numII.find(ii);
					if (itr == numII.end()) {
						numII[ii] = 1;
					}
					else {
						numII[ii] += 1;
					}
				}
				return numII;
			}



			/// <summary>
			/// 実際に評価するクラス
			/// 目的の方に応じて特殊化する
			/// 
			/// 特殊化した場合に以下の条件を満たす必要がある
			/// 
			/// クラステンプレート引数 CRTensors_ は 
			/// std::tuple{const TensorType&, ... }
			/// で表現される型である必要がある
			/// 
			/// クラステンプレート引数 IIndiceses_ は
			/// std::tuple{std::array{IIndex,N}, ... }
			/// で表現される型である必要がある
			/// 
			/// クラステンプレート引数 IIndiceses_ は
			/// std::array{IIndex,NR}
			/// で表現される型である必要がある
			/// 
			/// タイプエイリアス TensorR を持つ
			/// 
			/// public メンバ関数
			/// TensorR compute()const;
			/// を持つ必要がある
			/// </summary>
			template<class CRTensors_, class IIndiceses_, class IIndicesR_>
			class ToImpl {
			};


			/// <summary>
			/// einsum(...).from(...) が返すクラス
			/// .to(...) を実装している
			/// </summary>
			template<class CRTensors_, class IIndiceses_>
			class FromImpl {
			public:
				using CRTensors = CRTensors_;
				using IIndiceses = IIndiceses_;

				CRTensors ts;
				IIndiceses iiTs;



				template<Axis NR>
				typename ToImpl<CRTensors, IIndiceses, std::array<IIndex, NR>>::TensorR to(
					const std::array<IIndex, NR>& iiR
				)const {
					ToImpl<CRTensors, IIndiceses, std::array<IIndex, NR>> toImpl{ ts, iiTs, iiR };
					return toImpl.compute();
				}

				template<class Str, Axis NR>
				typename ToImpl<CRTensors, IIndiceses, std::array<IIndex, NR>>::TensorR to(
					const Str(&ar)[NR]
				)const {
					std::array<IIndex, NR> iiR;
					for (Axis d = 0; d < NR; ++d) {
						iiR[d] = ar[d];
					}

					return to(iiR);
				}


			};

			/// <summary>
			/// einsum(...) が返すクラス
			/// .from(...) を実装している
			/// </summary>
			template<class CRTensors_>
			class EinsumImpl {
			public:
				using CRTensors = CRTensors_;
				static constexpr Axis NumTensors = std::tuple_size<CRTensors>::value;

				CRTensors ts;

				template<Axis NA>
				FromImpl<CRTensors, std::tuple<std::array<IIndex, NA>>> from(
					const std::array<IIndex, NA>& iiT
				)const {
					std::tuple<std::array<IIndex, NA>> iiTs{ iiT };
					return FromImpl<CRTensors, std::tuple<std::array<IIndex, NA>>>{ts, iiTs};
				}


				template<Axis NA, Axis NB>
				FromImpl<CRTensors, std::tuple<std::array<IIndex, NA>, std::array<IIndex, NB>>> from(
					const std::array<IIndex, NA>& iiA,
					const std::array<IIndex, NB>& iiB
				)const {
					std::tuple<std::array<IIndex, NA>, std::array<IIndex, NB>> iiTs{ iiA,iiB };
					return FromImpl<CRTensors, std::tuple<std::array<IIndex, NA>, std::array<IIndex, NB>> >{ts, iiTs};
				};



				template<class StrA, Axis NA>
				FromImpl<CRTensors, std::tuple<std::array<IIndex, NA>>> from(
					const StrA(&ar)[NA]
				) {
					std::array<IIndex, NA> iiA;
					for (Axis d = 0; d < NA; ++d) {
						iiA[d] = ar[d];
					}
					std::tuple<std::array<IIndex, NA>> iiTs{ iiA };
					return FromImpl<CRTensors, std::tuple<std::array<IIndex, NA>>>{ts, iiTs};
				}


				template<class StrA, Axis NA, class StrB, Axis NB>
				FromImpl<CRTensors, std::tuple<std::array<IIndex, NA>, std::array<IIndex, NB>>> from(
					const StrA(&arA)[NA],
					const StrB(&arB)[NB]
				) {
					std::array<IIndex, NA> iiA;
					for (Axis d = 0; d < NA; ++d) {
						iiA[d] = arA[d];
					}
					std::array<IIndex, NB> iiB;
					for (Axis d = 0; d < NB; ++d) {
						iiB[d] = arB[d];
					}
					std::tuple<std::array<IIndex, NA>, std::array<IIndex, NB>> iiTs{ iiA,iiB };
					return FromImpl<CRTensors, std::tuple<std::array<IIndex, NA>, std::array<IIndex, NB>>>{ts, iiTs};
				}

			};





		}


		template<class Tensor>
		inline static einsum_impl::EinsumImpl<std::tuple<const Tensor&>> einsum(const Tensor& t) {
			std::tuple<const Tensor&> ts{ t };
			return einsum_impl::EinsumImpl<std::tuple<const Tensor&>>{ ts};
		}

		template<class TensorA, class TensorB>
		inline static einsum_impl::EinsumImpl<std::tuple<const TensorA&, const TensorB&>> einsum(const TensorA& tA, const TensorB& tB) {
			std::tuple<const TensorA&, const TensorB&> ts{ tA,tB };
			return einsum_impl::EinsumImpl<std::tuple<const TensorA&, const TensorB&>>{ ts };
		}





		/// <summary>
		/// 現在 Eigen::Tensor, Eigen::Tensor
		/// 現在 Eigen::Tensor
		/// の実装の特殊化を実装している
		/// </summary>
		namespace einsum_impl {


			/// <summary>
			/// ToImpl の 2テンソル縮約への特殊化
			/// 
			/// Eigen::Tensor{} 型を対象としている
			/// </summary>
			template<class CRTensorA_, class CRTensorB_, class CRIIndicesA_, class CRIIndicesB_, class IIndicesR_>
			class ToImpl<std::tuple<CRTensorA_, CRTensorB_>, std::tuple<CRIIndicesA_, CRIIndicesB_>, IIndicesR_> {
			public:
				using TensorA = typename std::remove_const<typename std::remove_reference<CRTensorA_>::type>::type;
				using TensorB = typename std::remove_const<typename std::remove_reference<CRTensorB_>::type>::type;

				using IIndicesA = typename std::remove_const<typename std::remove_reference<CRIIndicesA_>::type>::type;
				using IIndicesB = typename std::remove_const<typename std::remove_reference<CRIIndicesB_>::type>::type;
				using IIndicesR = IIndicesR_;

				static constexpr Axis NA = std::tuple_size<IIndicesA>::value;
				static constexpr Axis NB = std::tuple_size<IIndicesB>::value;
				static constexpr Axis NR = std::tuple_size<IIndicesR>::value;

				using ScalarA = typename TensorA::Scalar;
				using ScalarB = typename TensorB::Scalar;
				using ScalarR = typename std::remove_reference<decltype(ScalarA()* ScalarB())>::type;

				using TensorR = Eigen::Tensor<ScalarR, NR>;


				std::tuple<const TensorA&, const TensorB&> ts;
				std::tuple<IIndicesA, IIndicesB> iiTs;
				IIndicesR iiR;





			public:

				bool isValid()const {
					const TensorA& tA = std::get<0>(ts);
					const TensorB& tB = std::get<1>(ts);
					const IIndicesA& iiA = std::get<0>(iiTs);
					const IIndicesB& iiB = std::get<1>(iiTs);

					bool is_valid = true;

					// 軸の対応が妥当かどうか判定
					auto numIIA = makeNumII(iiA);
					auto numIIB = makeNumII(iiB);
					auto numIIR = makeNumII(iiR);
					for (auto& niiR_ : numIIR) {
						IIndex key = niiR_.first;
						auto itrA = numIIA.find(key);
						auto itrB = numIIB.find(key);
						bool isNotFound = (itrA == numIIA.end()) && (itrB == numIIB.end());
						if (isNotFound) {
							is_valid = false;
							break;
						}
					}

					// 同一軸のサイズ比較
					std::array<IIndex, NA + NB> iiAB;
					for (Axis dA = 0; dA < NA; ++dA) {
						iiAB[dA] = iiA[dA];
					}
					for (Axis dB = 0; dB < NB; ++dB) {
						iiAB[dB] = iiA[dB];
						iiAB[NA + dB] = iiB[dB];
					}

					for (Axis dAB = 0; dAB < NA + NB; ++dAB) {
						for (Axis dAB_ = 0; dAB_ < NA + NB; ++dAB_) {
							if (iiAB[dAB] == iiAB[dAB_]) {
								Axis dim;
								Axis dim_;
								if (dAB < NA) {
									dim = tA.dimension(dAB);
								}
								else {
									dim = tB.dimension(dAB - NA);
								}
								if (dAB_ < NA) {
									dim_ = tA.dimension(dAB_);
								}
								else {
									dim_ = tB.dimension(dAB_ - NA);
								}
								if (dim != dim_) {
									is_valid = false;
								}
							}
						}
					}

					return is_valid;
				}


				std::string makeInfo()const {
					std::string str;
					str += "from(";
					str += IIndicesToString(std::get<0>(iiTs));
					str += ",";
					str += IIndicesToString(std::get<1>(iiTs));
					str += ").to(";
					str += IIndicesToString(iiR);
					str += ")";

					return str;
				}



				TensorR compute()const {
					const TensorA& tA = std::get<0>(ts);
					const TensorB& tB = std::get<1>(ts);
					const IIndicesA& iiA = std::get<0>(iiTs);
					const IIndicesB& iiB = std::get<1>(iiTs);


					// 例外
					if (isValid() == false) {
						std::string str;
						str += "invalid indices for cmpt::EigenEx::eimsum(...).form(...).to(...)\n";
						str += "with " + makeInfo();
						throw RuntimeException(str.c_str());
						return TensorR();
					}

					// AとBを結合
					std::array<IIndex, NA + NB> iiAB;
					for (Axis d = 0; d < NA; ++d) {
						iiAB[d] = iiA[d];
					}
					for (Axis d = 0; d < NB; ++d) {
						iiAB[NA + d] = iiB[d];
					}

					auto kpAB = EigenEx::tensorKroneckerProduct(tA, tB);

					// 軸をシャッフルする
					// for [ 計算後に残すインデックス | 和をとって消えるインデックス | その他のインデックス(対角として統合される) ]
					std::array<Axis, NA + NB> spi_shuffle;

					// 計算後に残すインデックスの軸 [0,NR) を設定
					for (Axis d = 0; d < NA + NB; ++d) {
						spi_shuffle[d] = -1;
					}
					for (Axis dR = 0; dR < NR; ++dR) {
						for (Axis dAB = 0; dAB < NA + NB; ++dAB) {
							if (iiAB[dAB] == iiR[dR]) {
								spi_shuffle[dR] = dAB;
								break;
							}
						}
					}

					// 和をとって消えるインデックスの軸 [NR,NR+NC) を設定
					Axis NC = 0;
					{
						Axis d = NR;
						for (Axis dAB = 0; dAB < NA + NB; ++dAB) {
							bool isC = true;
							for (Axis dR = 0; dR < NR; ++dR) {
								if (iiR[dR] == iiAB[dAB]) {
									isC = false;
									break;
								}
							}
							for (Axis dC = NR; dC < NR + NC; ++dC) {
								if (iiAB[spi_shuffle[dC]] == iiAB[dAB]) {
									isC = false;
									break;
								}
							}

							if (isC) {
								spi_shuffle[d] = dAB;
								++d;
								++NC;
							}
						}
					}

					// 対角として統合されるインデックスの軸 [NR+NC,NA+NB) を設定
					{
						Axis d = NR + NC;
						for (Axis dAB = 0; dAB < NA + NB; ++dAB) {
							bool is_sum = true;
							for (Axis dRC = 0; dRC < NR + NC; ++dRC) {
								if (dAB == spi_shuffle[dRC]) {
									is_sum = false;
									break;
								}
							}
							if (is_sum) {
								spi_shuffle[d] = dAB;
								++d;
							}
						}
					}




					// 軸をシャッフルした ProductIndices を生成
					auto spi = kpAB.productIndices().shuffle(spi_shuffle);
					std::array<IIndex, NA + NB> iisAB;
					for (Axis d = 0; d < NA + NB; ++d) {
						iisAB[d] = iiAB[spi_shuffle[d]];
					}

					// 軸の一部を対角で評価して消す
					// [ 計算後に残すインデックス | コントラクションに使うインデックス | その他のインデックス ]
					// における [その他のインデックス] を前２つに統合する

					std::vector<Slice> slices_sspi;
					for (Axis d = 0; d < NR + NC; ++d) {
						Slice slice = spi.slices()[d];
						for (Axis dS = NR + NC; dS < NA + NB; ++dS) {
							if (iisAB[d] == iisAB[dS]) {
								slice.stride += spi.slices()[dS].stride;
							}
						}
						slices_sspi.push_back(slice);
					}
					auto sspi = DynamicProductIndices(slices_sspi);
					std::vector<Index> dims_sspi = sspi.dimensions();


					// ループして構成

					std::array<Index, NR> dimsR;
					for (Axis d = 0; d < NR; ++d) {
						dimsR[d] = dims_sspi[d];
					}
					std::vector<Index> dimsC;
					dimsC.resize(NC);
					for (Axis d = 0; d < NC; ++d) {
						dimsC[d] = dims_sspi[NR + d];
					}


					auto piR = ProductIndices<NR>(dimsR);
					auto piC = DynamicProductIndices(dimsC);

					Eigen::Tensor<ScalarR, NR> tR(dimsR);
					for (Index ir = 0, nir = piR.size(); ir < nir; ++ir) {
						auto indicesR = piR.indices(ir);
						Index startR = 0;
						for (Axis d = 0; d < NR; ++d) {
							startR += sspi.slices()[d].start + sspi.slices()[d].stride * indicesR[d];
						}
						ScalarR val = 0.0;
						for (Index ic = 0, nic = piC.size(); ic < nic; ++ic) {
							auto indicesC = piC.indices(ic);
							Index startC = startR;
							for (Axis dc = 0; dc < NC; ++dc) {
								startC += sspi.slices()[NR + dc].start + sspi.slices()[NR + dc].stride * indicesC[dc];
							}
							ScalarR val_ = kpAB.coeff(startC);
							val += val_;
						}
						tR(indicesR) = val;
					}

					return tR;
				}





			};




			/// <summary>
			/// ToImpl の 1テンソル縮約への特殊化
			/// 
			/// Eigen::Tensor{} 型を対象としている
			/// </summary>
			template<class CRTensorA_, class CRIIndicesA_, class IIndicesR_>
			class ToImpl<std::tuple<CRTensorA_>, std::tuple<CRIIndicesA_>, IIndicesR_> {
			public:
				using TensorA = typename std::remove_const<typename std::remove_reference<CRTensorA_>::type>::type;

				using IIndicesA = typename std::remove_const<typename std::remove_reference<CRIIndicesA_>::type>::type;

				using IIndicesR = IIndicesR_;

				static constexpr Axis NA = std::tuple_size<IIndicesA>::value;
				static constexpr Axis NR = std::tuple_size<IIndicesR>::value;

				using ScalarA = typename TensorA::Scalar;
				using ScalarR = ScalarA;

				using TensorR = Eigen::Tensor<ScalarR, NR>;


				std::tuple<const TensorA&> ts;
				std::tuple<IIndicesA> iiTs;
				IIndicesR iiR;


			protected:
				template<class IIndicesX>
				static std::map<IIndex, Index> makeNumII(const IIndicesX& iiX) {
					std::map<IIndex, Index> numII;
					for (auto& ii : iiX) {
						auto itr = numII.find(ii);
						if (itr == numII.end()) {
							numII[ii] = 1;
						}
						else {
							numII[ii] += 1;
						}
					}
					return numII;
				}



			public:

				bool isValid()const {
					const TensorA& tA = std::get<0>(ts);
					const IIndicesA& iiA = std::get<0>(iiTs);

					// 軸の対応が適当か判定
					bool is_valid = true;
					auto numII = makeNumII(iiA);
					auto numIIR = makeNumII(iiR);
					for (auto& niiR_ : numIIR) {
						IIndex key = niiR_.first;
						auto itr = numII.find(key);

						bool isNotFound = (itr == numII.end());
						if (isNotFound) {
							is_valid = false;
							break;
						}
					}


					// 軸のサイズが適当化判定
					for (Axis dA = 0; dA < NA; ++dA) {
						for (Axis dA_ = 0; dA_ < NA; ++dA_) {
							if (iiA[dA] == iiA[dA_]) {
								if (tA.dimension(dA) != tA.dimension(dA_)) {
									is_valid = false;
								}
							}
						}
					}

					return is_valid;
				}

				/// <summary>
				/// 現在中身を実装していない
				/// </summary>
				/// <returns></returns>
				std::string makeInfo()const {
					std::string str;
					str += "from(";
					str += IIndicesToString(std::get<0>(iiTs));
					str += ").to(";
					str += IIndicesToString(iiR);
					str += ")";
					return str;
				}





				TensorR compute()const {
					const TensorA& tA = std::get<0>(ts);
					const IIndicesA& iiA = std::get<0>(iiTs);

					// 例外
					if (isValid() == false) {
						throw RuntimeException("invalid indices for cmpt::EigenEx::eimsum(...).form(...).to(...)");
						return TensorR();
					}

					// pi を生成
					std::array<Index, NA> dims = tA.dimensions();
					auto pi = ProductIndices<NA>(dims);


					// 軸をシャッフルする
					// for [ 計算後に残すインデックス | 和をとって消えるインデックス | その他のインデックス(対角として統合される) ]
					std::array<Axis, NA> spi_shuffle;

					// 計算後に残すインデックスの軸 [0,NR) を設定
					for (Axis d = 0; d < NA; ++d) {
						spi_shuffle[d] = -1;
					}
					for (Axis dR = 0; dR < NR; ++dR) {
						for (Axis dd = 0; dd < NA; ++dd) {
							if (iiA[dd] == iiR[dR]) {
								spi_shuffle[dR] = dd;
								break;
							}
						}
					}

					// 和をとって消えるインデックスの軸 [NR,NR+NC) を設定
					Axis NC = 0;
					{
						Axis d = NR;
						for (Axis dd = 0; dd < NA; ++dd) {
							bool isC = true;
							for (Axis dR = 0; dR < NR; ++dR) {
								if (iiR[dR] == iiA[dd]) {
									isC = false;
									break;
								}
							}
							for (Axis dC = NR; dC < NR + NC; ++dC) {
								if (iiA[spi_shuffle[dC]] == iiA[dd]) {
									isC = false;
									break;
								}
							}

							if (isC) {
								spi_shuffle[d] = dd;
								++d;
								++NC;
							}
						}
					}

					// 対角として統合されるインデックスの軸 [NR+NC,NA+NB) を設定
					{
						Axis d = NR + NC;
						for (Axis dd = 0; dd < NA; ++dd) {
							bool is_sum = true;
							for (Axis dRC = 0; dRC < NR + NC; ++dRC) {
								if (dd == spi_shuffle[dRC]) {
									is_sum = false;
									break;
								}
							}
							if (is_sum) {
								spi_shuffle[d] = dd;
								++d;
							}
						}
					}




					// 軸をシャッフルした ProductIndices を生成
					auto spi = pi.shuffle(spi_shuffle);
					std::array<IIndex, NA> iis;
					for (Axis d = 0; d < NA; ++d) {
						iis[d] = iiA[spi_shuffle[d]];
					}

					// 軸の一部を対角で評価して消す
					// [ 計算後に残すインデックス | コントラクションに使うインデックス | その他のインデックス ]
					// における [その他のインデックス] を前２つに統合する

					std::vector<Slice> slices_sspi;
					for (Axis d = 0; d < NR + NC; ++d) {
						Slice slice = spi.slices()[d];
						for (Axis dS = NR + NC; dS < NA; ++dS) {
							if (iis[d] == iis[dS]) {
								slice.stride += spi.slices()[dS].stride;
							}
						}
						slices_sspi.push_back(slice);
					}
					auto sspi = DynamicProductIndices(slices_sspi);
					std::vector<Index> dims_sspi = sspi.dimensions();


					// ループして構成

					std::array<Index, NR> dimsR;
					for (Axis d = 0; d < NR; ++d) {
						dimsR[d] = dims_sspi[d];
					}
					std::vector<Index> dimsC;
					dimsC.resize(NC);
					for (Axis d = 0; d < NC; ++d) {
						dimsC[d] = dims_sspi[NR + d];
					}


					auto piR = ProductIndices<NR>(dimsR);
					auto piC = DynamicProductIndices(dimsC);

					Eigen::Tensor<ScalarR, NR> tR(dimsR);
					for (Index ir = 0, nir = piR.size(); ir < nir; ++ir) {
						auto indicesR = piR.indices(ir);
						Index startR = 0;
						for (Axis d = 0; d < NR; ++d) {
							startR += sspi.slices()[d].start + sspi.slices()[d].stride * indicesR[d];
						}
						ScalarR val = 0.0;
						for (Index ic = 0, nic = piC.size(); ic < nic; ++ic) {
							auto indicesC = piC.indices(ic);
							Index startC = startR;
							for (Axis dc = 0; dc < NC; ++dc) {
								startC += sspi.slices()[NR + dc].start + sspi.slices()[NR + dc].stride * indicesC[dc];
							}
							ScalarR val_ = tA(startC);
							val += val_;
						}
						tR(indicesR) = val;
					}
					return tR;
				}





			};



		}




	}


}

