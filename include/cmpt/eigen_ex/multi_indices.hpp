#pragma once

#include <vector>
#include <array>
#include <map>

#include "cmpt/eigen_ex/util.hpp"



namespace cmpt {

	/// <summary>
	/// mitype::Index
	/// mitype::Axis
	/// </summary>
	namespace EigenEx {
		namespace integer_type {

			using Axis = std::size_t;		// size_t 型、標準ライブラリとの連携で template deduction を行うのに必要
			using Index = Eigen::Index;	// 現在 Eigen::Index 型だが今後どうするかわからない、符号なし整数 size_t になる可能性があるのでそれを前提として利用する
			using BlockIndex = Index;		// Index に同じ
		}
	}

	/// <summary>
	/// ProductIndices
	/// DynamicProductIndices
	/// AddIndices
	/// </summary>
	namespace EigenEx {



		/// <summary>
		/// 周期構造の何番目の周期かを取得
		/// 負符号に対応
		/// </summary>
		template<class Index_>
		inline Index_ periodic_div(Index_ index, Index_ n_range) {
			CMPT_EIGENEX_ASSERT(n_range > 0, "invalid arguments");
			if (index >= 0) {
				return index / n_range;
			}
			else {
				return -((-index - 1) / n_range) - 1;
			}
		}


		/// <summary>
		/// 周期構造を利用して0周期に射影したインデックスを取得
		/// 負符号に対応
		/// </summary>
		template<class Index_>
		inline Index_ periodic_mod(Index_ index, Index_ n_range) {
			CMPT_EIGENEX_ASSERT(n_range > 0, "invalid arguments");
			Index_ slide = periodic_div(index, n_range);
			return index - slide * n_range;
		}


		template<integer_type::Axis N>
		inline std::array<integer_type::Axis, N> makeReverseShuffle(const std::array<integer_type::Axis, N>& shuffle) {
			std::array<integer_type::Axis, N> rshuffle;
			for (integer_type::Axis d = 0; d < N; ++d) {
				rshuffle[shuffle[d]] = d;
			}
		}


		



		/// <summary>
		/// スライス std::slice を参考に作成
		/// boost::multi_array_view とは少し違うので注意
		/// 可変長引数コンストラクタに対応
		/// </summary>
		class Slice {
		public:
			using Index = integer_type::Index;
			using Axis = integer_type::Axis;

			Index start;
			Index length;
			Index stride;

			Slice() :start(0), length(0), stride(1) {}

			Slice(
				Index start_,
				Index length_,
				Index stride_
			) :
				start(start_),
				length(length_),
				stride(stride_)
			{}


			bool operator==(const Slice& other)const {
				bool isEqual = false;
				if (start == other.start) {
					if (length == other.length) {
						if (stride == other.stride) {
							isEqual = true;
						}
					}
				}
				return isEqual;
			}

			bool operator!=(const Slice& other)const {
				return !(*this == other);
			}

		};


		/// <summary>
		/// 複数のインデックスと1次元インデックスを相互変換するためのクラス
		/// </summary>
		template<integer_type::Axis NumDims_>
		class ProductIndices {
		public:
			using Index = integer_type::Index;
			using Axis = integer_type::Axis;
			static constexpr Axis NumDimensions = NumDims_;
			using Indices = std::array<Index, NumDimensions>;
			using Slices = std::array<Slice, NumDimensions>;
			using Order = std::array<Axis, NumDimensions>;

		protected:
			Slices slices_;				// slice of each axis
		public:
			const Slices& slices()const { return slices_; }

			static constexpr Axis numDimensions() { return NumDimensions; }

			Index dimension(Axis d)const { return slices_[d].length; }

			Indices dimensions()const {
				auto dims=Indices();
				for (Axis d = 0; d < NumDimensions; ++d) {
					dims[d] = dimension(d);
				}
				return dims;
			}

			/// <summary>
			/// 全 dimensions の積を返す
			/// </summary>
			Index absoluteSize()const {
				Index size_ = 1;
				for (Axis i = 0; i < NumDimensions; ++i) {
					size_ *= dimension(i);
				}
				return size_;
			}

			/// <summary>
			/// 全 dimensions の積を返す
			/// </summary>
			Index size()const { return absoluteSize(); }

			/// <summary>
			/// 現在の slicing がストレージを密にアクセスするか判定する
			/// ストラージオーダーは問わない
			/// </summary>
			bool isDense()const {
				Slices sl = slices();
				std::sort(sl.begin(), sl.end(), [](const Slice& a, const Slice& b)->bool {return a.stride < b.stride; });

				bool is_dense = true;
				Index strd = 1;
				for (Axis d = 0; d < NumDimensions; ++d) {
					if (
						sl[d].start != 0
						||
						sl[d].stride != strd
						) {
						is_dense = false;
						break;
					}
					strd *= sl[d].length;
				}
				return is_dense;
			}


			/// <summary>
			/// 現在の dimensions を密に並べた1次元ストレージの productIndices を生成する
			/// </summary>
			ProductIndices makeDenseProductIndices()const {
				return ProductIndices(dimensions());
			}


			/// <summary>
			/// 1次元ストレージにおけるインデックスを取得
			/// 必ずしも密でないことに注意
			/// </summary>
			Index absoluteIndex(const Indices& indices)const {
				Index absidx = 0;
				for (Index d = 0; d < NumDimensions; ++d) {
					absidx += slices_[d].start + slices_[d].stride * indices[d];
				}
				return absidx;
			}


			/// <summary>
			/// 1次元ストレージにおけるインデックスを取得
			/// 必ずしも密でないことに注意
			/// </summary>
			template<class... Args>
			Index absoluteIndex(Args... args)const {
				return absoluteIndex({ args... });
			}




			/// <summary>
			/// 直積を撮って1次元化した絶対インデックスから各軸のインデックスを取得
			/// 引数は現在の slices のアクセス範囲に含まれる必要がある
			/// </summary>
			template<Axis Ax>
			Index index(Index absidx)const {
				Index cover = absidx / slices()[Ax].stride;
				Index close = dimension(Ax) * (absidx / (dimension(Ax) * slices()[Ax].stride));
				return cover - close;
			}

			/// <summary>
			/// 直積を撮って1次元化した絶対インデックスから各軸のインデックスを取得
			/// 引数は現在の slices のアクセス範囲に含まれる必要がある
			/// </summary>
			Indices indices(Index absidx)const {
				Indices indices;
				for (Axis Ax = 0; Ax < NumDimensions; ++Ax) {
					Index cover = absidx / slices()[Ax].stride;
					Index close = dimension(Ax) * (absidx / (dimension(Ax) * slices()[Ax].stride));
					indices[Ax] = cover - close;
				}
				return indices;
			}



			/// <summary>
			/// 現在の slicing でアクセスする全インデックスを羅列する
			/// </summary>
			std::vector<Index> arrangeAbsoluteIndexList()const {
				std::vector<Index> absidxes;
				absidxes.reserve(size());

				auto dpi = makeDenseProductIndices();
				for (Index di = 0, ndi = dpi.absoluteSize(); di < ndi; ++di) {
					Indices dindices = dpi.indices(di);
					absidxes.push_back(absoluteIndex(dindices));
				}

				return absidxes;
			}


			







			ProductIndices() {}

			template<class... Args>
			ProductIndices(Index head, Args... args) {
				init({ head,Index(args)... });
			}

			ProductIndices(const Indices& shape) {
				init(shape);
			}


			ProductIndices(const Slices& slices_a) {
				init(slices_a);
			}

			void init(const Slices& slices_a) {
				slices_ = slices_a;
			}


			template<class... Args>
			void init(const Indices& shape) {
				Slices slices_a;
				Index stride = 1;
				for (Axis i = 0; i < NumDimensions; ++i) {
					slices_a[i] = Slice(0, shape[i], stride);
					stride *= shape[i];
				}
				init(slices_a);
			}


			template<class... Args>
			void init(Index head, Args... args) {
				Indices indices_{ Index(head),Index(args)... };
				init(indices_);
			}

			void init() {
				Indices shape;
				for (Axis d = 0; d < NumDimensions; ++d) {
					shape[d] = 1;
				}
				init(shape);
			}


			ProductIndices shuffle(const Order& shfl)const {
				ProductIndices pi;
				for (Axis d = 0; d < NumDimensions; ++d) {
					pi.slices_[d] = slices_[shfl[d]];
				}
				return pi;
			}





			bool operator==(const ProductIndices& other)const {
				return slices_ == other.slices_;
			}
			bool operator!=(const ProductIndices& other)const {
				return !(*this == other);
			}







			/// <summary>
			/// 指定した軸の対角に対応するインデックスを末尾に持っていく
			/// 例
			/// pair=[0,2]
			/// (i,j,i,m)->(j,m,(i,i))
			/// </summary>
			ProductIndices<NumDimensions - 1> delta(const std::array<Axis, 2>& pair) {
				std::array<Slice, NumDimensions - 1> slices_a;
				Axis pid = 0;
				for (Axis d = 0; d < NumDimensions;) {
					if (d == pair[0] || d == pair[1]) {
						++d;
						continue;
					}
					slices_a[pid] = slices_[d];
					++pid;
					++d;
				}
				slices_a[pid] = Slice{
					slices()[pair[0]].start,
					slices()[pair[0]].length,
					slices()[pair[0]].stride + slices()[pair[1]].stride
				};
				ProductIndices<NumDimensions - 1> pi(slices_a);
				return pi;
			}


			using IIndex = std::string;


			struct FromToHelper {
			public:

				using IIndices = std::array<IIndex, NumDimensions>;

				const ProductIndices& pi;
				const IIndices ii;

				template<class Str, Axis NR>
				ProductIndices<NR> to(
					const Str(&ar)[NR]
				)const {

					std::array<IIndex, NR> iiR;
					for (Axis d = 0; d < NR; ++d) {
						iiR[d] = ar[d];
					}
					return to(iiR);
				}

				template<Axis NR>
				ProductIndices<NR> to(
					const std::array<IIndex, NR>& iiR
				)const {
					std::map<IIndex, std::vector<Axis>> axisInfo;	// {ii:[d_ii,...],...}
					for (Axis d = 0; d < NumDimensions; ++d) {
						axisInfo[ii[d]].push_back(d);
					}

					std::map<IIndex, Slice> sliceInfo;	// {ii:slice_ii,...}
					for (auto& ax_ : axisInfo) {
						auto& iindex = ax_.first;
						auto& axs = ax_.second;
						Slice slice{ -1,-1,0 };
						for (auto& ax : axs) {
							auto& piSlice = pi.slices()[ax];

							if (slice.start == -1) {
								slice.start = piSlice.start;
							}
							else {
								if (slice.start != piSlice.start) {
									throw RuntimeException("invalid iindex in ProductIndcies<>::from(...)");
								}
							}
							if (slice.length == -1) {
								slice.length = piSlice.length;
							}
							else {
								if (slice.length != piSlice.length) {
									throw RuntimeException("invalid iindex in ProductIndcies<>::from(...)");
								}
							}
							slice.stride += piSlice.stride;
						}
						sliceInfo[iindex] = slice;
					}


					std::array<Slice, NR> slicesR;
					for (Axis dR = 0; dR < NR; ++dR) {
						slicesR[dR] = sliceInfo[iiR[dR]];
					}
					ProductIndices<NR> piR(slicesR);
					return piR;
				}

			};


			/// <summary>
			/// pi.from({"i","j","i"}).to({"i","j"})
			/// のように使ってインデックスを変換する
			/// </summary>
			FromToHelper from(const std::array<IIndex, NumDimensions>& ii) {
				return FromToHelper{ *this,ii };
			}




		};



		/// <summary>
		/// 複数のインデックスと1次元インデックスを相互変換するためのクラス
		/// 動的バージョン
		/// </summary>
		class DynamicProductIndices {
		public:
			using Index = integer_type::Index;
			using Axis = integer_type::Axis;

			using Indices = std::vector<Index>;
			using Slices = std::vector<Slice>;
			using Order = std::vector<Axis>;

		protected:
			Slices slices_;				// slice of each axis
		public:
			const Slices& slices()const { return slices_; }

			Axis numDimensions()const { return slices().size(); }

			Index dimension(Axis d)const { return slices_[d].length; }

			Indices dimensions()const {
				Indices dims;
				dims.resize(slices().size());
				for (Axis d = 0, nd = dims.size(); d < nd; ++d) {
					dims[d] = dimension(d);
				}
				return dims;
			}

			/// <summary>
			/// 全 dimensions の積を返す
			/// </summary>
			Index absoluteSize()const {
				Index size_ = 1;
				for (Index i = 0, ni = numDimensions(); i < ni; ++i) {
					size_ *= dimension(i);
				}
				return size_;
			}

			/// <summary>
			/// 全 dimensions の積を返す
			/// </summary>
			Index size()const { return absoluteSize(); }

			/// <summary>
			/// 現在の slicing がストレージを密にアクセスするか判定する
			/// ストラージオーダーは問わない
			/// </summary>
			bool isDense()const {
				Slices sl = slices();
				std::sort(sl.begin(), sl.end(), [](const Slice& a, const Slice& b)->bool {return a.stride < b.stride; });

				bool is_dense = true;
				Index strd = 1;
				for (Axis d = 0, nd = numDimensions(); d < nd; ++d) {
					if (
						sl[d].start != 0
						||
						sl[d].stride != strd
						) {
						is_dense = false;
						break;
					}
					strd *= sl[d].length;
				}
				return is_dense;
			}


			/// <summary>
			/// 現在の dimensions を密に並べた1次元ストレージの productIndices を生成する
			/// </summary>
			DynamicProductIndices makeDenseProductIndices()const {
				return DynamicProductIndices(dimensions());
			}


			/// <summary>
			/// 1次元ストレージにおけるインデックスを取得
			/// 必ずしも密でないことに注意
			/// </summary>
			Index absoluteIndex(const Indices& indices)const {
				Index absidx = 0;
				for (Index d = 0, nd = numDimensions(); d < nd; ++d) {
					absidx += slices_[d].start + slices_[d].stride * indices[d];
				}
				return absidx;
			}


			/// <summary>
			/// 1次元ストレージにおけるインデックスを取得
			/// 必ずしも密でないことに注意
			/// </summary>
			template<class... Args>
			Index absoluteIndex(Args... args)const {
				return absoluteIndex({ args... });
			}




			/// <summary>
			/// 直積を撮って1次元化した絶対インデックスから各軸のインデックスを取得
			/// 引数は現在の slices のアクセス範囲に含まれる必要がある
			/// </summary>
			template<Axis Ax>
			Index index(Index absidx)const {
				Index cover = absidx / slices()[Ax].stride;
				Index close = dimension(Ax) * (absidx / (dimension(Ax) * slices()[Ax].stride));
				return cover - close;
			}

			/// <summary>
			/// 直積を撮って1次元化した絶対インデックスから各軸のインデックスを取得
			/// 引数は現在の slices のアクセス範囲に含まれる必要がある
			/// </summary>
			Indices indices(Index absidx)const {
				Indices indices;
				indices.resize(numDimensions());
				for (Axis Ax = 0, nAx = numDimensions(); Ax < nAx; ++Ax) {
					Index cover = absidx / slices()[Ax].stride;
					Index close = dimension(Ax) * (absidx / (dimension(Ax) * slices()[Ax].stride));
					indices[Ax] = cover - close;
				}
				return indices;
			}



			/// <summary>
			/// 現在の slicing でアクセスする全インデックスを羅列する
			/// </summary>
			std::vector<Index> arrangeAbsoluteIndexList()const {
				std::vector<Index> absidxes;
				absidxes.reserve(size());

				auto dpi = makeDenseProductIndices();
				for (Index di = 0, ndi = dpi.absoluteSize(); di < ndi; ++di) {
					Indices dindices = dpi.indices(di);
					absidxes.push_back(absoluteIndex(dindices));
				}

				return absidxes;
			}








			DynamicProductIndices() {}

			template<class... Args>
			DynamicProductIndices(Index head, Args... args) {
				init({ head,Index(args)... });
			}

			DynamicProductIndices(const Indices& shape) {
				init(shape);
			}

			DynamicProductIndices(const Slices& slices_a) {
				init(slices_a);
			}

			void init(const Slices& slices_a) {
				slices_ = slices_a;
			}


			template<class... Args>
			void init(const Indices& shape) {
				Slices slices_a;
				slices_a.resize(shape.size());
				Index stride = 1;
				for (Index i = 0, ni = slices_a.size(); i < ni; ++i) {
					slices_a[i] = Slice(0, shape[i], stride);
					stride *= shape[i];
				}
				init(slices_a);
			}


			template<class... Args>
			void init(Index head, Args... args) {
				Indices indices_{ Index(head),Index(args)... };
				init(indices_);
			}

			void init() {
				Indices shape;
				for (Axis d = 0, nd = numDimensions(); d < nd; ++d) {
					shape[d] = 1;
				}
				init(shape);
			}


			DynamicProductIndices shuffle(const Order& shfl)const {
				DynamicProductIndices pi;
				for (Axis d = 0, nd = numDimensions(); d < nd; ++d) {
					pi.slices_[d] = slices_[shfl[d]];
				}
				return pi;
			}




			bool operator==(const DynamicProductIndices& other)const {
				return slices_ == other.slices_;
			}
			bool operator!=(const DynamicProductIndices& other)const {
				return !(*this == other);
			}



			/// <summary>
			/// 指定した軸の対角に対応するインデックスを末尾に持っていく
			/// 例
			/// pair=[0,2]
			/// (i,j,i,m)->(j,m,(i,i))
			/// </summary>
			DynamicProductIndices  delta(const std::array<Axis, 2>& pair) {
				std::vector<Slice> slices_a;
				slices_a.resize(numDimensions() - 1);
				Axis pid = 0;
				for (Axis d = 0, nd = numDimensions(); d < nd;) {
					if (d == pair[0] || d == pair[1]) {
						++d;
						continue;
					}
					slices_a[pid] = slices_[d];
					++pid;
					++d;
				}
				slices_a[pid] = Slice{
					slices()[pair[0]].start,
					slices()[pair[0]].length,
					slices()[pair[0]].stride + slices()[pair[1]].stride
				};
				DynamicProductIndices pi(slices_a);
				return pi;
			}







			using IIndex = std::string;


			struct FromToHelper {
			public:

				using IIndices = std::vector<IIndex>;

				const DynamicProductIndices& pi;
				const IIndices ii;



				DynamicProductIndices to(
					const IIndices& iiR
				)const {
					std::map<IIndex, std::vector<Axis>> axisInfo;	// {ii:[d_ii,...],...}
					for (Axis d = 0, nd = pi.numDimensions(); d < nd; ++d) {
						axisInfo[ii[d]].push_back(d);
					}

					std::map<IIndex, Slice> sliceInfo;	// {ii:slice_ii,...}
					for (auto& ax_ : axisInfo) {
						auto& iindex = ax_.first;
						auto& axs = ax_.second;
						Slice slice{ -1,-1,0 };
						for (auto& ax : axs) {
							auto& piSlice = pi.slices()[ax];

							if (slice.start == -1) {
								slice.start = piSlice.start;
							}
							else {
								if (slice.start != piSlice.start) {
									throw RuntimeException("invalid iindex in ProductIndcies<>::from(...)");
								}
							}
							if (slice.length == -1) {
								slice.length = piSlice.length;
							}
							else {
								if (slice.length != piSlice.length) {
									throw RuntimeException("invalid iindex in ProductIndcies<>::from(...)");
								}
							}
							slice.stride += piSlice.stride;
						}
						sliceInfo[iindex] = slice;
					}


					Slices slicesR;
					slicesR.resize(iiR.size());
					for (Axis dR = 0, ndR = slicesR.size(); dR < ndR; ++dR) {
						slicesR[dR] = sliceInfo[iiR[dR]];
					}
					DynamicProductIndices piR(slicesR);
					return piR;
				}

			};


			/// <summary>
			/// pi.from({"i","j","i"}).to({"i","j"})
			/// のように使ってインデックスを変換する
			/// </summary>
			FromToHelper from(const std::vector<IIndex>& ii) {
				return FromToHelper{ *this,ii };
			}




		};



		/// <summary>
		/// 直和
		/// </summary>
		class AddIndices {
		public:
			using Index = integer_type::Index;
			using Indices = std::vector<Index>;
		protected:

			Indices sizes_;		// [first]
			Indices begins_;	// [first]
			Indices ends_;		// [first]
		public:
			AddIndices(const Indices& sizes_a = Indices{}) :sizes_(sizes_a) {
				Index end = 0;
				for (auto& size : sizes_) {
					begins_.push_back(end);
					end += size;
					ends_.push_back(end);
				}
			}

			void push_back(Index size) {
				Index end_pre = 0;
				if (sizes_.size() != 0) {
					end_pre = ends_.back();
				}
				sizes_.push_back(size);
				begins_.push_back(end_pre);
				ends_.push_back(end_pre + size);
			}

			const Indices& sizes()const { return sizes_; }
			const Indices& begins() const { return begins_; }
			const Indices& ends() const { return ends_; }

			/// <summary>
			/// 1周期に含まれる全要素数を返す
			/// </summary>
			Index absoluteSize()const { return ends_.back(); }

			/// <summary>
			/// first,second から絶対インデックスを計算する
			/// projection_on=false (デフォルト)のとき1周期内に射影しない
			/// </summary>
			Index absoluteIndex(Index first_, Index second_, bool projection_on = false)const {
				Index thre = 0;
				if (!projection_on) {
					thre = periodic_div(first_, static_cast<Index>(sizes().size()));
				}
				Index first = periodic_mod(first_, static_cast<Index>(sizes().size()));
				Index second = periodic_mod(second_, sizes()[first]);
				return thre * absoluteSize() + begins()[first] + second;
			}


			/// <summary>
			/// 絶対インデックスから 属するグループのインデックスを取得する
			/// デフォルトで1周期内に射影しない
			/// </summary>
			Index first(Index absidx_, bool projection_on = false)const {
				Index slide = periodic_div(absidx_, ends().back());
				Index threshold = slide * sizes().size();
				if (projection_on) {
					threshold = 0;
				}
				Index absidx = periodic_mod(absidx_, ends().back());
				auto itr = std::upper_bound(ends().begin(), ends().end(), absidx);
				Index fir = std::distance(ends().begin(), itr);
				return threshold + fir;
			}

			/// <summary>
			/// 絶対インデックスから 属するグループ内におけるインデックスを取得する
			/// </summary>
			Index second(Index absidx_)const {
				Index absidx = periodic_mod(absidx_, static_cast<Index>(ends().back()));
				Index fir = first(absidx);
				Index sec = absidx - begins()[fir];
				return sec;
			}

			bool operator==(const AddIndices& other)const {
				return (sizes() == other.sizes()) && (begins() == other.begins()) && (ends() == other.ends());
			}
			bool operator!=(const AddIndices& other)const {
				return !(*this == other);
			}



		};







	}



	/// <summary>
	/// DTensor を構築する
	/// 開発中
	/// 
	/// クラスヒエラルキー
	/// Base とつくクラスは基底クラスであり、外からは利用しない
	/// 
	/// DTensorConstRefBase					先頭ポインタとインデックスオブジェクトの const 操作
	///	^		^   
	///	|	  |
	/// |	  DTensorMutableRefBase		先頭ポインタとインデックスオブジェクトの mutable 操作
	/// |   ^
	/// |   |
	/// DTensorRefBase							テンソルの参照として使うオブジェクトのCRTP基底クラス 上2つを適当に継承元として呼び分ける
	/// ^
	/// |
	/// DTensorRef									実際にテンソルの参照として使うオブジェクト
	/// ^
	/// |
	/// DTensorBase									ポインタの所有権を自前で持つテンソルオブジェクトのCRTP基底クラス
	/// ^
	/// |
	/// DTensor											ポインタの所有権を自前で持つテンソルオブジェクト
	/// 
	/// 
	/// einsum関連
	/// 以下のように計算するのが目標
	/// C._("i","j")=A._("i","k")*B._("k","j")
	/// 
	/// 中間クラスとして以下を作成
	/// A()
	/// _(...) が返すクラスとして
	/// DTensorConstRefWithIIndex{Scalar}:DTensorRef{const Scalar}
	/// 
	/// DTensorMutableRefWithIIndex{Scalar}:DTensorRef{Scalar}
	/// 
	/// DTensorRefWithIIndex{Scalar}
	/// 
	/// _(...) の積で表すクラスとして
	/// DTensorKroneckerProduct{Scalar}
	/// ^
	/// |
	/// DTensorKroneckerProductWithIIndex{Scalar}
	/// 
	/// operator= を評価するクラスとて
	/// 
	/// DTensorEinsum{Scalar}
	/// 
	/// 評価の仕方
	/// 1. 各テンソルの直積インデックスと、全テンソルの直積インデックスを生成し、
	/// 2. 絶対インデックスを全インデックスに変換するオブジェクトを生成、このとき、コントラクションを取る軸を末尾におく
	/// 3. エレメントごとにループし、和をループさせ足していく
	/// 
	/// 
	/// 
	/// 
	/// 遅延評価 einsum 以外は後回し
	/// 
	/// Scalar operator({...}) のインターフェースを持つ基底クラスを継承して作るCRTPで適当に呼び分ける
	/// conjugate() operator+-*/ で作られる他 einsum 関係で作られる
	/// 
	/// 
	/// 
	/// 
	/// 
	/// 
	/// 
	/// スパース
	/// 
	/// BlockDTensorBase
	/// ^
	/// |
	/// BlockDTensor
	/// 
	/// 
	/// </summary>
	namespace EigenEx {












		/// <summary>
		/// テンソル系の計算の実装部分
		/// 現在は速度より保守性を重視した実装になっている
		/// 必要なときにランクを固定した実装を呼ぶことによる速度の最適化を施すつもり
		/// </summary>
		namespace DTensorImpl {

			using Axis = integer_type::Axis;
			using Index = integer_type::Index;
			using Indices = std::vector<Index>;


			template<class PIA, class PIB>
			inline bool shapeIsMatched(
				const PIA& idxA,
				const PIB& idxB
			) {
				if (idxA.numDimensions() != idxB.numDimensions()) { return false; }
				for (Axis d = 0, nd = idxA.numDimensions(); d < nd; ++d) {
					if (idxA.slices()[d].length != idxB.slices()[d].length) { return false; }
				}
				return true;
			}

			template<class Scalar, class Op>
			inline void opData(
				Scalar const* data_in,
				const DynamicProductIndices& idx_in,
				Scalar* data_out,
				const DynamicProductIndices& idx_out,
				Op op
			) {
				if (!shapeIsMatched(idx_in, idx_out)) {
					throw RuntimeException("shape is not matched");
				}

				Indices indices;
				indices.resize(idx_in.numDimensions());
				for (Index i = 0, ni = idx_in.absoluteSize(); i < ni; ++i) {
					indices = idx_in.indices(i);
					Index abs_in = idx_in.absoluteIndex(indices);
					Index abs_out = idx_out.absoluteIndex(indices);
					op(data_in[abs_in], data_out[abs_out]);
				}

			}


			template<class Scalar>
			inline void copyData(Scalar const* data_in, const DynamicProductIndices& idx_in, Scalar* data_out, const DynamicProductIndices& idx_out) {
				return opData(data_in, idx_in, data_out, idx_out, [](const Scalar& in, Scalar& out) {out = in; });
			}

			template<class Scalar>
			inline void addData(Scalar const* data_in, const DynamicProductIndices& idx_in, Scalar* data_out, const DynamicProductIndices& idx_out) {
				return opData(data_in, idx_in, data_out, idx_out, [](const Scalar& in, Scalar& out) {out += in; });
			}

			template<class Scalar>
			inline void subData(Scalar const* data_in, const DynamicProductIndices& idx_in, Scalar* data_out, const DynamicProductIndices& idx_out) {
				return opData(data_in, idx_in, data_out, idx_out, [](const Scalar& in, Scalar& out) {out -= in; });
			}

			template<class Scalar>
			inline void mulData(Scalar const* data_in, const DynamicProductIndices& idx_in, Scalar* data_out, const DynamicProductIndices& idx_out) {
				return opData(data_in, idx_in, data_out, idx_out, [](const Scalar& in, Scalar& out) {out *= in; });
			}

			template<class Scalar>
			inline void divData(Scalar const* data_in, const DynamicProductIndices& idx_in, Scalar* data_out, const DynamicProductIndices& idx_out) {
				return opData(data_in, idx_in, data_out, idx_out, [](const Scalar& in, Scalar& out) {out /= in; });
			}

			template<class Scalar, class Op>
			inline void opInPlace(Scalar* data, const DynamicProductIndices& idx, Op op) {
				Indices indices;
				indices.resize(idx.numDimensions());
				for (Index i = 0, ni = idx.absoluteSize(); i < ni; ++i) {
					indices = idx.indices(i);
					Index abs_i = idx.absoluteIndex(indices);
					op(data[abs_i]);
				}
			}

			template<class Scalar>
			inline void conjugateInPlace(Scalar* data, const DynamicProductIndices& idx) {
				opInPlace(data, idx, [](Scalar& val) {val = std::conj(val); });
			}
			template<class Scalar>
			inline void setZero(Scalar* data, const DynamicProductIndices& idx) {
				opInPlace(data, idx, [](Scalar& val) {val = Scalar(0.0); });
			}

		}



		/// <summary>
		/// ポインタとテンソルインデックスを所有するオブジェクト
		/// ここではポインタの指す先に対して const な操作のみ実装し、mutable な操作は DTensorMutableRefBase に記述する
		/// </summary>
		template<class Scalar_,class Derived_>
		class DTensorConstRefBase:public CRTPBase<Derived_> {
		public:
			using Scalar= Scalar_;
			using Derived = Derived_;

			using ScalarMutable = typename std::remove_const<Scalar>::type;
			using ScalarConst = const ScalarMutable;
			using Idx = DynamicProductIndices;

			using Axis = integer_type::Axis;
			using Index = integer_type::Index;
			using Axises = std::vector<Axis>;
			using Indices = std::vector<Index>;
			

		protected:
			Scalar* data_;
			Idx idx_;

			void init(Scalar* data_a,const Idx& idx_a) {
				data_ = data_a;
				idx_ = idx_a;
			}

		public:
			
			ScalarConst* data()const { return data_; }
			const Idx& productIndices()const { return idx_; }

			
			DTensorConstRefBase& operator=(const DTensorConstRefBase& other) = delete;
			DTensorConstRefBase& operator=(DTensorConstRefBase&& other) = delete;
			

			ScalarConst& operator[](Index i)const { return data()[i]; }
			ScalarConst& operator()(const Indices& indices)const { 
				return data()[productIndices().absoluteIndex(indices)]; 
			}
			

		};

		/// <summary>
		/// 非 const 型のポインタとテンソルインデックスを所有するオブジェクト
		/// </summary>
		template<class Scalar_,class Derived_>
		class DTensorMutableRefBase: public DTensorConstRefBase <Scalar_,Derived_>{
		public:
			using Scalar = Scalar_;
			using Derived = Derived_;

			using ScalarMutable = typename std::remove_const<Scalar>::type;
			using ScalarConst = const ScalarMutable;
			using Idx = DynamicProductIndices;

			using Axis = integer_type::Axis;
			using Index = integer_type::Index;
			using Axises = std::vector<Axis>;
			using Indices = std::vector<Index>;

		protected:
			

		public:

			template<class DerivedA>
			Derived& operator=(const DTensorConstRefBase<ScalarMutable,DerivedA>& other) {
				DTensorImpl::copyData(other.data(), other.productIndices(), this->data(), this->productIndices());
				return this->derived();
			}
			template<class DerivedA>
			Derived& operator=(const DTensorConstRefBase<ScalarConst, DerivedA>& other) {
				DTensorImpl::copyData(other.data(), other.productIndices(), this->data(), this->productIndices());
				return this->derived();
			}

			template<class DerivedA>
			Derived& operator+=(const DTensorConstRefBase<Scalar, DerivedA>& other) {
				DTensorImpl::addData(other.data(), other.productIndices(), this->data(), this->productIndices());
				return this->derived();
			}
			template<class DerivedA>
			Derived& operator-=(const DTensorConstRefBase<Scalar, DerivedA>& other) {
				DTensorImpl::subData(other.data(), other.productIndices(), this->data(), this->productIndices());
				return this->derived();
			}
			template<class DerivedA>
			Derived& operator*=(const DTensorConstRefBase<Scalar, DerivedA>& other) {
				DTensorImpl::mulData(other.data(), other.productIndices(), this->data(), this->productIndices());
				return this->derived();
			}
			template<class DerivedA>
			Derived& operator/=(const DTensorConstRefBase<Scalar, DerivedA>& other) {
				DTensorImpl::divData(other.data(), other.productIndices(), this->data(), this->productIndices());
				return this->derived();
			}

			Derived& conjugateInPlace() {
				DTensorImpl::conjugateInPlace(this->data(), this->productIndices());
			}

			Derived& setZero() {
				DTensorImpl::setZero(this->data(), this->productIndices());
			}

			


			ScalarMutable* data(){ return this->data_; }
			
			Scalar& operator[](Index i) { return this->data()[i]; }
			Scalar& operator()(const Indices& indices) { return this->data()[this->productIndices().absoluteIndex(indices)]; }


		};

		/// <summary>
		/// const の有無で DTensorConstRefBase と DTensorMutableRefBase を適当に呼び分ける推論クラス
		/// </summary>
		template<class Scalar_,class Derived_>
		struct DTensorRefTraits {
			using Scalar = Scalar_;
			using Derived = Derived_;
			
			template<class IsConst,class AlwaysBool>
			struct Detail {};

			template<class AlwaysBool>
			struct Detail<std::true_type, AlwaysBool> {
				using Type = DTensorConstRefBase<Scalar, Derived>;
			};
			template<class AlwaysBool>
			struct Detail<std::false_type, AlwaysBool> {
				using Type = DTensorMutableRefBase<Scalar, Derived>;
			};


			using Type = typename Detail<typename std::is_const<Scalar>::type, bool>::Type;



		};


		/// <summary>
		/// CRTPの派生中継点
		/// </summary>
		template<class Scalar_, class Derived_>
		class DTensorRefBase :public DTensorRefTraits<Scalar_,Derived_>::Type{
		public:
			
		};

		/// <summary>
		/// DTensorRef コンストラクタ関連以外の実装は継承元で行っている
		/// </summary>
		template<class Scalar_>
		class DTensorRef :public DTensorRefBase<Scalar_, DTensorRef<Scalar_>> {
		public:
			using Scalar = Scalar_;

			using ScalarMutable = typename std::remove_const<Scalar>::type;
			using ScalarConst = const ScalarMutable;
			using Idx = DynamicProductIndices;

			using Axis = integer_type::Axis;
			using Index = integer_type::Index;
			using Axises = std::vector<Axis>;
			using Indices = std::vector<Index>;

			DTensorRef() { this->init(nullptr, Idx()); }
			DTensorRef(Scalar* data_a, const Idx& idx_a) { this->init(data_a, idx_a); }
			DTensorRef(const DTensorRef& other) = delete;
			DTensorRef(DTensorRef&& other) = delete;
			
			
			DTensorRef& operator=(const DTensorRef& other) {
				return DTensorMutableRefBase<Scalar, DTensorRef>::operator=(other);
			}
			
			template<class DerivedA> DTensorRef& operator=(const DTensorConstRefBase<ScalarConst, DerivedA>& other) {
				return DTensorMutableRefBase<Scalar, DTensorRef>::operator=(other);
			}
			template<class DerivedA> DTensorRef& operator=(const DTensorConstRefBase<ScalarMutable, DerivedA>& other) {
				return DTensorMutableRefBase<Scalar, DTensorRef>::operator=(other);
			}
		};


		/// <summary>
		/// DTensor の基底クラス
		/// 独自のメモリ領域を確保する
		/// 本クラスはインデックスを常に colmn-major で割り当てる
		/// </summary>
		template<class Scalar_,class Derived_>
		class DTensorBase:public DTensorRefBase<Scalar_,Derived_> {
		public:
			using Scalar = Scalar_;
			using Derived = Derived_;

			using ScalarMutable = typename std::remove_const<Scalar>::type;
			using ScalarConst = const ScalarMutable;
			using Idx = DynamicProductIndices;

			using Axis = integer_type::Axis;
			using Index = integer_type::Index;
			using Axises = std::vector<Axis>;
			using Indices = std::vector<Index>;

		protected:
			std::vector<Scalar> buffer_;
			
			void init() { 
				DTensorRefBase<Scalar, Derived>::init(nullptr, Idx()); 
				buffer_.resize(0); 
			}

			
			void init(const Indices& shape) {
				Idx idx(shape);

				Index bufferSize = 0;
				for (Axis d = 0, nd = idx.numDimensions(); d < nd; ++d) {
					auto slice = idx.slices()[d];
					Index end = slice.start + slice.length * slice.stride;
					if (bufferSize < end) {
						bufferSize = end;
					}
				}
				buffer_.resize(bufferSize);
				DTensorRefBase<Scalar, Derived>::init(buffer_.data(), idx);
				
			}

			//void init(const Idx& idx) {init(idx.dimensions());}



			template<class DerivedA>
			void init(const DTensorRefBase<ScalarMutable, DerivedA>& other) {
				Idx pi(other.productIndices().dimensions());
				this->init(pi);
				DTensorMutableRefBase<Scalar, Derived>::operator=(other);	// Mutable 必要 なぜかわからない
			}

			template<class DerivedA>
			void init(const DTensorRefBase<ScalarConst, DerivedA>& other) {
				this->init(other.productIndices().dimensions());
				
				DTensorMutableRefBase<Scalar, Derived>::operator=(other); // Mutable 必要 なぜかわからない
			}
			
		public:

			const std::vector<Scalar>& buffer()const { return buffer_; }

			
		};


		/// <summary>
		/// DTensor コンストラクタ関連以外の実装は継承元で行っている
		/// </summary>
		template<class Scalar_>
		class DTensor:public DTensorBase<Scalar_,DTensor<Scalar_>>{
		public:
			using Scalar = Scalar_;

			using ScalarMutable = typename std::remove_const<Scalar>::type;
			using ScalarConst = const ScalarMutable;
			using Idx = DynamicProductIndices;

			using Axis = integer_type::Axis;
			using Index = integer_type::Index;
			using Axises = std::vector<Axis>;
			using Indices = std::vector<Index>;


		public:

			DTensor() { 
				DTensorBase<Scalar, DTensor<Scalar>>::init(); 
			}

			DTensor(const Indices& shape) { 
				DTensorBase<Scalar, DTensor<Scalar>>::init(shape); 
			}

			template<class DerivedA>
			DTensor(const DTensorRefBase<ScalarMutable, DerivedA>& other) { 
				DTensorBase<Scalar, DTensor<Scalar>>::init(other);
			}
			template<class DerivedA>
			DTensor(const DTensorRefBase<ScalarConst, DerivedA>& other) {
				DTensorBase<Scalar, DTensor<Scalar>>::init(other);
			}



		};


		using IIndex = std::string;

		template<class Scalar_>
		class DTensorRefWithIIndex{
		public:
			using Scalar = Scalar_;
			
			DTensorRef<Scalar> ref_;
			std::vector<IIndex> iindices_;
			

		};


		template<class Scalar_>
		class DTensorKroneckerProductRef {
		public:
			using Scalar = Scalar_;
			using Idx = DynamicProductIndices;

		protected:
			std::vector<DTensorRef<Scalar>> dts_;	// [i] -> ([absi] <-> [i0,...,iN-1])
			Idx idx_;	// [abs] <-> [abs0,...,absM-1]
		public:

		};

		template<class Scalar_>
		class DTensorKroneckerProductRefWithIIndex {
		public:
			using Scalar = Scalar_;

			DTensorKroneckerProductRef<Scalar> kpRef_;
			std::vector<IIndex> iIndices_;


		};




	}


	



}
