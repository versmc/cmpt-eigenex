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
				Indices dims;
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
			AddIndices(const Indices& sizes_a = Indices{ 1 }) :sizes_(sizes_a) {
				Index end = 0;
				for (auto& size : sizes_) {
					begins_.push_back(end);
					end += size;
					ends_.push_back(end);
				}
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



}
