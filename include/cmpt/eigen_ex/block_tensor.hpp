#pragma once

#include <vector>
#include <array>
#include <algorithm>


#include "cmpt/eigen_ex/multi_indices.hpp"
#include "cmpt/eigen_ex/einsum.hpp"


namespace cmpt{
	/// <summary>
	/// BlockTensor とそれ関連の実装
	/// </summary>
	namespace EigenEx {

		
		/// <summary>
		/// 開発に失敗したバージョンの BlockTensor
		/// 削除予定
		/// </summary>
		namespace old {

			/// <summary>
			/// BlockTensor の基底クラス
			/// Derived& を返す関数の判定のために CRTP を利用している
			/// 外部から利用するときは、このクラスは一時オブジェクトとして使い、代わりに継承先の BlockTensor に変換して使う
			/// 
			/// 本クラスを直接実体化した場合 自身の参照を返すタイプのメンバ関数で矛盾が生じる
			/// 
			/// TODO
			/// 
			/// 把握しているバグとして 
			/// NumDimensions_==0 のときに一部のメンバ関数を呼ぶとコンパイルが終わらなくなる
			/// 該当するメンバ関数として以下を把握している
			/// getElement(Indices)
			/// makeDenseTensor()
			/// </summary>
			template<class Scalar_, integer_type::Axis NumDimensions_>
			class BlockTensor {

				// タイプエイリアス
			public:



				using Index = integer_type::Index;
				using Axis = integer_type::Axis;
				using Scalar = Scalar_;
				using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
				static constexpr Axis NumDimensions = NumDimensions_;

				using Indices = std::array<Index, NumDimensions>;
				using BlockIndex = Index;
				using BlockIndices = std::array<BlockIndex, NumDimensions>;
				using DenseTensorType = Eigen::Tensor<Scalar, NumDimensions>;
				using DenseTensorsType = std::map<std::array<BlockIndex, NumDimensions>, DenseTensorType>;

				using ProductIndicesType = ProductIndices<NumDimensions>;
				using AddIndicesType = AddIndices;
				using BlockAddIndicesType = std::array<AddIndicesType, NumDimensions>;



				// メンバ変数
			protected:

				ProductIndicesType productIndices_;			// [i] <-> [i_0,...,i_{N-1}]
				BlockAddIndicesType blockAddIndices_;		// [i_d] <->[q_d,i_{q_d}]
				DenseTensorsType blocks_;			// [{q_0,...,q_{N-1}}](i_{q_0},...,i_{q_{N-1}})



				// アクセッサ
			public:

				const ProductIndicesType& productIndices()const { return productIndices_; }
				const BlockAddIndicesType& blockAddIndices()const { return blockAddIndices_; }
				const DenseTensorsType& blocks()const { return blocks_; }



				/// <summary>
				/// get the dimension of the given axis
				/// </summary>
				Index dimension(Axis axis)const {
					return productIndices().dimension(axis);
				}

				/// <summary>
				/// get the all dimensions
				/// </summary>
				Indices dimensions()const {
					return productIndices().dimensions();
				}

				/// <summary>
				/// get the number of division of the given axis
				/// </summary>
				BlockIndex blockDimension(Axis axis)const {
					return blockAddIndices()[axis].sizes().size();
				}

				/// <summary>
				/// get the number of division of each axis
				/// </summary>
				BlockIndices blockDimensions()const {
					BlockIndices bIndices;
					for (Axis d = 0; d < NumDimensions; ++d) {
						bIndices[d] = blockAddIndices()[d].sizes().size();
					}
					return bIndices;
				}


				/// <summary>
				/// get the intra-block dimension at the given block
				/// </summary>
				Index intraBlockDimension(Axis axis, BlockIndex si)const {
					return blockAddIndices()[axis].sizes()[si];
				}

				/// <summary>
				/// get the intra-block dimension at the given block
				/// </summary>
				Indices intraBlockDimensions(const BlockIndices& sindices)const {
					auto bdims = Indices();
					for (Axis d = 0; d < NumDimensions; ++d) {
						bdims[d] = blockAddIndices()[d].sizes()[sindices[d]];
					}
					return bdims;
				}

				/// <summary>
				/// ブロックインデックスとブロック内インデックスによる要素取得
				/// map の検索コストがかかるため log の探索時間がかかる
				/// </summary>
				Scalar getElement(
					const BlockIndices& sIndices,
					const Indices& indicesInQn
				)const {
					auto itr = blocks().find(sIndices);
					if (itr == blocks().end()) {
						return Scalar(0.0);
					}
					else {
						return itr->second(indicesInQn);
					}
				}

				/// <summary>
				/// 各軸の絶対インデックスによる要素取得
				/// map の検索コストがかかるため log の探索時間がかかる
				/// </summary>
				Scalar getElement(
					const Indices& indices
				)const {
					BlockIndices sIndices;
					Indices indicesIntra;
					for (Axis d = 0; d < NumDimensions; ++d) {
						sIndices[d] = blockAddIndices()[d].first(indices[d]);
						indicesIntra[d] = blockAddIndices()[d].second(indices[d]);
					}
					return getElement(sIndices, indicesIntra);
				}

				const DenseTensorType& getBlock(const BlockIndices& sIndices)const {
					return blocks().at(sIndices);
				}



				/// <summary>
				/// 各軸の絶対インデックスにより要素を取得
				/// map の検索コストにより log の探索時間がかかる
				/// </summary>
				template<class... Args>
				Scalar operator()(const Args&... args)const {
					return (*this)(make_indices(args...));
				}


				/// <summary>
				/// 各軸の絶対インデックスにより要素を取得
				/// map の検索コストにより log の探索時間がかかる
				/// </summary>
				Scalar operator()(Indices& indices)const {
					BlockIndices sIndices;
					Indices indicesIntra;
					for (Axis d = 0; d < NumDimensions; ++d) {
						sIndices[d] = blockAddIndices()[d].first(indices[d]);
						indicesIntra[d] = blockAddIndices()[d].second(indices[d]);
					}
					return getElement(sIndices, indicesIntra);
				}


				/// <summary>
				/// 対応する Eigen::Tensor 型のオブジェクトを取得
				/// </summary>
				DenseTensorType makeDenseTensor()const {
					DenseTensorType dt = DenseTensorType(dimensions()).setZero();

					for (auto& pr : blocks()) {
						auto& bindices = pr.first;
						auto& block = pr.second;
						std::array<Index, NumDimensions> offsets;
						for (Axis d = 0; d < NumDimensions; ++d) {
							offsets[d] = blockAddIndices()[d].begins()[bindices[d]];
						}
						std::array<Index, NumDimensions> extents;
						for (Axis d = 0; d < NumDimensions; ++d) {
							extents[d] = blockAddIndices()[d].sizes()[bindices[d]];
						}
						dt.slice(offsets, extents) = block;
					}

					return dt;
				}


				/// <summary>
				/// 対応するテンソルを表現する 有限要素の vector を取得
				/// インデックスは各軸ごとの絶対インデックスを表す
				/// </summary>
				std::vector<std::tuple<Indices, Scalar>> makeFiniteElementsVector()const {
					std::vector<std::tuple<Indices, Scalar>> fev;
					for (auto& block_ : blocks()) {
						auto& bIndices = block_.first;
						auto& block = block_.second;
						ProductIndicesType pib(block.dimensions());
						for (Index bi = 0, nbi = pib.absoluteSize(); bi < nbi; ++bi) {
							Indices iIndices = pib.indices(bi);
							Indices indices;
							Scalar value = block(iIndices);
							if (value == Scalar(0.0)) {
								continue;
							}
							for (Axis d = 0; d < NumDimensions; ++d) {
								indices[d] = blockAddIndices()[d].absoluteIndex(bIndices[d], iIndices[d]);
							}
							fev.push_back({ indices,block(iIndices) });
						}
					}
				}



				/// <summary>
				/// ブロックの形状が等しいか判定する
				/// </summary>
				template<class S_, Axis N_>
				bool equalsBlocks(const BlockTensor<S_, N_>& other)const {
					if (dimensions() != other.dimensions()) {
						return false;
					}
					for (Axis d = 0; d < NumDimensions; ++d) {
						if (blockAddIndices()[d].sizes() != other.blockAddIndices()[d].sizes()) {
							return false;
						}
					}
					return true;
				}


				/// <summary>
				/// キャスト
				/// </summary>
				template<class S_>
				BlockTensor<S_, NumDimensions> cast()const {
					BlockTensor<S_, NumDimensions> bt(blockAddIndices());
					for (auto& pr : blocks()) {
						auto& bIndices = pr.first;
						auto& block = pr.second;
						bt.addBlock(bIndices, block.template cast<S_>());
					}
					return bt;
				}



				// init 系
			protected:


				BlockTensor& init(
					const BlockAddIndicesType& bi_,
					const DenseTensorsType& dt_
				) {

					Indices dims;
					for (Axis d = 0; d < NumDimensions; ++d) {
						dims[d] = bi_[d].absoluteSize();
					}

					// 構築
					productIndices_ = ProductIndicesType(dims);
					blockAddIndices_ = bi_;
					blocks_ = dt_;

					return *this;
				}

				/// <summary>
				/// 要素は全てゼロで構築
				/// </summary>
				BlockTensor& init(
					const BlockAddIndicesType& bi_
				) {
					init(bi_, DenseTensorsType());
					return *this;
				}


				/// <summary>
				/// 要素は全てゼロ
				///	ブロックは大きい1つ
				/// により構築
				/// </summary>
				BlockTensor& init(
					const ProductIndicesType& pi_
				) {
					BlockAddIndicesType bi_;
					for (Axis d = 0; d < NumDimensions; ++d) {
						bi_[d] = AddIndices({ pi_.dimension(d) });
					}
					init(bi_);
					return *this;
				}

				/// <summary>
				/// 要素数0 で構築
				/// </summary>
				BlockTensor& init() {
					std::array<Index, NumDimensions> shape;
					for (Axis d = 0; d < NumDimensions; ++d) {
						shape[d] = 0;
					}
					ProductIndicesType pi_(shape);
					init(pi_);
					return *this;
				}

				// コンストラクタ群
			public:
				BlockTensor() { init(); }
				BlockTensor(const BlockAddIndicesType& bi_) { init(bi_); }
				BlockTensor(const ProductIndicesType& pi_) { init(pi_); }
				BlockTensor(
					const BlockAddIndicesType& bi_,
					const DenseTensorsType& dt_
				) {
					init(bi_, dt_);
				}
				BlockTensor(
					const std::vector<std::vector<Index>>& vec
				) {
					BlockAddIndicesType bai;
					for (Axis d = 0; d < NumDimensions; ++d) {
						bai[d] = AddIndicesType(vec[d]);
					}
					init(bai);
				}


				BlockTensor(const BlockTensor& other) = default;
				BlockTensor(BlockTensor&& other) = default;
				BlockTensor& operator=(const BlockTensor& other) = default;
				BlockTensor& operator=(BlockTensor&& other) = default;


			public:
				// ミューテータ

				/// <summary>
				/// ブロックを加算する
				/// </summary>
				BlockTensor& addBlock(
					const BlockIndices& sindices,
					const DenseTensorType& t
				) {
					if (t.dimensions() != intraBlockDimensions(sindices)) {
						throw RuntimeException("");
					}
					else {
						auto itr = blocks_.find(sindices);
						if (itr == blocks_.end()) {
							blocks_[sindices] = t;
						}
						else {
							itr->second += t;
						}
					}
					return *this;
				}




				BlockTensor& mulBlock(
					const BlockIndices& sindices,
					const DenseTensorType& t
				) {
					if (t.dimensions() != intraBlockDimensions(sindices)) {
						throw RuntimeException("");
					}
					else {
						auto itr = blocks_.find(sindices);
						if (itr == blocks_.end()) {
						}
						else {
							itr->second *= t;
						}
					}
					return *this;
				}


				/// <summary>
				/// ゼロクリア
				/// 形状は変更しない
				/// </summary>
				void clear() {
					blocks_.clear();
				}





				/// <summary>
				/// ブロックインデックスとブロック内インデックスによる要素の設定
				/// 要素が所属するブロックが存在しない場合は作る
				/// map の検索コストがかかるため log の探索時間がかかる
				/// </summary>
				BlockTensor& setElement(
					const BlockIndices& sIndices,
					const Indices& indicesIntra,
					const Scalar& value
				) {
					auto itr = blocks().find(sIndices);
					if (itr == blocks().end()) {
						if (value == Scalar(0.0)) {
						}
						else {
							DenseTensorType dt = DenseTensorType(intraBlockDimensions(sIndices)).setZero();
							dt(indicesIntra) = value;
							blocks_[sIndices] = dt;
						}
					}
					else {
						blocks_[sIndices](indicesIntra) = value;
					}

					return *this;
				}

				/// <summary>
				/// 各軸の絶対インデックスによる要素取得よる要素の設定
				/// 要素が所属するブロックが存在しない場合は作る
				/// map の検索コストがかかるため log の探索時間がかかる
				/// </summary>
				BlockTensor& setElement(
					const Indices& indices,
					const Scalar& value
				) {
					BlockIndices sIndices;
					Indices intraIndices;
					for (Axis d = 0; d < NumDimensions; ++d) {
						sIndices[d] = blockAddIndices()[d].first(indices[d]);
						intraIndices[d] = blockAddIndices()[d].second(indices[d]);
					}
					return setElement(sIndices, intraIndices, value);
				}




				BlockTensor& setBlock(
					const BlockIndices& sIndices,
					DenseTensorType& dt
				) {
					if (dt.dimensions() != intraBlockDimensions(sIndices)) {
						RuntimeException("In BTensor::setBlock(...), the shape of block is invalid");
					}
					else {
						blocks_[sIndices] = dt;
					}
					return *this;
				}

				BlockTensor& eraseBlock(const BlockIndices& sIndices) {
					blocks_.erase(sIndices);
					return *this;;
				}

				/// <summary>
				/// 密行列から構築
				/// 各ブロックの形状は維持する
				/// </summary>
				BlockTensor& setFromDenseTensor(const DenseTensorType& dt) {
					if (dimensions() != dt.dimensions()) {
						throw RuntimeException("in BTensor::setFromDenseTensor(...), dense tensor dimensions are invalid");
					}

					BlockIndices bdims = blockDimensions();
					auto bidx = ProductIndicesType(bdims);
					for (Index b = 0, nb = bidx.absoluteSize(); b < nb; ++b) {
						BlockIndices bIndices = bidx.indices(b);
						DenseTensorType block;
						Indices offsets;
						Indices extents;
						Indices iota;
						for (Axis d = 0; d < NumDimensions; ++d) {
							offsets[d] = blockAddIndices()[d].begins()[bIndices[d]];
							extents[d] = blockAddIndices()[d].sizes()[bIndices[d]];
							iota[d] = d;
						}
						block = dt.slice(offsets, extents);

						Eigen::Tensor<Scalar, 0> max_ = block.abs().maximum(iota);
						if (max_() == RealScalar(0.0)) {
							continue;
						}
						else {
							setBlock(bIndices, block);
						}
					}

					return *this;
				}


				BlockTensor& shuffleInPlace(const std::array<Axis, NumDimensions>& sfl) {

					BlockAddIndicesType baIndices;
					for (Axis d = 0; d < NumDimensions; ++d) {
						baIndices[d] = blockAddIndices()[sfl[d]];
					}
					DenseTensorsType dts;
					for (auto& block_ : blocks()) {
						auto& bIndices = block_.first;
						auto& block = block_.second;
						BlockIndices bIndicesNew;
						for (Axis d = 0; d < NumDimensions; ++d) {
							bIndicesNew[d] = bIndices[sfl[d]];
						}
						dts[bIndicesNew] = DenseTensorType(block.shuffle(sfl));
					}


					init(baIndices, dts);
					return *this;
				}


				BlockTensor& blockShuffleInPlace(Axis axis, const std::vector<BlockIndex>& bShuffle) {
					std::vector<BlockIndex> reverseShuffle;
					reverseShuffle.resize(bShuffle.size());
					for (BlockIndex bi = 0, nbi = blockAddIndices()[axis].sizes().size(); bi < nbi; ++bi) {
						reverseShuffle[bShuffle[bi]] = bi;
					}

					std::vector<BlockIndex> sizes;
					sizes.resize(blockAddIndices()[axis].sizes().size());

					for (BlockIndex bi = 0, nbi = sizes.size(); bi < nbi; ++bi) {
						sizes[bi] = blockAddIndices()[axis].sizes()[bShuffle[bi]];
					}
					blockAddIndices_[axis] = AddIndices(sizes);

					DenseTensorsType dt;
					for (auto& block_ : blocks_) {
						auto key = block_.first;
						key[axis] = reverseShuffle[key[axis]];
						std::swap(dt[key], block_.second);
					}
					blocks_ = dt;
					return *this;
				}


				BlockTensor& blockShuffleInPlace(const std::array<std::vector<BlockIndex>, NumDimensions>& bshuffles) {
					for (Axis d = 0; d < NumDimensions; ++d) {
						blockShuffleInPlace(d, bshuffles[d]);
					}
					return *this;
				}



				BlockTensor& setFromFiniteElementsVector(const std::vector<std::tuple<Indices, Scalar>>& v) {
					clear();
					for (auto& e_ : v) {
						Indices indices = std::get<0>(e_);
						Scalar value = std::get<1>(e_);
						setElement(indices, value);
					}
					return *this;
				}

				/// <summary>
				/// ブロックのうち絶対値最大の要素のが threshold 以下のブロックを削除する(ゼロとする)
				/// 全体の形状は変化しない
				/// </summary>
				void truncate(const RealScalar& threshold) {
					std::vector<BlockIndices> erase_keys;
					for (auto& block_ : blocks_) {
						auto& block = block_.second;
						Eigen::Tensor<RealScalar, 0> max_ = block.abs().maximum();
						if (max_() < threshold) {
							erase_keys.push_back(block_.first);
						}
					}
					for (auto& key : erase_keys) {
						blocks_.erase(key);
					}
				}


				BlockTensor& reblock(const BlockAddIndicesType& bi) {
					auto v = this->makeFiniteElementsVector();
					this->init(bi);
					this->setFromFiniteElementsVector(v);
					return *this;
				}


				BlockTensor& conjugateInPlace() {
					for (auto& bt_ : this->blocks_) {
						bt_.second = bt_.second.conjugate();
					}
					return *this;
				}

				BlockTensor& scalarMultiple(const Scalar& c) {
					for (auto& block_ : this->blocks_) {
						auto& block = block_.second;
						block = -block;
					}
					return *this;
				}

				template<class S_, Axis N_, class D_>
				BlockTensor& operator+=(const BlockTensor<Scalar, NumDimensions>& other) {
					if (!*this.equalsBlocks(other)) {
						RuntimeException("");
					}
					for (auto& block_ : other.blocks()) {
						auto& bIndices = block_.first;
						auto& block = block_.second;
						*this.addBlock(bIndices, block);
					}
					return *this;
				}

				template<class S_, Axis N_, class D_>
				BlockTensor& operator-=(const BlockTensor<Scalar, NumDimensions>& other) {
					return *this += (-other);
				}

				template<class S_, Axis N_, class D_>
				BlockTensor& operator*=(const BlockTensor<Scalar, NumDimensions>& other) {
					if (!*this.equalsBlocks(other)) {
						RuntimeException("");
					}
					for (auto& block_ : other.blocks()) {
						auto& bIndices = block_.first;
						auto& block = block_.second;
						*this.mulBlock(bIndices, block);
					}
					return *this;
				}

				template<class S_, Axis N_, class D_>
				BlockTensor& operator/=(const BlockTensor<Scalar, NumDimensions>& other) {
					if (!*this.equalsBlocks(other)) {
						RuntimeException("");
					}
					for (auto& block_ : other.blocks()) {
						auto& bIndices = block_.first;
						auto& block = block_.second;
						DenseTensorType block_inv = DenseTensorType(block.dimensions()).setConstant(Scalar(1.0)) / block;
						*this.mulBlock(bIndices, block_inv);
					}
					return *this;
				}



			public:
				// 自身と同じ型を返す系


				BlockTensor shuffle(const std::array<Axis, NumDimensions>& sfl)const {
					BlockTensor drvd(*this);
					drvd.shuffleInPlace(sfl);
					return drvd;
				}

				BlockTensor blockShuffle(Axis axis, const std::vector<BlockIndex>& bShuffle)const {
					BlockTensor drvd(*this);
					drvd.blockShuffleInPlace(axis, bShuffle);
					return drvd;
				}

				BlockTensor blockShuffle(const std::array<std::vector<BlockIndex>, NumDimensions>& bShuffle)const {
					BlockTensor drvd(*this);
					drvd.blockShuffleInPlace(bShuffle);
					return drvd;
				}

				/// <summary>
				/// ブロックのうち絶対値最大の要素のが threshold 以下のブロックを削除したものを返す
				/// </summary>
				BlockTensor truncated(const RealScalar& threshold)const {
					BlockTensor drvd(*this);
					drvd.truncate(threshold);
					return drvd;
				}


				BlockTensor conjugate()const {
					BlockTensor ret(*this);
					ret.conjugateInPlace();
					return ret;
				}

				BlockTensor scalarMultipled(const Scalar& c)const {
					BlockTensor drvd(*this);
					drvd.scalarMultiple(c);
					return drvd;
				}


				BlockTensor operator+()const {
					return *this;
				}

				BlockTensor operator-()const {
					BlockTensor ret(*this);
					ret.scalarMultiple(Scalar(-1.0));
					return ret;
				}





				template<Axis NB_, class Pairs_, class FContractDenseTensor>
				BlockTensor<Scalar, NumDimensions + NB_ - 2 * std::tuple_size<Pairs_>::value> contract(
					const BlockTensor<Scalar, NB_>& btB,
					const Pairs_& pairs,
					const FContractDenseTensor& contractDenseTensor
				)const {
					using BTR = BlockTensor<Scalar, NumDimensions + NB_ - 2 * std::tuple_size<Pairs_>::value>;

					constexpr Axis NA = NumDimensions;
					constexpr Axis NB = NB_;
					constexpr Axis NR = BTR::NumDimensions;


					auto& btA = *this;

					// 引数エラーチェック
					{
						bool is_valid = true;
						for (auto& pair : pairs) {
							Axis dA = pair.first;
							Axis dB = pair.second;
							if (btA.blockAddIndices()[dA].sizes() != btB.blockAddIndices()[dB].sizes()) {
								is_valid = false;
								break;
							}
						}
						if (!is_valid) {
							throw RuntimeException("in BlockTensorBase{...}::contract(...), invalid contractions");
						}
					}




					// A, B の各軸について contraction をとる軸かどうかを判定して保持
					std::array<bool, NA> isPairA;
					for (Axis dA = 0; dA < NA; ++dA) {
						bool is_pair = false;
						for (auto& pair : pairs) {
							if (pair.first == dA) {
								is_pair = true;
								break;
							}
						}
						isPairA[dA] = is_pair;
					}

					std::array<bool, NB> isPairB;
					for (Axis dB = 0; dB < NB; ++dB) {
						bool is_pair = false;
						for (auto& pair : pairs) {
							if (pair.second == dB) {
								is_pair = true;
								break;
							}
						}
						isPairB[dB] = is_pair;
					}

					// R の各軸についてブロック構造を設定
					std::array<AddIndices, NR> baiR;
					{
						Axis dR = 0;
						for (Axis dA = 0; dA < NA; ++dA) {
							if (!isPairA[dA]) {
								baiR[dR] = btA.blockAddIndices()[dA];
								++dR;
							}
						}
						for (Axis dB = 0; dB < NB; ++dB) {
							if (!isPairB[dB]) {
								baiR[dR] = btB.blockAddIndices()[dB];
								++dR;
							}
						}
					}

					// R の各軸についてサイズを取得
					std::array<int, NR> dimsR;
					for (Axis dR = 0; dR < NR; ++dR) {
						dimsR[dR] = baiR[dR].absoluteSize();
					}


					// 各ブロックについて contraction を撮って加える
					BTR tR(baiR);
					for (auto& bA_ : btA.blocks()) {
						auto& biA = bA_.first;
						auto& bA = bA_.second;
						for (auto& bB_ : btB.blocks()) {
							auto& biB = bB_.first;
							auto& bB = bB_.second;

							// 現在の block 同士のコントラクションが可能か(非ゼロになるか)判定
							bool canContract = true;
							for (auto& pair : pairs) {
								if (biA[pair.first] != biB[pair.second]) {
									canContract = false;
									break;
								}
							}

							// コントラクションを取れる場合にとった結果を加える
							if (canContract) {

								// 結果のブロックインデックスを取得
								std::array<Index, NR> biR;
								Axis dR = 0;
								for (Axis dA = 0; dA < NA; ++dA) {
									if (!isPairA[dA]) {
										biR[dR] = biA[dA];
										++dR;
									}
								}
								for (Axis dB = 0; dB < NB; ++dB) {
									if (!isPairB[dB]) {
										biR[dR] = biB[dB];
										++dR;
									}
								}

								Eigen::Tensor<Scalar, NR> dtR = contractDenseTensor(bA, bB, pairs);
								tR.addBlock(biR, dtR);
							}

						}
					}

					return tR;





				}

				template<Axis NB_, class Pairs_>
				BlockTensor<Scalar, NumDimensions + NB_ - 2 * std::tuple_size<Pairs_>::value> contract(
					const BlockTensor<Scalar, NB_>& btB,
					const Pairs_& pairs
				)const {

					using BTR = BlockTensor<Scalar, NumDimensions + NB_ - 2 * std::tuple_size<Pairs_>::value>;
					constexpr Axis NA = NumDimensions;
					constexpr Axis NB = NB_;
					constexpr Axis NR = BTR::NumDimensions;
					using DTR = Eigen::Tensor<Scalar, NR>;

					return contract(
						btB,
						pairs,
						[](
							const Eigen::Tensor<Scalar, NA>& dtA,
							const Eigen::Tensor<Scalar, NB>& dtB,
							const Pairs_& pairs_
							)->DTR {
								return DTR(dtA.contract(dtB, pairs_));
						}
					);

				}




				/// <summary>
				/// trace
				/// </summary>
				template<Axis M>
				BlockTensor<Scalar, NumDimensions - M> trace(
					const std::array<Axis, M>& axises
				)const {
					constexpr Axis NR = NumDimensions - M;

					// 返り値テンソルの軸取得
					std::array<Axis, NR> axisR;
					{
						Axis dR = 0;
						for (Axis d = 0; d < NumDimensions; ++d) {
							bool is_trace = false;
							for (auto& ax : axises) {
								if (d == ax) {
									is_trace = true;
									break;
								}
							}
							if (!is_trace) {
								axisR[dR] = d;
								dR += 1;
							}
						}
					}

					// 返り値テンソルの形状取得
					std::array<AddIndices, NR> baiR;
					for (Axis dR = 0; dR < NR; ++dR) {
						baiR[dR] = this->blockAddIndices()[axisR[dR]];
					}

					// 返り値テンソル設定
					BlockTensor<Scalar, NR> btR(baiR);
					for (auto& pr : this->blocks()) {
						auto& bIndices = pr.first;
						auto& block = pr.second;

						bool can_trace = true;
						for (Axis a = 0, na = axises.size(); a < na - 1; ++a) {
							if (bIndices[axises[a]] != bIndices[axises[a + 1]]) {
								can_trace = false;
								break;
							}
						}

						if (can_trace) {
							std::array<BlockIndex, NR> bIndicesR;
							for (Axis dR = 0; dR < NR; ++dR) {
								bIndicesR[dR] = bIndices[axisR[dR]];
							}

							btR.addBlock(bIndicesR, block.trace(axises));
						}


					}

					return btR;
				}



				/// <summary>
				/// 特定の軸を特定のインデックスで固定したテンソルを生成して返す
				/// </summary>
				template<Axis M>
				BlockTensor<Scalar, NumDimensions - M> axisFixed(
					std::array<std::tuple<Axis, BlockIndex, Index>, M>& fixers
				)const {

					// 軸の対応を設定
					constexpr Axis NR = NumDimensions - M;
					std::array<Axis, NR> axisR;	// [dR]->Axis
					{
						Axis dR = 0;
						for (Axis d = 0; d < NumDimensions; ++d) {
							bool is_found = false;
							for (auto& fixer : fixers) {
								Axis ax = std::get<0>(fixer);
								if (d == ax) {
									is_found = true;
									break;
								}
							}
							if (!is_found) {
								axisR[dR] = d;
								++dR;
							}
						}
						if (dR != NR) {
							RuntimeException("in BlockTensor<...>::axisFixed(...), invalid argument");
						}
					}

					// btR の各軸の情報を取得
					std::array<AddIndices, NR> baiR;
					for (Axis dR = 0; dR < NR; ++dR) {
						baiR[dR] = this->blockAddIndices()[axisR[dR]];
					}


					// 返り値を構築
					BlockTensor<Scalar, NR> btR(baiR);
					for (auto& pr : this->blocks()) {
						auto& bIndices = pr.first;
						auto& dt = pr.second;

						// 指定されたブロックが切り取った後に生き残るか判定
						bool is_target = true;
						for (auto& fixer : fixers) {
							auto& axisFix = std::get<0>(fixer);
							auto& biFix = std::get<1>(fixer);
							if (bIndices[axisFix] != biFix) {
								is_target = false;
								break;
							}
						}

						// ブロックを slice, reshape して取り出して設定
						if (is_target) {


							std::array<Index, NumDimensions> offsets;
							for (Axis dR = 0; dR < NR; ++dR) {
								offsets[axisR[dR]] = 0;
							}
							for (auto& fixer : fixers) {
								offsets[std::get<0>(fixer)] = std::get<2>(fixer);
							}

							std::array<Index, NumDimensions> extents;
							for (Axis dR = 0; dR < NR; ++dR) {
								extents[axisR[dR]] = this->intraBlockDimension(axisR[dR], bIndices[axisR[dR]]);
							}
							for (auto& fixer : fixers) {
								extents[std::get<0>(fixer)] = 1;
							}

							std::array<Index, NR> shapeR;
							for (Axis dR = 0; dR < NR; ++dR) {
								shapeR[dR] = this->intraBlockDimension(axisR[dR], bIndices[axisR[dR]]);
							}

							std::array<BlockIndex, NR> bIndicesR;
							for (Axis dR = 0; dR < NR; ++dR) {
								bIndicesR[dR] = bIndices[axisR[dR]];
							}

							Eigen::Tensor<Scalar, NR> t = dt.slice(offsets, extents).reshape(shapeR);

							btR.addBlock(bIndicesR, t);
						}


					}

					return btR;
				}


				template<Axis M>
				BlockTensor<Scalar, NumDimensions - M> axisFixed(
					const std::tuple<Axis, BlockIndex, Index>(&ar)[M]
				)const {
					std::array<std::tuple<Axis, BlockIndex, Index>, M> fixers;
					for (Axis d = 0; d < M; ++d) {
						fixers[d] = ar[d];
					}
					return axisFixed(fixers);
				}





			};




			template<class Scalar, integer_type::Axis NumDims>
			inline BlockTensor<Scalar, NumDims> operator+(
				const BlockTensor<Scalar, NumDims>& a,
				const BlockTensor<Scalar, NumDims>& b
				) {
				BlockTensor<Scalar, NumDims> r = a;
				r += b;
				return r;
			}

			template<class Scalar, integer_type::Axis NumDims>
			inline BlockTensor<Scalar, NumDims> operator-(
				const BlockTensor<Scalar, NumDims>& a,
				const BlockTensor<Scalar, NumDims>& b
				) {
				BlockTensor<Scalar, NumDims> r = a;
				r -= b;
				return r;
			}

			template<class Scalar, integer_type::Axis NumDims>
			inline BlockTensor<Scalar, NumDims> operator*(
				const BlockTensor<Scalar, NumDims>& a,
				const BlockTensor<Scalar, NumDims>& b
				) {
				BlockTensor<Scalar, NumDims> r = a;
				r *= b;
				return r;
			}

			template<class Scalar, integer_type::Axis NumDims>
			inline BlockTensor<Scalar, NumDims> operator/(
				const BlockTensor<Scalar, NumDims>& a,
				const BlockTensor<Scalar, NumDims>& b
				) {
				BlockTensor<Scalar, NumDims> r = a;
				r /= b;
				return r;
			}


		}



		/// <summary>
		/// BlockTensorBase の Derived 引数として利用するための前方宣言
		/// </summary>
		template<class Scalar_, integer_type::Axis NumDimensions_>
		class BlockTensor;



		/// <summary>
		/// BlockTensor の基底クラス(インターフェイス)
		/// Derived& を返す関数の判定のために CRTP を利用している
		/// 
		/// 本クラスは実体化できるが外部での直接構築はしない
		/// 代わりに継承先の BlockTensor を使う
		/// 
		/// ただし、一時オブジェクトとして生成される場合はある。
		/// 
		/// TODO
		/// 
		/// 演算子と contract において Scalar 型の暗黙キャストが行われるが
		/// Eigen::Tensor との対応を考えると良くないので削除予定
		/// 
		/// 
		/// 
		/// 把握しているバグとして 
		/// NumDimensions_==0 のときに一部のメンバ関数を呼ぶとコンパイルが終わらなくなる
		/// これは Eigen のバグである可能性がある
		/// 該当するメンバ関数として以下を把握している
		/// getElement(Indices)
		/// makeDenseTensor()
		/// </summary>
		template<class Scalar_, integer_type::Axis NumDimensions_, class Derived_>
		class BlockTensorBase :public CRTPBase<Derived_> {

			// タイプエイリアス
		public:

			using Derived = Derived_;

			using Index = integer_type::Index;
			using Axis = integer_type::Axis;
			using Scalar = Scalar_;
			using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
			static constexpr Axis NumDimensions = NumDimensions_;

			using Indices = std::array<Index, NumDimensions>;
			using BlockIndex = Index;
			using BlockIndices = std::array<BlockIndex, NumDimensions>;
			using DenseTensorType = Eigen::Tensor<Scalar, NumDimensions>;
			using DenseTensorsType = std::map<std::array<BlockIndex, NumDimensions>, DenseTensorType>;

			using ProductIndicesType = ProductIndices<NumDimensions>;
			using AddIndicesType = AddIndices;
			using BlockAddIndicesType = std::array<AddIndicesType, NumDimensions>;



			// メンバ変数
		protected:

			ProductIndicesType productIndices_;			// [i] <-> [i_0,...,i_{N-1}]
			BlockAddIndicesType blockAddIndices_;		// [i_d] <->[q_d,i_{q_d}]
			DenseTensorsType blocks_;			// [{q_0,...,q_{N-1}}](i_{q_0},...,i_{q_{N-1}})



			// アクセッサ
		public:

			const ProductIndicesType& productIndices()const { return productIndices_; }
			const BlockAddIndicesType& blockAddIndices()const { return blockAddIndices_; }
			const DenseTensorsType& blocks()const { return blocks_; }
			// DenseTensorsType& denseTensors(){ denseTensors_; } // 便利だが危ないので今の所消しておく


			/// <summary>
			/// get the dimension of the given axis
			/// </summary>
			Index dimension(Axis axis)const {
				return productIndices().dimension(axis);
			}

			/// <summary>
			/// get the all dimensions
			/// </summary>
			Indices dimensions()const {
				return productIndices().dimensions();
			}

			/// <summary>
			/// get the number of division of the given axis
			/// </summary>
			BlockIndex blockDimension(Axis axis)const {
				return blockAddIndices()[axis].sizes().size();
			}

			/// <summary>
			/// get the number of division of each axis
			/// </summary>
			BlockIndices blockDimensions()const {
				BlockIndices bIndices;
				for (Axis d = 0; d < NumDimensions; ++d) {
					bIndices[d] = blockAddIndices()[d].sizes().size();
				}
				return bIndices;
			}


			/// <summary>
			/// get the intra-block dimension at the given block
			/// </summary>
			Index intraBlockDimension(Axis axis, BlockIndex si)const {
				return blockAddIndices()[axis].sizes()[si];
			}

			/// <summary>
			/// get the intra-block dimension at the given block
			/// </summary>
			Indices intraBlockDimensions(const BlockIndices& sindices)const {
				auto bdims = Indices();
				for (Axis d = 0; d < NumDimensions; ++d) {
					bdims[d] = blockAddIndices()[d].sizes()[sindices[d]];
				}
				return bdims;
			}

			/// <summary>
			/// ブロックインデックスとブロック内インデックスによる要素取得
			/// map の検索コストがかかるため log の探索時間がかかる
			/// </summary>
			Scalar getElement(
				const BlockIndices& sIndices,
				const Indices& indicesInQn
			)const {
				auto itr = blocks().find(sIndices);
				if (itr == blocks().end()) {
					return Scalar(0.0);
				}
				else {
					return itr->second(indicesInQn);
				}
			}

			/// <summary>
			/// 各軸の絶対インデックスによる要素取得
			/// map の検索コストがかかるため log の探索時間がかかる
			/// </summary>
			Scalar getElement(
				const Indices& indices
			)const {
				BlockIndices sIndices;
				Indices indicesIntra;
				for (Axis d = 0; d < NumDimensions; ++d) {
					sIndices[d] = blockAddIndices()[d].first(indices[d]);
					indicesIntra[d] = blockAddIndices()[d].second(indices[d]);
				}
				return getElement(sIndices, indicesIntra);
			}

			const DenseTensorType& getBlock(const BlockIndices& sIndices)const {
				return blocks().at(sIndices);
			}



			/// <summary>
			/// 各軸の絶対インデックスにより要素を取得
			/// map の検索コストにより log の探索時間がかかる
			/// </summary>
			template<class... Args>
			Scalar operator()(const Args&... args)const {
				return (*this)(make_indices(args...));
			}


			/// <summary>
			/// 各軸の絶対インデックスにより要素を取得
			/// map の検索コストにより log の探索時間がかかる
			/// </summary>
			Scalar operator()(Indices& indices)const {
				BlockIndices sIndices;
				Indices indicesIntra;
				for (Axis d = 0; d < NumDimensions; ++d) {
					sIndices[d] = blockAddIndices()[d].first(indices[d]);
					indicesIntra[d] = blockAddIndices()[d].second(indices[d]);
				}
				return getElement(sIndices, indicesIntra);
			}


			/// <summary>
			/// 対応する Eigen::Tensor 型のオブジェクトを取得
			/// </summary>
			DenseTensorType makeDenseTensor()const {
				DenseTensorType dt = DenseTensorType(dimensions()).setZero();

				for (auto& pr : blocks()) {
					auto& bindices = pr.first;
					auto& block = pr.second;
					std::array<Index, NumDimensions> offsets;
					for (Axis d = 0; d < NumDimensions; ++d) {
						offsets[d] = blockAddIndices()[d].begins()[bindices[d]];
					}
					std::array<Index, NumDimensions> extents;
					for (Axis d = 0; d < NumDimensions; ++d) {
						extents[d] = blockAddIndices()[d].sizes()[bindices[d]];
					}
					dt.slice(offsets, extents) = block;
				}

				return dt;
			}


			/// <summary>
			/// 対応するテンソルを表現する 有限要素の vector を取得
			/// インデックスは各軸ごとの絶対インデックスを表す
			/// </summary>
			std::vector<std::tuple<Indices, Scalar>> makeFiniteElementsVector()const {
				std::vector<std::tuple<Indices, Scalar>> fev;
				for (auto& block_ : blocks()) {
					auto& bIndices = block_.first;
					auto& block = block_.second;
					ProductIndicesType pib(block.dimensions());
					for (Index bi = 0, nbi = pib.absoluteSize(); bi < nbi; ++bi) {
						Indices iIndices = pib.indices(bi);
						Indices indices;
						Scalar value = block(iIndices);
						if (value == Scalar(0.0)) {
							continue;
						}
						for (Axis d = 0; d < NumDimensions; ++d) {
							indices[d] = blockAddIndices()[d].absoluteIndex(bIndices[d], iIndices[d]);
						}
						fev.push_back({ indices,block(iIndices) });
					}
				}
			}



			/// <summary>
			/// ブロックの形状が等しいか判定する
			/// </summary>
			template<class S_, Axis N_, class D_>
			bool equalsBlocks(const BlockTensorBase<S_, N_, D_>& other)const {
				if (dimensions() != other.dimensions()) {
					return false;
				}
				for (Axis d = 0; d < NumDimensions; ++d) {
					if (blockAddIndices()[d].sizes() != other.blockAddIndices()[d].sizes()) {
						return false;
					}
				}
				return true;
			}


			/// <summary>
			/// キャスト
			/// </summary>
			template<class S_>
			BlockTensorBase<S_, NumDimensions, BlockTensor<S_,NumDimensions>> cast()const {
				BlockTensorBase<S_, NumDimensions, BlockTensor<S_,NumDimensions>> btb(blockAddIndices());
				for (auto& pr : blocks()) {
					auto& bIndices = pr.first;
					auto& block = pr.second;
					btb.addBlock(bIndices, block.template cast<S_>());
				}
				return btb;
			}



			// init 系
		protected:


			Derived& init(
				const BlockAddIndicesType& bi_,
				const DenseTensorsType& dt_
			) {

				Indices dims;
				for (Axis d = 0; d < NumDimensions; ++d) {
					dims[d] = bi_[d].absoluteSize();
				}

				// 構築
				productIndices_ = ProductIndicesType(dims);
				blockAddIndices_ = bi_;
				blocks_ = dt_;

				return this->derived();
			}

			/// <summary>
			/// 要素は全てゼロで構築
			/// </summary>
			Derived& init(
				const BlockAddIndicesType& bi_
			) {
				init(bi_, DenseTensorsType());
				return this->derived();
			}


			/// <summary>
			/// 要素は全てゼロ
			///	ブロックは大きい1つ
			/// により構築
			/// </summary>
			Derived& init(
				const ProductIndicesType& pi_
			) {
				BlockAddIndicesType bi_;
				for (Axis d = 0; d < NumDimensions; ++d) {
					bi_[d] = AddIndices({ pi_.dimension(d) });
				}
				init(bi_);
				return this->derived();
			}

			/// <summary>
			/// 要素数0 で構築
			/// </summary>
			Derived& init() {
				std::array<Index, NumDimensions> shape;
				for (Axis d = 0; d < NumDimensions; ++d) {
					shape[d] = 0;
				}
				ProductIndicesType pi_(shape);
				init(pi_);
				return this->derived();
			}

			// コンストラクタ群
		public:
			BlockTensorBase() { init(); }
			BlockTensorBase(const BlockAddIndicesType& bi_) { init(bi_); }
			BlockTensorBase(const ProductIndicesType& pi_) { init(pi_); }
			BlockTensorBase(
				const BlockAddIndicesType& bi_,
				const DenseTensorsType& dt_
			) {
				init(bi_, dt_);
			}


			BlockTensorBase(const BlockTensorBase& other) = default;
			BlockTensorBase(BlockTensorBase&& other) = default;
			BlockTensorBase& operator=(const BlockTensorBase& other) = default;
			BlockTensorBase& operator=(BlockTensorBase&& other) = default;






		public:
			// ミューテータ

			/// <summary>
			/// ブロックを加算する
			/// ゼロの評価をしない
			/// </summary>
			Derived& addBlock(
				const BlockIndices& sindices,
				const DenseTensorType& t
			) {
				if (t.dimensions() != intraBlockDimensions(sindices)) {
					throw RuntimeException("in BlockTensorBase::addBlock(...), invalid dimensions");
				}
				else {
					auto itr = blocks_.find(sindices);
					if (itr == blocks_.end()) {
						blocks_[sindices] = t;
					}
					else {
						itr->second += t;
					}
				}
				return this->derived();
			}




			Derived& mulBlock(
				const BlockIndices& sindices,
				const DenseTensorType& t
			) {
				if (t.dimensions() != intraBlockDimensions(sindices)) {
					throw RuntimeException("");
				}
				else {
					auto itr = blocks_.find(sindices);
					if (itr == blocks_.end()) {
					}
					else {
						itr->second *= t;
					}
				}
				return this->derived();
			}


			/// <summary>
			/// ゼロクリア
			/// 形状は変更しない
			/// </summary>
			void clear() {
				blocks_.clear();
			}





			/// <summary>
			/// ブロックインデックスとブロック内インデックスによる要素の設定
			/// 要素が所属するブロックが存在しない場合は作る
			/// map の検索コストがかかるため log の探索時間がかかる
			/// </summary>
			Derived& setElement(
				const BlockIndices& sIndices,
				const Indices& indicesIntra,
				const Scalar& value
			) {
				auto itr = blocks().find(sIndices);
				if (itr == blocks().end()) {
					if (value == Scalar(0.0)) {
					}
					else {
						DenseTensorType dt = DenseTensorType(intraBlockDimensions(sIndices)).setZero();
						dt(indicesIntra) = value;
						blocks_[sIndices] = dt;
					}
				}
				else {
					blocks_[sIndices](indicesIntra) = value;
				}

				return this->derived();
			}

			/// <summary>
			/// 各軸の絶対インデックスによる要素取得よる要素の設定
			/// 要素が所属するブロックが存在しない場合は作る
			/// map の検索コストがかかるため log の探索時間がかかる
			/// </summary>
			Derived& setElement(
				const Indices& indices,
				const Scalar& value
			) {
				BlockIndices sIndices;
				Indices intraIndices;
				for (Axis d = 0; d < NumDimensions; ++d) {
					sIndices[d] = blockAddIndices()[d].first(indices[d]);
					intraIndices[d] = blockAddIndices()[d].second(indices[d]);
				}
				return setElement(sIndices, intraIndices, value);
			}



			/// <summary>
			/// ブロックをセットする
			/// ゼロ判定をしない(必ずブロック分のストレージが確保される)
			/// </summary>
			Derived& setBlock(
				const BlockIndices& sIndices,
				DenseTensorType& dt
			) {
				if (dt.dimensions() != intraBlockDimensions(sIndices)) {
					RuntimeException("In BTensor::setBlock(...), the shape of block is invalid");
				}
				else {
					blocks_[sIndices] = dt;
				}
				return this->derived();
			}

			/// <summary>
			/// 指定したブロックを削除する
			/// 対応するストレージが開放される
			/// ゼロ判定をしない(必ずブロック分のストレージが確保される)
			/// </summary>
			Derived& eraseBlock(const BlockIndices& sIndices) {
				blocks_.erase(sIndices);
				return this->derived();;
			}

			/// <summary>
			/// 密行列から構築
			/// 各ブロックの形状は維持する
			/// ゼロ判定を行う(ゼロブロックのストレージは確保されない)
			/// </summary>
			Derived& setFromDenseTensor(const DenseTensorType& dt) {
				if (dimensions() != dt.dimensions()) {
					throw RuntimeException("in BTensor::setFromDenseTensor(...), dense tensor dimensions are invalid");
				}

				BlockIndices bdims = blockDimensions();
				auto bidx = ProductIndicesType(bdims);
				for (Index b = 0, nb = bidx.absoluteSize(); b < nb; ++b) {
					BlockIndices bIndices = bidx.indices(b);
					DenseTensorType block;
					Indices offsets;
					Indices extents;
					Indices iota;
					for (Axis d = 0; d < NumDimensions; ++d) {
						offsets[d] = blockAddIndices()[d].begins()[bIndices[d]];
						extents[d] = blockAddIndices()[d].sizes()[bIndices[d]];
						iota[d] = d;
					}
					block = dt.slice(offsets, extents);

					Eigen::Tensor<RealScalar, 0> max_ = block.abs().maximum(iota);
					if (max_() == RealScalar(0.0)) {
						continue;
					}
					else {
						setBlock(bIndices, block);
					}
				}

				return this->derived();
			}


			Derived& shuffleInPlace(const std::array<Axis, NumDimensions>& sfl) {

				BlockAddIndicesType baIndices;
				for (Axis d = 0; d < NumDimensions; ++d) {
					baIndices[d] = blockAddIndices()[sfl[d]];
				}
				DenseTensorsType dts;
				for (auto& block_ : blocks()) {
					auto& bIndices = block_.first;
					auto& block = block_.second;
					BlockIndices bIndicesNew;
					for (Axis d = 0; d < NumDimensions; ++d) {
						bIndicesNew[d] = bIndices[sfl[d]];
					}
					dts[bIndicesNew] = DenseTensorType(block.shuffle(sfl));
				}


				init(baIndices, dts);
				return this->derived();
			}


			Derived& blockShuffleInPlace(Axis axis, const std::vector<BlockIndex>& bShuffle) {
				std::vector<BlockIndex> reverseShuffle;
				reverseShuffle.resize(bShuffle.size());
				for (BlockIndex bi = 0, nbi = blockAddIndices()[axis].sizes().size(); bi < nbi; ++bi) {
					reverseShuffle[bShuffle[bi]] = bi;
				}

				std::vector<BlockIndex> sizes;
				sizes.resize(blockAddIndices()[axis].sizes().size());

				for (BlockIndex bi = 0, nbi = sizes.size(); bi < nbi; ++bi) {
					sizes[bi] = blockAddIndices()[axis].sizes()[bShuffle[bi]];
				}
				blockAddIndices_[axis] = AddIndices(sizes);

				DenseTensorsType dt;
				for (auto& block_ : blocks_) {
					auto key = block_.first;
					key[axis] = reverseShuffle[key[axis]];
					std::swap(dt[key], block_.second);
				}
				blocks_ = dt;
				return this->derived();
			}


			Derived& blockShuffleInPlace(const std::array<std::vector<BlockIndex>, NumDimensions>& bshuffles) {
				for (Axis d = 0; d < NumDimensions; ++d) {
					blockShuffleInPlace(d, bshuffles[d]);
				}
				return this->derived();
			}



			Derived& setFromFiniteElementsVector(const std::vector<std::tuple<Indices, Scalar>>& v) {
				clear();
				for (auto& e_ : v) {
					Indices indices = std::get<0>(e_);
					Scalar value = std::get<1>(e_);
					setElement(indices, value);
				}
				return this->derived();
			}

			/// <summary>
			/// ブロックのうち絶対値最大の要素のが threshold 以下のブロックを削除する(ゼロとする)
			/// 全体の形状は変化しない
			/// </summary>
			void truncate(const RealScalar& threshold) {
				std::vector<BlockIndices> erase_keys;
				for (auto& block_ : blocks_) {
					auto& block = block_.second;
					Eigen::Tensor<RealScalar, 0> max_ = block.abs().maximum();
					if (max_() <= threshold) {
						erase_keys.push_back(block_.first);
					}
				}
				for (auto& key : erase_keys) {
					blocks_.erase(key);
				}
			}


			Derived& reblock(const BlockAddIndicesType& bi) {
				auto v = this->makeFiniteElementsVector();
				this->init(bi);
				this->setFromFiniteElementsVector(v);
				return this->derived();
			}


			Derived& conjugateInPlace() {
				for (auto& bt_ : this->blocks_) {
					bt_.second = bt_.second.conjugate();
				}
				return this->derived();
			}

			Derived& scalarMultiple(const Scalar& c) {
				for (auto& block_ : this->blocks_) {
					auto& block = block_.second;
					block = block*block.constant(c);
				}
				return this->derived();
			}

			template<class D_>
			Derived& operator+=(const BlockTensorBase<Scalar, NumDimensions, D_>& other) {
				if (!this->derived().equalsBlocks(other)) {
					RuntimeException("");
				}
				for (auto& block_ : other.blocks()) {
					auto& bIndices = block_.first;
					auto& block = block_.second;
					this->derived().addBlock(bIndices, block);
				}
				return this->derived();
			}

			template<class D_>
			Derived& operator-=(const BlockTensorBase<Scalar, NumDimensions, D_>& other) {
				return this->derived() += (-other);
			}

			template<class D_>
			Derived& operator*=(const BlockTensorBase<Scalar, NumDimensions, D_>& other) {
				if (!this->derived().equalsBlocks(other)) {
					RuntimeException("");
				}
				for (auto& block_ : other.blocks()) {
					auto& bIndices = block_.first;
					auto& block = block_.second;
					this->derived().mulBlock(bIndices, block);
				}
				return this->derived();
			}

			template<class D_>
			Derived& operator/=(const BlockTensorBase<Scalar, NumDimensions, D_>& other) {
				if (!this->derived().equalsBlocks(other)) {
					RuntimeException("");
				}
				for (auto& block_ : other.blocks()) {
					auto& bIndices = block_.first;
					auto& block = block_.second;
					DenseTensorType block_inv = DenseTensorType(block.dimensions()).setConstant(Scalar(1.0)) / block;
					this->derived().mulBlock(bIndices, block_inv);
				}
				return this->derived();
			}



		public:
			// 自身と同じ型を返す系


			Derived shuffle(const std::array<Axis, NumDimensions>& sfl)const {
				Derived drvd(this->derived());
				drvd.shuffleInPlace(sfl);
				return drvd;
			}

			Derived blockShuffle(Axis axis, const std::vector<BlockIndex>& bShuffle)const {
				Derived drvd(this->derived());
				drvd.blockShuffleInPlace(axis, bShuffle);
				return drvd;
			}

			Derived blockShuffle(const std::array<std::vector<BlockIndex>, NumDimensions>& bShuffle)const {
				Derived drvd(this->derived());
				drvd.blockShuffleInPlace(bShuffle);
				return drvd;
			}

			/// <summary>
			/// ブロックのうち絶対値最大の要素のが threshold 以下のブロックを削除したものを返す
			/// </summary>
			Derived truncated(const RealScalar& threshold)const {
				Derived drvd(this->derived());
				drvd.truncate(threshold);
				return drvd;
			}


			Derived conjugate()const {
				Derived ret(this->derived());
				ret.conjugateInPlace();
				return ret;
			}

			Derived scalarMultipled(const Scalar& c)const {
				Derived drvd(this->derived());
				drvd.scalarMultiple(c);
				return drvd;
			}


			Derived operator+()const {
				return this->derived();
			}

			Derived operator-()const {
				Derived drvd(this->derived());
				drvd.scalarMultiple(Scalar(-1.0));
				return drvd;
			}



			/// <summary>
			/// コントラクションの型推論と計算を行う構造体
			/// </summary>
			template<class TensorA_, class TensorB_, class Pairs_>
			struct ContractedTensorTraits {
				using TensorA = TensorA_;
				using TensorB = TensorB_;

				using DerivedA = typename TensorA::Derived;
				using DerivedB = typename TensorB::Derived;

				using ScalarA = typename TensorA::Scalar;
				using ScalarB = typename TensorB::Scalar;
				using ScalarR = typename std::remove_reference<decltype(ScalarA()* ScalarB())>::type;


				using Pairs = Pairs_;
				static constexpr Axis NP = std::tuple_size<Pairs>::value;
				static constexpr Axis NA = TensorA::NumDimensions;
				static constexpr Axis NB = TensorB::NumDimensions;
				static constexpr Axis NR = NA + NB - 2 * NP;

				using DerivedR = BlockTensor<ScalarR, NR>;
				using TensorR = BlockTensorBase<ScalarR, NR, DerivedR>;

				using IndicesA = typename TensorA::Indices;
				using IndicesB = typename TensorB::Indices;
				using IndicesR = typename TensorR::Indices;


			};




			template<Axis NB, class DerivedB, class Pairs, class FContractDenseTensor>
			BlockTensorBase<
				Scalar,
				NumDimensions+NB-2* std::tuple_size<Pairs>::value,
				BlockTensor<Scalar, NumDimensions + NB - 2 * std::tuple_size<Pairs>::value>
			> contract(
				const BlockTensorBase<Scalar, NB, DerivedB>& btB,
				const Pairs& pairs,
				const FContractDenseTensor& contractDenseTensor
			)const{

				// 変数準備
				constexpr Axis NA = NumDimensions;
				constexpr Axis NP = std::tuple_size<Pairs>::value;
				constexpr Axis NR = NA+NB-2*NP;
				
				const auto& btA = *this;

				using BTR = BlockTensorBase<Scalar,NR,BlockTensor<Scalar,NR>>;

				// 引数エラーチェック
				{
					bool is_valid = true;
					for (auto& pair : pairs) {
						Axis dA = pair.first;
						Axis dB = pair.second;
						if (btA.blockAddIndices()[dA].sizes() != btB.blockAddIndices()[dB].sizes()) {
							is_valid = false;
							break;
						}
					}
					if (!is_valid) {
						throw RuntimeException("in BlockTensorBase{...}::contract(...), invalid contractions");
					}
				}




				// A, B の各軸について contraction をとる軸かどうかを判定して保持
				std::array<bool, NA> isPairA;
				for (Axis dA = 0; dA < NA; ++dA) {
					bool is_pair = false;
					for (auto& pair : pairs) {
						if (pair.first == dA) {
							is_pair = true;
							break;
						}
					}
					isPairA[dA] = is_pair;
				}

				std::array<bool, NB> isPairB;
				for (Axis dB = 0; dB < NB; ++dB) {
					bool is_pair = false;
					for (auto& pair : pairs) {
						if (pair.second == dB) {
							is_pair = true;
							break;
						}
					}
					isPairB[dB] = is_pair;
				}

				// R の各軸についてブロック構造を設定
				std::array<AddIndices, NR> baiR;
				{
					Axis dR = 0;
					for (Axis dA = 0; dA < NA; ++dA) {
						if (!isPairA[dA]) {
							baiR[dR] = btA.blockAddIndices()[dA];
							++dR;
						}
					}
					for (Axis dB = 0; dB < NB; ++dB) {
						if (!isPairB[dB]) {
							baiR[dR] = btB.blockAddIndices()[dB];
							++dR;
						}
					}
				}

				// R の各軸についてサイズを取得
				std::array<int, NR> dimsR;
				for (Axis dR = 0; dR < NR; ++dR) {
					dimsR[dR] = baiR[dR].absoluteSize();
				}


				// 各ブロックについて contraction を撮って加える
				BTR btR(baiR);
				for (auto& bA_ : btA.blocks()) {
					auto& biA = bA_.first;
					auto& bA = bA_.second;
					for (auto& bB_ : btB.blocks()) {
						auto& biB = bB_.first;
						auto& bB = bB_.second;

						// 現在の block 同士のコントラクションが可能か(非ゼロになるか)判定
						bool canContract = true;
						for (auto& pair : pairs) {
							if (biA[pair.first] != biB[pair.second]) {
								canContract = false;
								break;
							}
						}

						// コントラクションを取れる場合にとった結果を加える
						if (canContract) {

							// 結果のブロックインデックスを取得
							std::array<Index, NR> biR;
							Axis dR = 0;
							for (Axis dA = 0; dA < NA; ++dA) {
								if (!isPairA[dA]) {
									biR[dR] = biA[dA];
									++dR;
								}
							}
							for (Axis dB = 0; dB < NB; ++dB) {
								if (!isPairB[dB]) {
									biR[dR] = biB[dB];
									++dR;
								}
							}

							Eigen::Tensor<Scalar, NR> dtR = contractDenseTensor(bA, bB, pairs);
							btR.addBlock(biR, dtR);
						}

					}
				}

				return btR;

			}


			template<Axis NB, class DerivedB, class Pairs>
			BlockTensorBase<
				Scalar,
				NumDimensions + NB - 2 * std::tuple_size<Pairs>::value,
				BlockTensor<Scalar, NumDimensions + NB - 2 * std::tuple_size<Pairs>::value>
			> contract(
				const BlockTensorBase<Scalar, NB, DerivedB>& btB,
				const Pairs& pairs
			)const {

				// 変数準備
				constexpr Axis NA = NumDimensions;
				constexpr Axis NP = std::tuple_size<Pairs>::value;
				constexpr Axis NR = NA + NB - 2 * NP;

				const auto& btA = *this;

				using DenseTensorA = typename BlockTensorBase<Scalar, NumDimensions, Derived>::DenseTensorType;
				using DenseTensorB = typename BlockTensorBase<Scalar, NB, DerivedB>::DenseTensorType;
				using DenseTensorR = typename BlockTensorBase<Scalar, NR, BlockTensor<Scalar,NR>>::DenseTensorType;

				return contract(
					btB,
					pairs,
					[](
						const DenseTensorA& dtA,
						const DenseTensorB& dtB,
						const Pairs& pairs_
						)->DenseTensorR {
							return DenseTensorR(dtA.contract(dtB, pairs_));
					}
				);
			}


			




			/// <summary>
			/// trace
			/// </summary>
			template<Axis M>
			BlockTensorBase<Scalar, NumDimensions - M, BlockTensor<Scalar, NumDimensions - M>> trace(
				const std::array<Axis, M>& axises
			)const {
				constexpr Axis NR = NumDimensions - M;

				// 返り値テンソルの軸取得
				std::array<Axis, NR> axisR;
				{
					Axis dR = 0;
					for (Axis d = 0; d < NumDimensions; ++d) {
						bool is_trace = false;
						for (auto& ax : axises) {
							if (d == ax) {
								is_trace = true;
								break;
							}
						}
						if (!is_trace) {
							axisR[dR] = d;
							dR += 1;
						}
					}
				}

				// 返り値テンソルの形状取得
				std::array<AddIndices, NR> baiR;
				for (Axis dR = 0; dR < NR; ++dR) {
					baiR[dR] = this->blockAddIndices()[axisR[dR]];
				}

				// 返り値テンソル設定
				BlockTensorBase<Scalar, NumDimensions - M, BlockTensor<Scalar, NumDimensions - M>> btR(baiR);
				for (auto& pr : this->blocks()) {
					auto& bIndices = pr.first;
					auto& block = pr.second;

					bool can_trace = true;
					for (Axis a = 0, na = axises.size(); a < na - 1; ++a) {
						if (bIndices[axises[a]] != bIndices[axises[a + 1]]) {
							can_trace = false;
							break;
						}
					}

					if (can_trace) {
						std::array<BlockIndex, NR> bIndicesR;
						for (Axis dR = 0; dR < NR; ++dR) {
							bIndicesR[dR] = bIndices[axisR[dR]];
						}

						btR.addBlock(bIndicesR, block.trace(axises));
					}


				}

				return btR;
			}



			/// <summary>
			/// 特定の軸を特定のインデックスで固定したテンソルを生成して返す
			/// </summary>
			template<Axis M>
			BlockTensorBase<Scalar, NumDimensions - M, BlockTensor<Scalar, NumDimensions - M>> axisFixed(
				std::array<std::tuple<Axis, BlockIndex, Index>, M>& fixers
			)const {

				// 軸の対応を設定
				constexpr Axis NR = NumDimensions - M;
				std::array<Axis, NR> axisR;	// [dR]->Axis
				{
					Axis dR = 0;
					for (Axis d = 0; d < NumDimensions; ++d) {
						bool is_found = false;
						for (auto& fixer : fixers) {
							Axis ax = std::get<0>(fixer);
							if (d == ax) {
								is_found = true;
								break;
							}
						}
						if (!is_found) {
							axisR[dR] = d;
							++dR;
						}
					}
					if (dR != NR) {
						RuntimeException("in BlockTensor<...>::axisFixed(...), invalid argument");
					}
				}

				// btR の各軸の情報を取得
				std::array<AddIndices, NR> baiR;
				for (Axis dR = 0; dR < NR; ++dR) {
					baiR[dR] = this->blockAddIndices()[axisR[dR]];
				}


				// 返り値を構築
				BlockTensorBase<Scalar, NumDimensions - M, BlockTensor<Scalar, NumDimensions - M>> btR(baiR);
				for (auto& pr : this->blocks()) {
					auto& bIndices = pr.first;
					auto& dt = pr.second;

					// 指定されたブロックが切り取った後に生き残るか判定
					bool is_target = true;
					for (auto& fixer : fixers) {
						auto& axisFix = std::get<0>(fixer);
						auto& biFix = std::get<1>(fixer);
						if (bIndices[axisFix] != biFix) {
							is_target = false;
							break;
						}
					}

					// ブロックを slice, reshape して取り出して設定
					if (is_target) {


						std::array<Index, NumDimensions> offsets;
						for (Axis dR = 0; dR < NR; ++dR) {
							offsets[axisR[dR]] = 0;
						}
						for (auto& fixer : fixers) {
							offsets[std::get<0>(fixer)] = std::get<2>(fixer);
						}

						std::array<Index, NumDimensions> extents;
						for (Axis dR = 0; dR < NR; ++dR) {
							extents[axisR[dR]] = this->intraBlockDimension(axisR[dR], bIndices[axisR[dR]]);
						}
						for (auto& fixer : fixers) {
							extents[std::get<0>(fixer)] = 1;
						}

						std::array<Index, NR> shapeR;
						for (Axis dR = 0; dR < NR; ++dR) {
							shapeR[dR] = this->intraBlockDimension(axisR[dR], bIndices[axisR[dR]]);
						}

						std::array<BlockIndex, NR> bIndicesR;
						for (Axis dR = 0; dR < NR; ++dR) {
							bIndicesR[dR] = bIndices[axisR[dR]];
						}

						Eigen::Tensor<Scalar, NR> t = dt.slice(offsets, extents).reshape(shapeR);

						btR.addBlock(bIndicesR, t);
					}


				}

				return btR;
			}


			template<Axis M>
			BlockTensorBase<Scalar, NumDimensions - M, BlockTensor<Scalar, NumDimensions - M>> axisFixed(
				const std::tuple<Axis, BlockIndex, Index>(&ar)[M]
			)const {
				std::array<std::tuple<Axis, BlockIndex, Index>, M> fixers;
				for (Axis d = 0; d < M; ++d) {
					fixers[d] = ar[d];
				}
				return axisFixed(fixers);
			}





		};





		/// <summary>
		/// BlockTensor の実装
		/// コンストラクタを実装している
		/// </summary>
		template<class Scalar_, integer_type::Axis NumDimensions_>
		class BlockTensor :public BlockTensorBase<Scalar_, NumDimensions_, BlockTensor<Scalar_, NumDimensions_>> {
		public:


			using Base = BlockTensorBase<Scalar_, NumDimensions_, BlockTensor<Scalar_, NumDimensions_>>;


			using Index = typename Base::Index;
			using Axis = typename Base::Axis;
			using Scalar = typename Base::Scalar;
			using RealScalar = typename Base::RealScalar;
			static constexpr Axis NumDimensions = Base::NumDimensions;

			using Indices = typename Base::Indices;
			using BlockIndex = typename Base::BlockIndex;
			using BlockIndices = typename Base::BlockIndices;
			using DenseTensorType = typename Base::DenseTensorType;
			using DenseTensorsType = typename Base::DenseTensorsType;

			using ProductIndicesType = typename Base::ProductIndicesType;
			using AddIndicesType = typename Base::AddIndicesType;
			using BlockAddIndicesType = typename Base::BlockAddIndicesType;





			BlockTensor() { Base::init(); }

			BlockTensor(
				const BlockAddIndicesType& bi_
			) {
				Base::init(bi_);
			}

			/// <summary>
			/// initializer_list による構築を誘導するためのコンストラクタ
			/// </summary>
			BlockTensor(
				const std::vector<std::vector<Index>>& vec
			) {
				BlockAddIndicesType bai;
				for (Axis d = 0; d < NumDimensions; ++d) {
					bai[d] = AddIndicesType(vec[d]);
				}
				Base::init(bai);
			}


			BlockTensor(
				const BlockAddIndicesType& bi_,
				const DenseTensorsType& dt_
			) {
				Base::init(bi_, dt_);
			}


			/// <summary>
			/// BlockTensorBase からのダウンキャスト
			/// Derived は自由
			/// </summary>
			template<class Derived>
			BlockTensor(
				const BlockTensorBase<Scalar, NumDimensions, Derived>& other
			) {
				Base::init(other.blockAddIndices());
				for (auto& dt_ : other.blocks()) {
					this->blocks_[dt_.first] = dt_.second;
				}
			}


			BlockTensor(const BlockTensor& other) = default;
			BlockTensor(BlockTensor&& other) = default;
			BlockTensor& operator=(const BlockTensor& other) = default;
			BlockTensor& operator=(BlockTensor&& other) = default;


			template<class Derived>
			BlockTensor& operator=(
				const BlockTensorBase<Scalar, NumDimensions, Derived>& other
				) {
				*this = BlockTensor(other);
				return *this;
			}


		};


		template<class Scalar,integer_type::Axis NumDims,class DerivedA,class DerivedB>
		inline BlockTensor<Scalar, NumDims> operator+(
			const BlockTensorBase<Scalar, NumDims, DerivedA>& a,
			const BlockTensorBase<Scalar, NumDims, DerivedB>& b
			) {
			BlockTensor<Scalar, NumDims> r = a;
			r += b;
			return r;
		}
		template<class Scalar, integer_type::Axis NumDims, class DerivedA, class DerivedB>
		inline BlockTensor<Scalar, NumDims> operator-(
			const BlockTensorBase<Scalar, NumDims, DerivedA>& a,
			const BlockTensorBase<Scalar, NumDims, DerivedB>& b
			) {
			BlockTensor<Scalar, NumDims> r = a;
			r -= b;
			return r;
		}
		template<class Scalar, integer_type::Axis NumDims, class DerivedA, class DerivedB>
		inline BlockTensor<Scalar, NumDims> operator*(
			const BlockTensorBase<Scalar, NumDims, DerivedA>& a,
			const BlockTensorBase<Scalar, NumDims, DerivedB>& b
			) {
			BlockTensor<Scalar, NumDims> r = a;
			r *= b;
			return r;
		}
		template<class Scalar, integer_type::Axis NumDims, class DerivedA, class DerivedB>
		inline BlockTensor<Scalar, NumDims> operator/(
			const BlockTensorBase<Scalar, NumDims, DerivedA>& a,
			const BlockTensorBase<Scalar, NumDims, DerivedB>& b
			) {
			BlockTensor<Scalar, NumDims> r = a;
			r /= b;
			return r;
		}


		






		/// <summary>
		/// einsum(...).from(...).to(...)
		/// を BlockTensor に拡張している
		/// 
		/// TODO 1テンソルはまだ実装していない
		/// </summary>
		namespace einsum_impl {

			/// <summary>
			/// einsum 実装のための
			/// テンプレート特殊化
			/// 2テンソル用
			/// </summary>
			template<class ScalarA_, Axis NA_, class ScalarB_, Axis NB_, class CRIIndicesesA_, class CRIIndicesesB_, class IIndicesR_>
			class ToImpl<std::tuple<const BlockTensor<ScalarA_, NA_>&, const BlockTensor<ScalarB_, NB_>&>, std::tuple<CRIIndicesesA_, CRIIndicesesB_>, IIndicesR_> {
			public:
				using Axis = integer_type::Axis;
				using Index = integer_type::Index;
				using BlockIndex = integer_type::Index;
				using IIndex = std::string;

				using ScalarA = ScalarA_;
				using ScalarB = ScalarB_;
				using ScalarR = typename std::remove_reference<decltype(ScalarA()* ScalarB())>::type;
				static constexpr Axis NA = NA_;
				static constexpr Axis NB = NB_;
				static constexpr Axis NR = std::tuple_size<IIndicesR_>::value;

				using BTA = BlockTensor<ScalarA, NA>;
				using BTB = BlockTensor<ScalarB, NB>;

				using IIndicesA = std::array<IIndex, NA>;
				using IIndicesB = std::array<IIndex, NB>;
				using IIndicesR = IIndicesR_;


				using BTR = BlockTensor<ScalarR, NR>;
				using TensorR = BTR;

				std::tuple<const BTA&, const BTB&> ts;
				std::tuple<IIndicesA, IIndicesB> iiTs;
				IIndicesR iiR;


				/// <summary>
				/// einsum を計算する上で軸の対応が適当か判定する
				/// 現在実装していない
				/// </summary>
				bool isValid()const {
					const BTA& btA = std::get<0>(ts);
					const BTB& btB = std::get<1>(ts);
					const IIndicesA& iiA = std::get<0>(iiTs);
					const IIndicesA& iiB = std::get<1>(iiTs);

					bool isValid_ = true;

					// 軸の対応が適切かどうか判定
					for (Axis dR = 0; dR < NR; ++dR) {
						bool is_found = false;
						for (Axis dA = 0; dA < NA; ++dA) {
							if (iiA[dA] == iiR[dR]) {
								is_found = true;
								break;
							}
						}
						for (Axis dB = 0; dB < NB; ++dB) {
							if (iiB[dB] == iiR[dR]) {
								is_found = true;
								break;
							}
						}
						if (is_found == false) {
							isValid_ = false;
							break;
						}

					}


					// 軸のサイズが適切かどうか判定
					for (Axis dA = 0; dA < NA; ++dA) {
						for (Axis dA_ = 0; dA_ < NA; ++dA_) {
							if (iiA[dA] == iiA[dA_]) {
								if (btA.blockAddIndices()[dA] != btA.blockAddIndices()[dA_]) {
									isValid_ = false;
								}
							}
						}
					}
					for (Axis dA = 0; dA < NA; ++dA) {
						for (Axis dB_ = 0; dB_ < NB; ++dB_) {
							if (iiA[dA] == iiB[dB_]) {
								if (btA.blockAddIndices()[dA] != btB.blockAddIndices()[dB_]) {
									isValid_ = false;
								}
							}
						}
					}
					for (Axis dB = 0; dB < NB; ++dB) {
						for (Axis dA_ = 0; dA_ < NA; ++dA_) {
							if (iiB[dB] == iiA[dA_]) {
								if (btB.blockAddIndices()[dB] != btA.blockAddIndices()[dA_]) {
									isValid_ = false;
								}
							}
						}
					}
					for (Axis dB = 0; dB < NB; ++dB) {
						for (Axis dB_ = 0; dB_ < NB; ++dB_) {
							if (iiB[dB] == iiB[dB_]) {
								if (btB.blockAddIndices()[dB] != btB.blockAddIndices()[dB_]) {
									isValid_ = false;
								}
							}
						}
					}


					return isValid_;
				}


				BTR compute()const {
					const BTA& btA = std::get<0>(ts);
					const BTB& btB = std::get<1>(ts);
					const IIndicesA& iiA = std::get<0>(iiTs);
					const IIndicesA& iiB = std::get<1>(iiTs);



					if (!isValid()) {
						throw RuntimeException("invalid argument");
						return BTR();
					}




					// A と B の kronecker product のインデックスの情報を取得
					std::array<IIndex, NA + NB> iiAB;
					for (Axis d = 0; d < NA; ++d) {
						iiAB[d] = iiA[d];
					}
					for (Axis d = 0; d < NB; ++d) {
						iiAB[NA + d] = iiB[d];
					}

					std::array<AddIndices, NA + NB> baiAB;
					for (Axis d = 0; d < NA; ++d) {
						baiAB[d] = btA.blockAddIndices()[d];
					}
					for (Axis d = 0; d < NB; ++d) {
						baiAB[NA + d] = btB.blockAddIndices()[d];
					}

					// 返すテンソルの軸の情報を取得

					std::array<AddIndices, NR> baiR;
					for (Axis d = 0; d < NR; ++d) {
						for (Axis dab = 0; dab < NA + NB; ++dab) {
							if (iiAB[dab] == iiR[d]) {
								baiR[d] = baiAB[dab];
								break;
							}
						}
					}

					std::array<Axis, NR> axR;
					for (Axis dR = 0; dR < NR; ++dR) {
						bool isFound = false;
						for (Axis dA = 0; dA < NA; ++dA) {
							if (isFound) {
								break;
							}
							if (iiA[dA] == iiR[dR]) {
								axR[dR] = dA;
								isFound = true;
							}
						}
						for (Axis dB = 0; dB < NA; ++dB) {
							if (isFound) {
								break;
							}
							if (iiB[dB] == iiR[dR]) {
								axR[dR] = NA + dB;
								isFound = true;
							}
						}
					}

					std::array<Index, NR> dimsR;
					for (Axis d = 0; d < NR; ++d) {
						dimsR[d] = baiR[d].absoluteSize();
					}


					// 返すテンソルを計算
					BTR btR(baiR);
					for (auto& bA_ : btA.blocks()) {
						const std::array<BlockIndex, NA>& biA = bA_.first;
						const Eigen::Tensor<ScalarA, NA>& bA = bA_.second;
						for (auto& bB_ : btB.blocks()) {
							const std::array<BlockIndex, NB>& biB = bB_.first;
							const Eigen::Tensor<ScalarB, NB>& bB = bB_.second;

							// 
							bool canCalc = true;
							for (Axis dAB = 0; dAB < NA + NB; ++dAB) {
								for (Axis dAB_ = 0; dAB_ < NA + NB; ++dAB_) {
									if (iiAB[dAB] == iiAB[dAB_]) {
										if (dAB < NA && dAB_ < NA) {
											if (biA[dAB] != biA[dAB_]) {
												canCalc = false;
												break;
											}
										}
										if (dAB >= NA && dAB_ < NA) {
											if (biB[dAB - NA] != biA[dAB_]) {
												canCalc = false;
												break;
											}
										}
										if (dAB < NA && dAB_ >= NA) {
											if (biA[dAB] != biB[dAB_ - NA]) {
												canCalc = false;
												break;
											}
										}
										if (dAB >= NA && dAB_ >= NA) {
											if (biB[dAB - NA] != biB[dAB_ - NA]) {
												canCalc = false;
												break;
											}
										}
									}
								}
								if (!canCalc) {
									break;
								}
							}
							if (canCalc) {
								using BlockIndicesR = typename BTR::BlockIndices;
								BlockIndicesR biR;
								for (Axis dR = 0; dR < NR; ++dR) {
									if (axR[dR] < NA) {
										biR[dR] = biA[axR[dR]];
									}
									if (axR[dR] >= NA) {
										biR[dR] = biB[axR[dR] - NA];
									}
								}
								btR.addBlock(
									biR,
									einsum(bA, bB).from(iiA, iiB).to(iiR)
								);
							}
						}
					}

					return btR;
				}


			};


			/// <summary>
			/// einsum 実装のための
			/// テンプレート特殊化
			/// 1テンソル用
			/// </summary>
			template<class ScalarA_, Axis NA_, class CRIIndicesesA_, class IIndicesR_>
			class ToImpl<std::tuple<const BlockTensor<ScalarA_, NA_>&>, std::tuple<CRIIndicesesA_>, IIndicesR_> {
			public:
				using Axis = integer_type::Axis;
				using Index = integer_type::Index;
				using BlockIndex = integer_type::Index;
				using IIndex = std::string;

				using ScalarA = ScalarA_;

				using ScalarR = ScalarA;
				static constexpr Axis NA = NA_;
				static constexpr Axis NR = std::tuple_size<IIndicesR_>::value;

				using BTA = BlockTensor<ScalarA, NA>;

				using IIndicesA = std::array<IIndex, NA>;
				using IIndicesR = IIndicesR_;

				using BTR = BlockTensor<ScalarR, NR>;
				using TensorR = BTR;

				std::tuple<const BTA&> ts;
				std::tuple<IIndicesA> iiTs;
				IIndicesR iiR;


				/// <summary>
				/// einsum を計算する上で軸の対応が適当か判定する
				/// 現在実装していない
				/// </summary>
				bool isValid()const {
					const BTA& btA = std::get<0>(ts);
					const IIndicesA& iiA = std::get<0>(iiTs);

					bool isValid_ = true;

					// 軸の対応が適切かどうか判定
					for (Axis dR = 0; dR < NR; ++dR) {
						bool is_found = false;
						for (Axis dA = 0; dA < NA; ++dA) {
							if (iiA[dA] == iiR[dR]) {
								is_found = true;
								break;
							}
						}
						if (is_found == false) {
							isValid_ = false;
							break;
						}
					}


					// 軸のサイズが適切かどうか判定
					for (Axis dA = 0; dA < NA; ++dA) {
						for (Axis dA_ = 0; dA_ < NA; ++dA_) {
							if (iiA[dA] == iiA[dA_]) {
								if (btA.blockAddIndices()[dA] != btA.blockAddIndices()[dA_]) {
									isValid_ = false;
								}
							}
						}
					}


					return isValid_;
				}


				BTR compute()const {
					const BTA& btA = std::get<0>(ts);
					const IIndicesA& iiA = std::get<0>(iiTs);


					if (!isValid()) {
						throw RuntimeException("invalid argument");
						return BTR();
					}


					std::array<AddIndices, NA> baiA;
					for (Axis d = 0; d < NA; ++d) {
						baiA[d] = btA.blockAddIndices()[d];
					}


					// 返すテンソルの軸の情報を取得

					std::array<AddIndices, NR> baiR;
					for (Axis d = 0; d < NR; ++d) {
						for (Axis da = 0; da < NA; ++da) {
							if (iiA[da] == iiR[d]) {
								baiR[d] = baiA[da];
								break;
							}
						}
					}

					std::array<Axis, NR> axR;
					for (Axis dR = 0; dR < NR; ++dR) {
						bool isFound = false;
						for (Axis dA = 0; dA < NA; ++dA) {
							if (isFound) {
								break;
							}
							if (iiA[dA] == iiR[dR]) {
								axR[dR] = dA;
								isFound = true;
							}
						}
					}

					std::array<Index, NR> dimsR;
					for (Axis d = 0; d < NR; ++d) {
						dimsR[d] = baiR[d].absoluteSize();
					}


					// 返すテンソルを計算
					BTR btR(baiR);
					for (auto& bA_ : btA.blocks()) {
						const std::array<BlockIndex, NA>& biA = bA_.first;
						const Eigen::Tensor<ScalarA, NA>& bA = bA_.second;

						bool canCalc = true;
						for (Axis dA = 0; dA < NA; ++dA) {
							for (Axis dA_ = 0; dA_ < NA; ++dA_) {
								if (iiA[dA] == iiA[dA_]) {
									if (biA[dA] != biA[dA_]) {
										canCalc = false;
										break;
									}
								}
							}
							if (!canCalc) {
								break;
							}
						}
						if (canCalc) {
							using BlockIndicesR = typename BTR::BlockIndices;
							BlockIndicesR biR;
							for (Axis dR = 0; dR < NR; ++dR) {
								biR[dR] = biA[axR[dR]];
							}
							btR.addBlock(
								biR,
								einsum(bA).from(iiA).to(iiR)
							);
						}

					}

					return btR;
				}


			};



		};







	}

		


}


