#pragma once


#include <complex>
#include <array>
#include <vector>
#include <functional>
#include <algorithm>

#include "Eigen/Core"
#include "Eigen/Sparse"




namespace cmpt {
	namespace EigenEx {
		/// <summary>
		/// This class represents Matrix with [(row,col,value)] triplets
		/// 
		/// This class has Eigen-like members
		/// 
		/// * TOTO
		/// operator* adaption
		/// </summary>
		template<class Scalar_>
		class TripletsMatrix {
		public:
			// type aliases

			using Scalar = Scalar_;
			using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
			using Index = int;
			using Triplet = Eigen::Triplet<Scalar>;
			using Triplets = std::vector<Eigen::Triplet<Scalar>>;

			using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
			using RealVectorType = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;
			using DenseMatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
			using SparseMatrixType = Eigen::SparseMatrix<Scalar>;
			using MatMulFunction = std::function<void(Scalar const*, Scalar*)>;

			using iterator = typename Triplets::iterator;
			using const_iterator = typename Triplets::const_iterator;



		protected:
			Index rows_;
			Index cols_;
			Triplets triplets_;

		public:
			// accessor

			Index rows()const { return rows_; }
			Index cols()const { return cols_; }



			const Triplets& triplets()const { return triplets_; }
			Triplets& ref_triplets() { return triplets_; }
			TripletsMatrix& setTriplets(const Triplets& triplets_a) { triplets_ = triplets_a; return *this; }


			iterator begin() { return triplets_.begin(); }
			const_iterator begin()const { return triplets_.begin(); }
			iterator end() { return triplets_.end(); }
			const_iterator end()const { return triplets_.end(); }


			/// <summary>
			/// check whether this->triplets_ is invalid against the this->rows_ or this->cols_
			/// </summary>
			bool rangeIsInvalid() {
				bool valid = std::all_of(
					begin(),
					end(),
					[this](const Triplet& tri)->bool {
						return tri.row() < rows_ && tri.col() < cols_;
					}
				);
				return !valid;
			}


			TripletsMatrix& clear() {
				rows_ = 0;
				cols_ = 0;
				Triplets().swap(triplets_);
				return *this;
			}


			/// <summary>
			/// change the shape of matrix. then only in-range triplets will survive
			/// </summary>
			TripletsMatrix& resize(Index rows, Index cols, bool clear_elements = false) {
				rows_ = rows;
				cols_ = cols;
				if (clear_elements) {
					triplets_.clear();
				}
				else {
					auto eitr = std::remove_if(
						begin(),
						end(),
						[this](const Triplet& tri)->bool {
							return (tri.row() < rows_ && tri.col() < cols_);
						}
					);
					triplets_.erase(eitr, triplets_.end());
				}
				return *this;
			}

			/// <summary>
			/// change the shape of matrix to the minimul shape for current elements in triplets_
			/// </summary>
			TripletsMatrix& fitSize() {
				Index rows = 0;
				Index cols = 0;
				for (const auto& tri : triplets_) {
					Index r = tri.row();
					Index c = tri.col();
					if (rows <= r) {
						rows = r + 1;
					}
					if (cols <= c) {
						cols = c + 1;
					}
				}
				rows_ = rows;
				cols_ = cols;
				return *this;
			}


			TripletsMatrix& appendTriplet(const Triplet& tri) {
				triplets_.push_back(tri);
				return *this;
			}

			TripletsMatrix& appendTriplet(Index row, Index col, Scalar value) {
				return appendTriplet(Triplet(row, col, value));
			}

			TripletsMatrix& setFromTriplets(const Triplets& triplets, Index rows, Index cols) {
				clear();
				resize(rows, cols);
				for (auto& tri : triplets) {
					appendTriplet(tri);
				}
				return *this;
			}

			TripletsMatrix& setFromDenseMatrix(const DenseMatrixType& mat, bool ignore_zero = true) {
				clear();
				resize(mat.rows(), mat.cols());
				for (Index c = 0, nc = mat.cols(); c < nc; ++c) {
					for (Index r = 0, nr = mat.rows(); r < nr; ++r) {
						Scalar v = mat(r, c);
						if (ignore_zero && v == Scalar(0.0)) {
							break;
						}
						appendTriplet(r, c, v);
					}
				}
				return *this;
			}




			TripletsMatrix& setZero() {
				triplets_.swap(Triplets());
				return *this;
			}

			TripletsMatrix& setIdentity() {
				setZero();
				Index ni = min_value(rows(), cols());
				triplets_.reserve(ni);
				for (Index i = 0; i < ni; ++i) {
					triplets_.push_back(Triplet(i, i, Scalar(1.0)));
				}
				return *this;
			}


			/// <summary>
			/// the default definition to sort triplets. column-major(==fortran-like)
			/// </summary>
			static bool less_than_for_sort_default(const Triplet& lhs, const Triplet& rhs) {
				if (lhs.col() < rhs.col()) {
					return true;
				}
				else if (rhs.col() < lhs.col()) {
					return false;
				}
				else {
					if (lhs.row() < rhs.row()) {
						return true;
					}
					else {
						return false;
					}
				}
			}


			/// <summary>
			/// sort triplets inplace. the default predicate is column-major(==fortran-like) order
			/// </summary>
			TripletsMatrix& sort(
				const std::function<bool(const Triplet&, const Triplet&)> pred = less_than_for_sort_default
			) {
				std::sort(triplets_.begin(), triplets_.end(), pred);
				return *this;
			}

			/// <summary>
			/// return sorted object, current object keeps constant
			/// </summary>
			TripletsMatrix sorted()const {
				return TripletsMatrix(*this).sort();
			}

			void eraseIf(const std::function<bool(const Triplet&)>& pred) {
				auto itr = std::remove_if(triplets_.begin(), triplets_.end(), pred);
				triplets_.erase(itr, triplets_.end());
			}


			/// <summary>
			/// 1. sort, 2. add same elements, 3. erase zero term.
			/// </summary>
			TripletsMatrix& shrink(RealScalar threshold = 0.0) {
				sort();

				if (triplets_.empty()) {
					return *this;
				}
				Triplets tri_new;

				auto itr1 = triplets_.begin();
				auto itr2 = itr1;
				Scalar c = itr2->value();
				++itr2;
				while (true) {
					if (
						(itr2 != triplets_.end())
						&&
						(itr1->row() == itr2->row())
						&&
						(itr1->col() == itr2->col())
						) {
						c += itr2->value();
					}
					else {
						if (c != Scalar(0.0)) {
							tri_new.push_back(Triplet(itr1->row(), itr1->col(), c));
						}
						if (itr2 == triplets_.end()) {
							break;
						}
						else {
							c = itr2->value();
						}
					}

					itr1 = itr2;
					++itr2;
				}
				triplets_ = std::move(tri_new);

				eraseIf(
					[threshold](const Triplet& tri)->bool {
						return std::abs(tri.value()) < threshold;
					}
				);
				return *this;
			}

			TripletsMatrix shrinked()const {
				return TripletsMatrix(*this).shrink();
			}


		public:
			// constructors

			TripletsMatrix() :rows_(0), cols_(0), triplets_() {}

			TripletsMatrix(Index rows, Index cols, const Triplets& triplets = Triplets()) {
				resize(rows, cols);
				triplets_ = triplets;
			}

			TripletsMatrix(const Triplets& triplets) {
				triplets_ = triplets;
				fitSize();
			}




		public:
			// mathematics

			/// <summary>
			/// add matrix-producted "in" to "out"
			/// </summary>
			void addOperatedVector(Scalar const* in, Scalar* out)const {
				for (auto& tri : triplets_) {
					out[tri.row()] += in[tri.col()] * tri.value();
				}
			}


			/// <summary>
			/// substitue matrix-operated "in" for "out" 
			/// </summary>
			void operate(Scalar const* in, Scalar* out)const {
				for (Index r = 0, nr = rows(); r < nr; ++r) {
					out[r] = Scalar(0.0);
				}
				addOperatedVector(in, out);
			}



			/// <summary>
			/// add matrix-operated "in" to "out" 
			/// </summary>
			void addOperatedVector(const VectorType& in, VectorType& out)const {
				add_operated_vector(in.data(), out.data());
			}

			/// <summary>
			/// substitue matrix-operated "in" for "out" 
			/// </summary>
			void operate(const VectorType& in, VectorType& out)const {
				operate(in.data(), out.data());
			}

			/// <summary>
			/// return matrix-operated "in"
			/// </summary>
			VectorType makeOperated(const VectorType& in)const {
				VectorType out(rows());
				operate(in, out);
				return out;
			}

			/// <summary>
			/// return matrix-operated "in"
			/// </summary>
			DenseMatrixType makeOperated(const DenseMatrixType& in)const {
				DenseMatrixType m_out(rows(), in.cols());
				for (Index c = 0, nc = in.cols(); c < nc; ++c) {
					VectorType v_in = in.col(c);
					m_out.col(c) = makeOperated(v_in);
				}
				return m_out;
			}


			/// <summary>
			/// returns the function object which represent the matrix-vector-multiplication.
			/// the function object has the copy-captured *this
			/// </summary>
			MatMulFunction makeMatMulFunction()const {
				TripletsMatrix m(*this);
				return MatMulFunction(
					[m](Scalar const* in, Scalar* out) {
						m.operate(in, out);
					}
				);
			}


			/// <summary>
			/// return transposed object. this object keeps constant
			/// </summary>
			TripletsMatrix transpose()const {
				TripletsMatrix triop;
				triop.resize(cols(), rows());

				for (auto& tri : triplets_) {
					triop.triplets_.push_back(
						Triplet(
							tri.col(),
							tri.row(),
							tri.value()
						)
					);
				}
				return triop;
			}


			/// <summary>
			/// return adjoint(conjugate and transpose) object. this object keeps constant
			/// </summary>
			TripletsMatrix adjoint()const {
				TripletsMatrix triop;
				triop.resize(cols(), rows());

				for (auto& tri : triplets_) {
					triop.triplets_.push_back(
						Triplet(
							tri.col(),
							tri.row(),
							std::conj(tri.value())
						)
					);
				}
				return triop;
			}


			TripletsMatrix& scalarMultiple(const Scalar& c) {
				for (auto& tri : triplets_) {
					tri = Eigen::Triplet<Scalar>(tri.row(), tri.col(), tri.value() * c);
				}
				return *this;
			}
			TripletsMatrix scalarMultipled(const Scalar& c)const {
				TripletsMatrix trimat(*this);
				return trimat.scalarMultiple(c);
			}



			DenseMatrixType makeDenseMatrix()const {
				Eigen::MatrixXcd mat = Eigen::MatrixXcd::Zero(rows(), cols());
				for (auto& tri : triplets_) {
					mat(tri.row(), tri.col()) += tri.value();
				}
				return mat;
			}


			SparseMatrixType makeSparseMatrix()const {
				SparseMatrixType m(rows(), cols());
				m.setFromTriplets(begin(), end());
				return m;
			}


			RealScalar l1norm()const {
				RealScalar l1norm = 0.0;
				for (auto& tri : triplets_) {
					l1norm += std::abs(tri.value());
				}
				return l1norm;
			}

			RealScalar l2norm()const {
				RealScalar l2norm = 0.0;
				for (auto& tri : triplets_) {
					l2norm += std::norm(tri.value());
				}
				return l2norm;
			}

			/// <summary>
			/// returns L^\infinity norm
			/// </summary>
			/// <returns></returns>
			RealScalar linorm()const {
				RealScalar li = 0.0;
				for (auto& tri : triplets_) {
					RealScalar nv = std::norm(tri.value());
					if (li < nv) {
						li = nv;
					}
				}
				return li;
			}

			/// <summary>
			/// compute Gershgorin discs and return it
			/// </summary>
			std::vector<std::pair<Scalar, RealScalar>> makeGershgorinDiscs()const {
				std::vector<std::pair<Scalar, RealScalar>> discs;
				discs.resize(rows());
				for (Index i = 0, n = rows(); i < n; ++i) {
					discs[i].first = Scalar(0.0);
					discs[i].second = RealScalar(0.0);
				}
				for (auto& tri : triplets_) {
					Index r = tri.row();
					Index c = tri.col();
					Scalar v = tri.value();
					if (r == c) {
						discs[r].first += v;
					}
					else {
						discs[r].second += std::abs(v);
					}
				}
				return discs;
			}



			/// <summary>
			/// compute lower-bound and upper-bound of eigenvalues, using Gershgorin theorem
			/// </summary>
			std::array<RealScalar, 2> estimateEigenvalueRange()const {
				std::vector<std::pair<Scalar, RealScalar>> discs = makeGershgorinDiscs();
				RealScalar min = std::numeric_limits<RealScalar>::max();
				RealScalar max = std::numeric_limits<RealScalar>::min();
				for (const auto& disc : discs) {
					RealScalar dmin = std::real(disc.first - disc.second);
					RealScalar dmax = std::real(disc.first + disc.second);
					if (dmin < min) { min = dmin; }
					if (dmax > max) { max = dmax; }
				}
				return std::array<RealScalar, 2>{min, max};
			}


		public:
			// arithmetic operators

			TripletsMatrix operator+()const {
				return *this;
			}

			TripletsMatrix operator-()const {
				TripletsMatrix trimat(*this);
				for (auto& tri : trimat.triplets_) {
					tri = Triplet(tri.row(), tri.col(), -tri.value());
				}
				return trimat;
			}


			TripletsMatrix& operator+=(const Triplet& triplet) {
				return appendTriplet(triplet);
			}

			TripletsMatrix& operator+=(const TripletsMatrix& other) {
				triplets_.insert(triplets_.end(), other.triplets_.begin(), other.triplets_.end());
				return *this;
			}


			TripletsMatrix& operator-=(const Triplet& triplet) {
				triplets_.push_back(TripletType(triplet.row(), triplet.col(), -triplet.value()));
				return *this;
			}

			TripletsMatrix& operator-=(const TripletsMatrix& other) {
				*this += (-other);
				return *this;
			}



		};

		template<class Scalar>
		inline TripletsMatrix<Scalar> operator+(const TripletsMatrix<Scalar>& a, const TripletsMatrix<Scalar>& b) {
			TripletsMatrix<Scalar> ret = a;
			ret += b;
			return ret;
		}

		template<class Scalar>
		inline TripletsMatrix<Scalar> operator-(const TripletsMatrix<Scalar>& a, const TripletsMatrix<Scalar>& b) {
			TripletsMatrix<Scalar> ret = a;
			ret -= b;
			return ret;
		}

	}
}


