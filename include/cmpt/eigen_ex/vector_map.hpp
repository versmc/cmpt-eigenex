#pragma once

#include <functional>


#include "Eigen/Core"
#include "cmpt/eigen_ex/util.hpp"


namespace cmpt {

  namespace EigenEx {

    /// <summary>
    /// exception class.
    /// this class is used in VectorMap
    /// </summary>
    class VectorMapException :public std::runtime_error {
    public:
      VectorMapException(const char* _Message)
        : std::runtime_error(_Message)
      {}
    };



    /// <summary>
    /// This class represents a mappings f: V^N -> V^M,
    /// or Eigen::Vector -> Eigen::Vector
    /// 
    /// operator overloads represet
    /// 
    /// (f+g)(x) = f(x)+g(x)
    /// (f*g)(x) = f(g(x))
    /// 
    /// </summary>
    template<class Scalar_>
    class VectorMap {
    public:
      using Scalar = Scalar_;
      using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
      using Index = Eigen::Index;
      using FunctionType = std::function<void(Scalar const*, Scalar*)>;
      using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
      using VectorType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;


    protected:
      FunctionType function_;
      Index sizeIn_;
      Index sizeOut_;

    public:
      


      VectorMap() :function_(), sizeIn_(0), sizeOut_(0) {}
      

      // アクセッサ
      const FunctionType& function()const { return function_; }
      Index sizeIn()const { return sizeIn_; }
      Index sizeOut()const { return sizeOut_; }

      VectorMap& setFromFunction(const FunctionType& func, Index size_in, Index size_out) {
        function_ = func;
        sizeIn_ = size_in;
        sizeOut_ = size_out;
        return *this;
      }


      /// <summary>
      /// 複数のVectorMapを合成して構築する
      /// 写像は begin から順に作用する
      /// </summary>
      VectorMap& setFromComposition(const std::vector<VectorMap>& vmaps) {

        // 次元チェック
        for (auto itr = vmaps.begin(); itr != vmaps.end(); ++itr) {
          auto jtr = itr;
          if (jtr == vmaps.end()) {
            continue;
          }
          else {
            if (itr->sizeOut() != jtr->sizeIn()) {
              throw VectorMapException("itr->sizeOut() != jtr->sizeIn()");
            }
          }
        }

        // set
        if (vmaps.size() == 0) {
          sizeIn_ = 0;
          sizeOut_ = 0;
        }
        else {
          sizeIn_ = vmaps.front().sizeIn();
          sizeOut_ = vmaps.back().sizeOut();
        }

        function_ = [vmaps](Scalar const* in, Scalar* out) {
          Index sin = vmaps.front().sizeIn();
          Index sout = vmaps.back().sizeOut();
          Eigen::Map<const VectorType> min(in, sin);
          Eigen::Map<VectorType> mout(out, sout);

          auto begin = vmaps.begin();
          auto end = vmaps.end();


          if (begin == end) {
            mout = min;
          }
          else {
            auto back = end;
            --back;
            if (begin == back) {
              mout = begin->makeOperated(min);
            }
            else {
              VectorType temp_out;
              VectorType temp_in;
              for (auto itr = begin; itr != end; ++itr) {
                if (itr == begin) {
                  temp_out = itr->makeOperated(min);
                  temp_out.swap(temp_in);
                }
                else if (itr == back) {
                  VectorType temp = itr->makeOperated(temp_in);
                  mout = temp;
                }
                else {
                  temp_out = itr->makeOperated(temp_in);
                  temp_out.swap(temp_in);
                }
              }
            }
          }
        };



        return *this;

      }

      /// <summary>
      /// 行列 A に対して
      /// f_A(x)=A*x
      /// で構築
      /// </summary>
      VectorMap& setFromMatrix(const MatrixType& mat) {
        setFromFunction(
          [mat](Scalar const* in, Scalar* out) {
            Eigen::Map<const VectorType> min(in, mat.cols());
            Eigen::Map<VectorType> mout(out, mat.rows());
            mout = mat * min;
          },
          mat.cols(),
            mat.rows()
            );
        return *this;
      }

      VectorType makeOperated(const VectorType& v_in)const {
        if (v_in.size() != sizeIn()) {
          throw VectorMapException("v_in.size()!=sizeIn()");
          return VectorType();
        }
        VectorType v_out(sizeOut());
        function()(v_in.data(), v_out.data());
        return v_out;
      }
      VectorType makeOperated(Eigen::Map<const VectorType> mv_in)const {
        if (mv_in.size() != sizeIn()) {
          throw VectorMapException("mv_in.size()!=sizeIn()");
          return VectorType();
        }
        VectorType v_out(sizeOut());
        function()(mv_in.data(), v_out.data());
        return v_out;
      }



      /// <summary>
      /// 写像後にスカラー倍の写像を追加する
      /// スカラーがゼロの時に限り節約的な特殊仕様をとる
      /// </summary>
      void scalarMultiple(const Scalar& c) {
        if (c == Scalar(0.0)) {
          Index so = sizeOut();
          *this = VectorMap().setFromFunction(
            [so](Scalar const* in, Scalar* out) {
              for (Index i = 0; i < so; ++i) {
                out[i] = Scalar(0.0);
              }
            },
            sizeIn(),
              sizeOut()
              );
        }
        else {
          Index so = sizeOut();
          auto vm_scalar = VectorMap().setFromFunction(
            [c, so](Scalar const* in, Scalar* out) {
              for (Index i = 0; i < so; ++i) {
                out[i] = c * in[i];
              }
            },
            sizeOut(),
              sizeOut()
              );
          *this = VectorMap().setFromComposition({ *this,vm_scalar });
        }
      }

      VectorMap scalarMultipled(const Scalar& c)const {
        VectorMap vm(*this);
        vm.scalarMultiple(c);
        return vm;
      }


      VectorMap operator+()const {
        return *this;
      }

      VectorMap operator-()const {
        return scalarMultipled(Scalar(-1.0));
      }

      VectorMap& operator+=(const VectorMap& other) {
        if (sizeIn() != other.sizeIn()) {
          throw VectorMapException("sizeIn()!=other.sizeIn()");
        }
        if (sizeOut() != other.sizeOut()) {
          throw VectorMapException("sizeOut()!=other.sizeOut()");
        }

        VectorMap tvm = *this;
        VectorMap ovm = other;

        function_ = [tvm, ovm](Scalar const* in, Scalar* out) {
          Eigen::Map<const VectorType> mvin(in, tvm.sizeIn());
          Eigen::Map<VectorType> mvout(out, tvm.sizeOut());
          VectorType tvout = tvm.makeOperated(mvin);
          VectorType ovout = ovm.makeOperated(mvin);
          mvout = tvout + ovout;
        };
        return *this;

      }

      VectorMap& operator-=(const VectorMap& other) {
        *this += (-other);
        return *this;
      }

      VectorMap& operator*=(const VectorMap& other) {
        *this = VectorMap().setFromComposition({ other,*this });
      }


    };

    // オペレータオーバーロード

    template<class Scalar_>
    VectorMap<Scalar_> operator+(const VectorMap<Scalar_>& a, const VectorMap<Scalar_>& b) {
      VectorMap<Scalar_> ret = a;
      ret += b;
      return ret;
    }

    template<class Scalar_>
    VectorMap<Scalar_> operator-(const VectorMap<Scalar_>& a, const VectorMap<Scalar_>& b) {
      VectorMap<Scalar_> ret = a;
      ret -= b;
      return ret;
    }

    template<class Scalar_>
    VectorMap<Scalar_> operator*(const VectorMap<Scalar_>& a, const VectorMap<Scalar_>& b) {
      VectorMap<Scalar_> ret = a;
      ret *= b;
      return ret;
    }


  }
  
}
