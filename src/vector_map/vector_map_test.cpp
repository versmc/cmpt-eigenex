#include <iostream>



#include <vector>
#include <array>
#include <type_traits>

#include "Eigen/Core"
#include "Eigen/CXX11/Tensor"


#include "cmpt/eigen_ex/vector_map.hpp"








namespace cmpt{
  namespace workspace{
    inline int main(){
      if(0){
        std::cout<<"hello"<<std::endl;
      }
      if(1){
        Eigen::VectorXd v=Eigen::VectorXd::Random(2);
        Eigen::MatrixXd A=Eigen::MatrixXd::Random(2,2);
        Eigen::VectorXd u0=A*v;
        Eigen::VectorXd u1=v;
        A.applyThisOnTheLeft(u1);
        Eigen::VectorXd u2=v;
        A.applyThisOnTheRight(u2);
        std::cout<<u0-u1<<std::endl;
        std::cout<<u0-u2<<std::endl;
      }
      if(0){
				using Scalar=double;
        using VectorType=Eigen::Matrix<Scalar,Eigen::Dynamic,1>;
        using MatrixType=Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>;

        MatrixType A=MatrixType::Random(2,2);

        EigenEx::VectorMap<Scalar> vmA=EigenEx::VectorMap<Scalar>().setFromMatrix(A);

        VectorType v=VectorType::Random(2);
        VectorType vo=A*v;







        VectorType vo_vmA=vmA.makeOperated(v);

        std::cout<<vo-vo_vmA<<std::endl;


				
			}
			return 0;
    }
  }
}

int main(){
  return cmpt::workspace::main();
}
