#include <iostream>



#include "cmpt/eigen_ex/util.hpp"



namespace cmpt{
  namespace workspace{
    inline int main(){
      // hello
      if(1){
        std::cout<<"hello"<<std::endl;
      }

      // Generate
      if(1){
        Eigen::MatrixXd A=EigenEx::Generate<Eigen::MatrixXd>::from({{0.0,1.0},{2.0,3.0}});
        std::cout<<"Test EigenEx::Generate"<<std::endl;
        std::cout<<"A:\n"<<A<<std::endl;
      }


      return 0;
    }
  }
}

int main(){
  return cmpt::workspace::main();
}
