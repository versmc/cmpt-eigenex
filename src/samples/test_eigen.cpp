#include <iostream>
#include "Eigen/Core"
#include "Eigen/CXX11/Tensor"

int main(){
  Eigen::MatrixXd A(2,2);
  A<<1.0,2.0,3.0,4.0;
  std::cout<<A<<std::endl;

  auto t3=Eigen::Tensor<double,3>(2,2,2).setValues({{{0,1},{2,3}},{{4,5},{6,7}}});
  std::cout<<t3<<std::endl;

  return 0;
}
