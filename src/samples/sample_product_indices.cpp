//
// This code is a sample for cmpt::EigenEx::ProductIndices
//

#include <iostream>
#include "cmpt/eigen_ex/multi_indices.hpp"

using namespace cmpt;

int main(){
  using Index=EigenEx::integer_type::Index; // == std::ptrdiff_t
  using Axis=EigenEx::integer_type::Axis;   // == std::size_t


  EigenEx::ProductIndices<3> idx(2,3,2);  // dense index-indices correspondence

  std::cout<<"index correspondence of idx\n";
  for(Index i=0,ni=idx.absoluteSize();i<ni;++i){
    std::array<Index,3> indices=idx.indices(i);
    std::cout<< i<<"    ";
    for(Axis d=0;d<idx.numDimensions();++d){
      std::cout<<indices[d]<<" ";
    }
    std::cout<<"   "<<idx.absoluteIndex(indices);
    std::cout<<"\n";
  }
  std::cout<<"\n";

  EigenEx::ProductIndices<2> idx2=idx.from({"i","j","i"}).to({"i","j"});  // index iji->ij (i is diagonal index)
  std::cout<<"index correspondence of idx2\n";
  
  for (auto&& absi : idx2.arrangeAbsoluteIndexList()) {
    auto indices = idx2.indices(absi);
    std::cout << absi << "   ";
    for (auto& i_ : indices) {
      std::cout << i_ << " ";
    }
    std::cout << std::endl;
  }
  

	return 0;
}
