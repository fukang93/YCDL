#include <generator.h>
#include <iostream>

YCDL::Generator<int> range(int max){
  return YCDL::Generator<int>([=](YCDL::Yield<int> &yield){
    for(int i = 0;i<max;++i) yield(i);
  });
}

int main(){
  for(int i:range(10)) std::cout << i << std::endl;
}

