#include <iostream>
#include <Eigen/Dense>
#include "eigen_func.h"
int main()
{
    Eigen::MatrixXf m(2,2), c(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;

    YCDL::matrix_multi(false, false, false, m, m, c);
    std::cout << c << std::endl;

    YCDL::matrix_multi_addition(false, false, false, m, m, c);
    std::cout << c << std::endl;
}