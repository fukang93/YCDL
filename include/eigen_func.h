#pragma once
#include <Eigen/Dense>

namespace YCDL {
    template<class A, class B, class C>
    void matrix_multi(bool transa, bool transb, bool transc, const A &a, const B &b, C &c) {
        if (transc) {
            return matrix_multi(!transb, !transa, false, b, a, c);
        }
        if (!transa) {
            if (!transb) {
                assert(a.cols() == b.rows());
                c.noalias() = a * b;
            } else {
                assert(a.cols() == b.cols());
                c.noalias() = a * b.transpose();
            }
        } else {
            if (!transb) {
                assert(a.rows() == b.rows());
                c.noalias() = a.transpose() * b;
            } else {
                assert(a.rows() == b.cols());
                c.noalias() = a.transpose() * b.transpose();
            }
        }
    }

    template<class A, class B, class C>
    void matrix_multi_addition(bool transa, bool transb, bool transc, const A &a, const B &b, C &c) {
        if (transc) {
            return matrix_multi_addition(!transb, !transa, false, b, a, c);
        }
        if (!transa) {
            if (!transb) {
                assert(a.cols() == b.rows());
                c.noalias() += a * b;
            } else {
                assert(a.cols() == b.cols());
                c.noalias() += a * b.transpose();
            }
        } else {
            if (!transb) {
                assert(a.rows() == b.rows());
                c.noalias() += a.transpose() * b;
            } else {
                assert(a.rows() == b.cols());
                c.noalias() += a.transpose() * b.transpose();
            }
        }
    }
}