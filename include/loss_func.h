#pragma once

#include "all.h"

namespace YCDL {
    class loss_func_layer : public Layer {
    public:
        virtual double forward(double logit, int label) = 0;

        virtual double backward(double logit, int label) = 0;
    };

    class sigmoid_cross_entroy_with_logits : public loss_func_layer {
    public:
        double forward(double logit, int label) {
            double pre = sigmoid(logit);
            double loss = -(label * log(pre) + (1 - label) * log(1 - pre));
            return loss;
        }

        double backward(double logit, int label) {
            double pre = sigmoid(logit);
            double grad = pre - label;
            return grad;
        }
    };

    class mse : public loss_func_layer {
    public:
        double forward(double logit, int label) {
            double loss = (label - logit) * (label - logit) / 2;
            return loss;
        }

        double backward(double logit, int label) {
            double grad = logit - label;
            return grad;
        }
    };

    class sigmoid_mse : public loss_func_layer {
    public:
        double forward(double logit, int label) {
            double pre = sigmoid(logit);
            double loss = (label - pre) * (label - pre) / 2;
            return loss;
        }

        double backward(double logit, int label) {
            double pre = sigmoid(logit);
            double grad = (pre - label) * pre * (1 - pre);
            return grad;
        }
    };
}