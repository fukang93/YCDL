#pragma once
#include "all.h"
#include "ioc.h"
#include "loss_func.h"
#include "optimizer.h"
#include "dist_optimizer.h"
#include "dense_layer.h"

namespace YCDL {
    void initialize() {
        RegisterLayer<loss_func_layer, sigmoid_cross_entroy_with_logits>("sigmoid_cross_entroy_with_logits");
        RegisterLayer<loss_func_layer, mse>("mse");
        RegisterLayer<loss_func_layer, sigmoid_mse>("sigmoid_mse");
        RegisterLayer<Optimizer, SGD>("SGD");
        RegisterLayer<Optimizer, Adagrad>("Adagrad");
        RegisterLayer<Optimizer, DistWorker>("DistWorker");
        RegisterLayer<NNLayer, Dense>("Dense");
        RegisterLayer<NNLayer, Loss>("Loss");
        RegisterLayer<NNLayer, Activation>("Activation");
        RegisterLayer2<Layer, AdagradServer>("DistAdagrad");
        RegisterLayer2<Layer, SGDServer>("DistSGD");
    }
}
