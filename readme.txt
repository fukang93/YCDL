1.基于pslite 的DNN框架

2.依赖的库有：
  boost
  Eigen
  ps-lite
  generator：https://github.com/TheLartians/Generator
  json: https://github.com/nlohmann/json.git
  
3. 依赖库安装
 ./run_build.sh
 
4. 编译
./run.sh build

5. test
./run.sh test 

6. 运行demo
单机LR ./bin/output/lr_uci
单机DNN ./bin/output/network
伪分布式LR sh script/local.sh 2 2 ./bin/output/lr_uci_dist
伪分布式DNN sh script/local.sh 2 2 ./bin/output/network_dist
