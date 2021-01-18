set -eux
DIR=`pwd`

## install protobuf zmq 
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.5.1/protobuf-cpp-3.5.1.tar.gz 
wget https://raw.githubusercontent.com/mli/deps/master/build/zeromq-4.1.4.tar.gz

## install eigen boost json ps-lite
mkdir -p ./thirdparty/ 
wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
tar -zxvf ./eigen-3.3.7.tar.gz 
mv eigen-3.3.7 ./thirdparty/ 

wget https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.gz
tar zxvf boost_1_64_0.tar.gz
cd boost_1_64_0
    ./bootstrap.sh --with-libraries=all --with-toolset=gcc
    ./b2 toolset=gcc
    ./b2 install --prefix=${DIR}/thirdparty/boost_configure 
cd - 

cd ./thirdparty/ 
git clone https://github.com/nlohmann/json.git 
cd -

cd ./thirdparty/ 
git clone https://github.com/dmlc/ps-lite.git 
cd 
