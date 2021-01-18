set -e
set -o pipefail

DIR=`pwd`
export CPPFLAGS=-I${DIR}/thirdparty/ps-lite/deps/include
export LDFLAGS=-L${DIR}/thirdparty/ps-lite/deps/lib
export PATH=${DIR}/thirdparty/ps-lite/deps/bin:$PATH
export LD_LIBRARY_PATH=${DIR}/thirdparty/ps-lite/deps/lib:$LD_LIBRARY_PATH

function build_all() {
    cd ./thirdparty/ps-lite
    make
    cd -
    mkdir -p build
    cd ./build/ 
    cmake ..
    make 
    cd .. 
}

function test_all() {
    echo "test begin"
    ls ./bin/test | while read file
    do 
        ./bin/test/$file > /dev/null
        echo "test $file done"
    done
    echo "test all done"
}

function run_all() {
    build_all 
    test_all
}

function main() {
    local cmd=${1:-help}
    case ${cmd} in
        all)
            run_all "$@"
            ;;
        build)
            build_all "$@"
            ;;
        test)
            test_all "$@"
            ;;
        help)
            echo "Usage: ${BASH_SOURCE} {test}"
            return 0;
            ;;
        *) 
            echo "unsupport command [${cmd}]"
            echo "Usage: ${BASH_SOURCE} {test}"
            return 1;
            ;;
    esac
            
}

main "$@"
