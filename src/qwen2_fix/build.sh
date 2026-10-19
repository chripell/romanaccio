#!/bin/bash
set -e

sys_root_dir=null
demo_dir=$(pwd)
install_prefix=${demo_dir}/build/

function usage() {
    echo -e "Usage:\n$0 --sys_root=<sys_root_dir> --prefix=<install_prefix>"
}

function main() {
    # 解析参数
    for i in "$@"; do
        case $i in
            # sys root dir, default null
            --sys_root=*)
                sys_root_dir="${i#*=}"
                shift
                ;;
            # debug mode: 0(off), 1(on), default 0(off).
            --prefix=*)
                install_prefix="${i#*=}"
                shift
                ;;
            # print help info
            -h|--help)
                usage
                exit 0
                ;;
            # unknown option
            *)
                echo "Error: unsupported argument \"${i#*=}\""
                usage
                exit 1
                ;;
        esac
    done

    if [ ! -d build ]; then
        mkdir -p build
    else
        rm build/* -rf
    fi

    FLAGS="-O2"
    if [ -n "${sys_root_dir}" ] && [ "${sys_root_dir}" != "null" ]; then
        FLAGS="-O2 --sysroot=${sys_root_dir}"
        SYSROOT_ARG="-DHOST_SYSROOT=${sys_root_dir}"
    else
        SYSROOT_ARG="-DHOST_SYSROOT=/"
    fi

    cd build
    cmake .. \
          -DBUILD_USE_LOCAL_ENV=1 \
          ${SYSROOT_ARG} \
          -DHOST_C_COMPLIER="${CROSS_COMPILE:-}gcc" \
          -DHOST_CXX_COMPLIER="${CROSS_COMPILE:-}g++" \
          -DHOST_COMPLIER_FLAGS="${FLAGS}" \
          -DCMAKE_INSTALL_PREFIX=${install_prefix}
    make -j$(nproc 2>/dev/null || echo 4)
}

main $@
