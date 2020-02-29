#!/bin/bash

docker run -i -t --rm \
    -v $PWD:/nptdms:z \
    adreeve/python-numpy:latest \
    /bin/bash -c "
        export LC_ALL=C.UTF-8 &&
        cd /nptdms &&
        pep8 ./nptdms &&
        pip install pytest &&
        pip install .[hdf,pandas,thermocouple_scaling] &&
        python -m pytest &&
        pip3 install pytest &&
        pip3 install .[hdf,pandas,thermocouple_scaling] &&
        python3 -m pytest"
