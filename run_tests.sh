#!/bin/bash

docker run -i -t --rm \
    -v $PWD:/nptdms:z \
    adreeve/python-numpy:latest \
    /bin/bash -c "
        cd /nptdms &&
        pep8 ./nptdms &&
        pip install .[hdf,pandas,thermocouple_scaling] &&
        nosetests &&
        pip3 install .[hdf,pandas] &&
        nosetests3"
