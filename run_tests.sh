#!/bin/bash

docker run -i -t --rm \
    -v $PWD:/nptdms:z \
    adreeve/python-numpy:latest \
    /bin/bash -c "
        cd /nptdms &&
        pep8 ./nptdms &&
        python2.7 setup.py install &&
        nosetests &&
        python3 setup.py install &&
        nosetests3"
