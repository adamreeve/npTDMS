#!/bin/bash
coverage run -m pytest && coverage html -d coverage
