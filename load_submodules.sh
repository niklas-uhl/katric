#!/bin/bash

git submodule update --init --recursive
cd extern/kagen/extlib/sampling/extlib/tlx/
git checkout master
