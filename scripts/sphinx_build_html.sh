#!/bin/bash

echo ""
echo 'Cleaning `docs/` sub-directory:'
echo ""

cd docs/
make clean 
rm -fR build/
cd ..

echo ""
echo 'Generating API (.rst) source files in `docs/source/` sub-directory:'
echo ""

sphinx-apidoc --output-dir "docs/source" -e -f "src/"
#sphinx-apidoc --output-dir "docs/source" -e -f "tests/"

echo ""
echo 'Building (.html) API documentation files with sphinx in `docs/build/` sub-directory:'
echo ""

sphinx-build -M coverage
sphinx-build -M html "docs/source" "docs/build" 
