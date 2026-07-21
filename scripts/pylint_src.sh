#!/bin/bash

echo "Running pylint locally on 'src/' source code directory."

echo ""
pylint --rcfile=./.pylintrc src/
echo ""

echo "Finished running pylint on 'src/' source code directory."
