#!/bin/bash

echo "Running script to clean up all ./**/__pycache__/ directories and ./**/.DS_Store files!"

echo ""
echo "Removing the following ./**/__pycache__/ directories from the local file system:"
echo "$(find src -name __pycache__)"
rm -fR $(find src -name __pycache__)
echo ""

echo "Removing the following ./**/.DS_Store files from the local file system:"
echo "$(find . -name .DS_Store)"
rm -f $(find . -name .DS_Store)
echo ""

echo "Finished running script to clean up all ./**/__pycache__/ directories and ./**/.DS_Store files!"
