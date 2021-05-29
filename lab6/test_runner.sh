#!/bin/bash

# Usage: ./test_runner.sh myPythonScript.py

python3 $1 -svd custom -k 5 -f img/pink-floyd.jpg -out img/pink-floyd_testedcustom.jpg
python3 $1 -svd library -k 5 -f img/pink-floyd.jpg  -out img/pink-floyd_testedlib.jpg
python3 $1 -svd custom -k 30 -f img/pink-floyd.jpg -out img/pink-floyd_testedcustomHigh.jpg
python3 $1 -svd custom -k 30 -f img/wittgenstein.jpg -out img/wittgenstein_testedcustomHigh.jpg
python3 $1 -svd custom -k 70 -f img/clouds.jpg

