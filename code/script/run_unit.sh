#!/bin/bash

TARGET=$1
TARGET_NAME=$(basename $TARGET)
TARGET_NAME=${TARGET_NAME%.*}

$py3 -m coverage run --include $TARGET \
	tests/unit/UNIT_${TARGET_NAME}.py
$py3 -m coverage html --directory data/unittest
$py3 -m coverage erase
