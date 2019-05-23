#!/bin/bash

#ROOT=/Users/kaifox/projects/art_physio
#GITHUB=/Users/kaifox/projects/dan_attfield_git/attfield2
#DRIVE=/Users/kaifox/GoogleDrive/attfield
ROOT=.
DRIVE=/Users/$(whoami)/gdrive/attfield

# Copy from local to google drive
cp -r $DRIVE/code $ROOT