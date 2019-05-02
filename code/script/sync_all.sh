#!/bin/bash

ROOT=/Users/kaifox/projects/art_physio
GITHUB=/Users/kaifox/projects/dan_attfield_git/attfield2
DRIVE=/Users/kaifox/GoogleDrive/attfield

# Copy from local to google drive
cp -r $ROOT/code $DRIVE

# Sync required folders to github
cp -R -P -p $ROOT/code $GITHUB
cp -R -P -p $ROOT/archive/scripts $GITHUB/archive
cp -R -P -p $ROOT/tests $GITHUB
vim $ROOT/msg_commit.txt
cd $GITHUB
git add code archive/scripts tests

# Allow user to back out if it looks like things went wrong
echo "No fatal errors. Continue to commit?"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) git commit -F $ROOT/msg_commit.txt; break;;
        No ) exit;;
    esac
done

# Allow user (another chance) to back out if it looks like things went wrong
echo "No fatal errors. Push commit to GitHub?"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) git push origin master; break;;
        No ) exit;;
    esac
done

cd -