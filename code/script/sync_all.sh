ROOT=/Users/kaifox/projects/art_physio
GITHUB=/Users/kaifox/projects/dan_attfield_git/attfield2
DRIVE=/Users/kaifox/GoogleDrive/attfield

# Copy from local to google drive
cp -r $ROOT/code $DRIVE/code

# Sync required folders to github
cp -R -P -p $ROOT/code $GITHUB/.
cp -R -P -p $ROOT/archive/scripts $GITHUB/.
cp -R -P -p $ROOT/tests $GITHUB/.
vim $ROOT/msg_commit.txt
cd $GITHUB
git add code archive/scripts tests
git commit -F $ROOT/msg_commit.txt
cd -