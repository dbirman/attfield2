from_loc=$1
to_loc=$2

# Isolate flags of the filesystem event
fs_args=($3)
eflags=${fs_args[@]:1}
modified_path=${fs_args[0]}

if [[ $to_loc == *:* ]]; then
	TO_LOC_IS_REMOTE=1
else
	TO_LOC_IS_REMOTE=0
fi


# If the event was a removal, sync the whole directory to perform the removal
if [[ " ${eflags[@]} " =~ " Removed " ]]; then
	modified_dir=$(dirname $modified_path)
	modified_dir=${modified_dir%/}
	modified_dir_rel=`realpath -s --relative-to="$from_loc" $modified_dir`
	if [[ $TO_LOC_IS_REMOTE == 1 ]]
		then to_loc_rel=$to_loc/$modified_dir_rel
		else to_loc_rel=$(realpath -s $to_loc/$modified_dir_rel)
	fi
	echo "U[R]:" "$to_loc/$modified_dir_rel"
	cmd="rsync -hvrPt --delete '$modified_dir/' $to_loc_rel \
		--exclude '*/__pycache__'"
	echo $cmd
	$cmd

# If there was no removal just sync the edited file
else
	modified_dir_rel=`realpath -s --relative-to="$from_loc" $modified_path`
	echo "U[E]:" "$to_loc/$modified_dir_rel"
	if [[ $TO_LOC_IS_REMOTE == 1 ]]; then
		to_loc_rel=$to_loc/$modified_dir_rel
	else
		to_loc_rel=$(realpath -s $to_loc/$modified_dir_rel)
	fi
	cmd="rsync -hvrPt $modified_path $to_loc_rel \
		--exclude '*/__pycache__'"
	echo $cmd
	$cmd
fi
