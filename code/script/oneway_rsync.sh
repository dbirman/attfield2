# Normalize paths
from_loc=$(realpath -s $1)
from_loc=${from_loc%/}
if [[ $2 == *:* ]]; then
	echo "Treating $2 as a remote desination."
	to_loc=${2%/}
else
	echo "Treating $2 as a local desination."
	to_loc=$(realpath -s $2)
	to_loc=${to_loc%/}
fi

# Figure out where the script is to run basepath_sync.sh
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Run a full backup in case things were changed since last time we ran
bash $script_dir/basepath_sync.sh $from_loc $to_loc $from_loc

# Main line to watch for file system events
fswatch -0 -xr $from_loc | xargs -0 -n 1 -I {} bash $script_dir/basepath_sync.sh $from_loc $to_loc {}