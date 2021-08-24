# Normalize paths
from_loc=$(realpath -s $1)
from_loc=${from_loc%/}
to_loc=$(realpath -s $2)
to_loc=${to_loc%/}

# Figure out where the script is to run basepath_sync.sh
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Main line to watch for file system events
fswatch -0 -xr $from_loc | xargs -0 -n 1 -I {} bash $script_dir/basepath_sync.sh $from_loc $to_loc {}