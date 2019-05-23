# Source me to load the software environment on Cori

# This is a fix for module load paths on our jupyter node
if [[ $NERSC_HOST == edison || $NERSC_HOST == cori ]]; then
    if [[ $MODULEPATH != */usr/common* ]]; then
        match=/opt/modulefiles
        nersc_modulepaths=/usr/common/software/modulefiles:/usr/syscom/nsg/modulefiles:/usr/common/das/modulefiles:/usr/common/ftg/modulefiles:/usr/common/graphics/modulefiles:/usr/common/jgi/modulefiles:/usr/common/tig/modulefiles
        export MODULEPATH=$MODULEPATH:$match:$nersc_modulepaths
    fi
fi

export HDF5_USE_FILE_LOCKING=FALSE

