 -- trax --
 
 
 -- External Dependencies --
 
 * Recent Boost library, at least version 1.51. ( taken from CERN  AFS, if availble )
 * Google Protofbuf Library ( take  from AFS , if available )
 * OpenCL SDK ( either from Indtel or  AMD)
 
 -- Checkout --

To clone the repository and the contained submodule:

git clone https://github.com/ThomasHauth/trax.git
cd trax
git submodule init
git submodule update 
    
to get the correct revision of the external repo.

DO NOT commit your locally changed ".gitmodules" file to the central repo !

-- Compiling -- 

to compile an CERN / SLC machines, do the following:

starting in the "trax" root folder with the bash:

export TRAX_DIR=<full path to the trax folder>
export LD_LIBRARY_PATH=/afs/cern.ch/cms/slc6_amd64_gcc472/external/gcc/4.7.2/lib64/:/afs/cern.ch/cms/slc6_amd64_gcc472/external/gcc/4.7.2/lib/
export CXX=/afs/cern.ch/cms/slc6_amd64_gcc472/external/gcc/4.7.2/bin/g++
mkdir build
cd build
cmake ../src/
make -j8

to run the tests:
trax_test/trax_test

-- Protobuf --

on SLC: 
If your protobuffer library is not recent enough, download from the web and use this command to load the new lib
export LD_LIBRARY_PATH=/build/hauth/dev/protobuf-bin/lib/
