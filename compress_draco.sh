#!/bin/bash
# TO RUN
#conda activate mesh_compress
#chmod +x compress_draco.sh
#./compress_draco.sh

mkdir -p draco_glbs

for file in glbs/*.glb
do
    name=$(basename "$file" .glb)

    echo "Compressing $file"

    gltf-transform draco \
        "$file" \
        "draco_glbs/${name}_draco.glb"

done