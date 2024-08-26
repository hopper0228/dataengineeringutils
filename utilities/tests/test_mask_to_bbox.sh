#!/bin/bash

directory_place=$1
recursion_time=$2

if ! [[ $recursion_time =~ ^[0-9]+$ ]]; then
    echo "Error: recursion_time must be a non-negative integer."
    exit 1
fi

cd "$directory_place" || { echo "Directory not found: $directory_place"; exit 1; }

if [ -f "build_here" ]; then
    rm -rf build_here
    echo deleted
fi

mkdir -p "build_here"
cd "build_here" || { echo "Failed to create or navigate to 'build_here'"; exit 1; }

# Define the recursive function
build_dir(){
    time=$1
    if (( time > $recursion_time )); then
        echo "done"
        exit 0
    else
        mkdir -p "a" "b"
        touch "image.png" "label.csv"
        cd "a" || { echo "Failed to navigate to 'a'"; exit 1; }
        build_dir $(( time + 1 ))
        cd .. || { echo "Failed to navigate back"; exit 1; }
    fi
}

build_dir 0
