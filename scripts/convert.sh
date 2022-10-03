#!/bin/bash

if [ $# -ne 3 ]
then
	echo "Wrong number of arguments"
	echo "./convert.sh input_folder instrument.sh output_folder"
	exit 1
fi

for outer in $(ls $1)
do
	for inner in $(ls $1/$outer/*.mid)
	do
		echo $(basename $inner)
		filename=$(basename $inner | sed -E 's/\.mid$//').wav
		fluidsynth -a alsa -T raw -F - $2 $inner | ffmpeg -f s32le -i - $3/$filename
		#PID=$!
		#wait $PID
	done
done

