#!/bin/sh

# This scripts downloads the DAVIS data and unzip it.
# Reference: https://github.com/davisvideochallenge/davis-2017/blob/master/data/get_davis.sh

URL=https://data.vision.ee.ethz.ch/jpont/davis

FILE_TRAINVAL=DAVIS-2017-trainval-480p.zip

if [ ! -f $FILE_TRAINVAL ]; then
  echo "Downloading DAVIS 2017 (train-val)..."
  wget $URL/$FILE_TRAINVAL
else
	echo "File $FILE_TRAINVAL already exists. Checking md5..."
fi

unzip -o $FILE_TRAINVAL

rm -rf $FILE_TRAINVAL

#Fix the '/data/DAVIS' to the corret path
mv DAVIS /data/DAVIS