#!/bin/bash

files=(2011_09_26_calib.zip
2011_09_26_drive_0002
2011_09_26_drive_0009
2011_09_26_drive_0013
2011_09_26_drive_0020
2011_09_26_drive_0023
2011_09_26_drive_0027
2011_09_26_drive_0029
2011_09_26_drive_0036
2011_09_26_drive_0046
2011_09_26_drive_0048
2011_09_26_drive_0052
2011_09_26_drive_0056
2011_09_26_drive_0059
2011_09_26_drive_0064
2011_09_26_drive_0084
2011_09_26_drive_0086
2011_09_26_drive_0093
2011_09_26_drive_0096
2011_09_26_drive_0101
2011_09_26_drive_0106
2011_09_26_drive_0117
2011_09_28_calib.zip
2011_09_28_drive_0002
2011_09_29_calib.zip
2011_09_29_drive_0071
2011_09_30_calib.zip
2011_09_30_drive_0016
2011_09_30_drive_0018
2011_09_30_drive_0027
2011_10_03_calib.zip
2011_10_03_drive_0027
2011_10_03_drive_0047)

for i in ${files[@]}; do
        if [ ${i:(-3)} != "zip" ]
        then
                shortname=$i'_sync.zip'
                fullname=$i'/'$i'_sync.zip'
        else
                shortname=$i
                fullname=$i
        fi
	echo "Downloading: "$shortname
        wget 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'$fullname
        unzip -o $shortname
        rm $shortname
done