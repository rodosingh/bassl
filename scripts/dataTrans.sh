seasons="S02 S03 S04 S05 S06 S07 S08 S09"
series="24"
# seasons="S02 S03"
# series="prison-break"
gn="79"


# TRANSFER from SHARE3 to GNODE SSD-SCRATCH
for season in $seasons; do
    for k in $(seq -f "%02g" 1 24); do
        i=$season"E"$k
        echo $i

        # make 240P_frames folder
        mkdir -p /ssd_scratch/cvit/rodosingh/data/${series}/movienet/240P_frames/$i/
        cp /ssd_scratch/cvit/rodosingh/data/${series}/$season/$i/shot_frames/* /ssd_scratch/cvit/rodosingh/data/${series}/movienet/240P_frames/$i/.
    done
done