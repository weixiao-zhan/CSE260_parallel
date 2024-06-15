#!/bin/bash
#Generate data.txt
#May need to set execute permission: "chmod +x genDATA.sh"
# matrix_sizes=(32 64 128 256 511 512 513 1023 1024 1025 2047 2048)
matrix_sizes=(32 64 128 256 511 512 513 1023 1024 1025 2047 2048)

> data.txt
for size in "${matrix_sizes[@]}"
do
	printf 'Testing N=%d\t' $size
	SUM=0
	PEAK=0
	for i in {1..20}
	do
		OUTPUT=$(./benchmark-blislab -n $size -g)
		GFLOPS=$(echo $OUTPUT | awk '{print $2}')
		SUM=$(awk '{print $1+$2}' <<<"${SUM} ${GFLOPS}")
		# Update PEAK if the current GFLOPS is greater than the current PEAK
		PEAK=$(awk -v current=$GFLOPS -v peak=$PEAK 'BEGIN {print (current>peak)?current:peak}')
	done
	AVG=$(awk '{print $1/20}' <<<"${SUM}")
	printf 'AVG %s GFLOPS, PEAK %s GFLOPS\n' $AVG $PEAK
	printf '%d\t%s\t%s\n' $size $AVG $PEAK >> data.txt
done

printf 'Results saved in data.txt\n'