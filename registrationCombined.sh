#!/bin/bash
# Script to test segmentation of Hyperfine scans (as well as scans from other scanners).
# Levente Baljer (k2035837@kcl.ac.uk)

declare -a subsTest=("2" "4" "5" "17" "47" "48" "62" "63" "70" "76" "77" "83" "88" "89" "94" "97" "117" "121" "146" "150" "167" "176" "181" "190" "201" "211" "212" "213" "217" "247" "248" "258" "259" "260" "266" "267" "270" "279" "287" "292" "296" "297" "303")
declare -a subsVal=("9" "11" "13" "19" "23" "26" "27" "44" "49" "51" "52" "67" "99" "100" "101" "105" "108" "111" "124" "128" "133" "168" "172" "185" "189" "195" "203" "233" "239" "242" "245" "262" "263" "275" "291")
 
for i in "${subsVal[@]}"; do

	echo "subject${i}"

	# python test.py --image /media/hdd/levibaljer/Combined_data/Fold1/test/images/${i}.nii.gz --result /media/hdd/levibaljer/Combined_data/Fold1/test/resvit_final/${i}_resvit.nii.gz
	python test.py --image /media/hdd/levibaljer/Combined_data/Fold1/val/images/${i}.nii.gz --result /media/hdd/levibaljer/Combined_data/Fold1/val/gambas/gambas_1block/${i}_gambas1.nii.gz
done