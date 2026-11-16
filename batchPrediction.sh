#!/bin/bash
# Script to test segmentation of Hyperfine scans (as well as scans from other scanners).
# Levente Baljer (k2035837@kcl.ac.uk)

declare -a subsMICCAI=("POCEMR088" "POCEMR092" "POCEMR093" "POCEMR095" "POCEMR097" "POCEMR098" "POCEMR099" "POCEMR101" "POCEMR102" "POCEMR104")
declare -a subsKhula=("0" "1" "2" "4" "5" "6" "7" "8" "9" "11")
declare -a subsKM=("2" "5" "9" "14" "35" "58" "69" "76" "77" "86" "120" "141" "142" "146" "152" "153" "168" "182" "185" "201" "209" "213" "228" "231" "232" "234" "235" "250" "265" "276" "280" "293" "294" "304")
declare -a test_KM=("23" "24" "25" "47" "48" "53" "54" "62" "63" "73" "75" "81" "100" "101" "104" "108" "114" "115" "122" "124" "129" "155" "161" "172" "175" "189" "190" "195" "196" "203" "205" "215" "217" "223" "224" "237" "239" "242" "243" "258" "259" "300" "301")

for i in "${test_KM[@]}"; do

	echo "subject${i}"

	# mkdir -p /media/hdd/levibaljer/MICCAI_challenge/Validation_data/outputs_300_100_1_1_test4/${i}
	# mkdir -p /media/hdd/levibaljer/MICCAI_challenge/Validation_data/outputs_300_100_1_1_test4/${i}/Enhanced

	python test.py --image /media/hdd/levibaljer/Combined_data_test/test2/images/${i}_T2_AXI.nii.gz --result /media/hdd/levibaljer/Combined_data_test/test2/results/GAMBAS_OG/${i}_T2.nii.gz
	# python test.py --image /media/hdd/levibaljer/Combined_data_test/test/images_MONASH_reg/${i}_T2.nii.gz --result /media/hdd/levibaljer/Combined_data_test/test/results_MONASH_reg/wavelet_noise/${i}_T2.nii.gz
	# python test.py --image /media/hdd/levibaljer/MICCAI_challenge/labels_T1/${i}_T1.nii.gz --result /media/hdd/levibaljer/Combined_data_test/test/results/MONASH_FM_wavelet_HF_redo/${i}_FLAIR.nii.gz

done