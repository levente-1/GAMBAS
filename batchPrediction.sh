#!/bin/bash
# Script to test segmentation of Hyperfine scans (as well as scans from other scanners).
# Levente Baljer (k2035837@kcl.ac.uk)

# declare -a subsTest=("2" "4" "5" "17" "47" "48" "62" "63" "70" "76" "77" "83" "88" "89" "94" "97" "117" "121" "146" "150" "167" "176" "181" "190" "201" "211" "212" "213" "217" "247" "248" "258" "259" "260" "266" "267" "270" "279" "287" "292" "296" "297" "303")
# declare -a subsTestFold2=("9" "15" "23" "24" "25" "54" "60" "69" "73" "78" "79" "80" "82" "86" "90" "91" "92" "99" "104" "114" "115" "132" "155" "169" "182" "192" "193" "208" "209" "222" "223" "228" "231" "232" "242" "250" "252" "274" "276" "281" "283" "293" "302")
# declare -a subsVal=("9" "11" "13" "19" "23" "26" "27" "44" "49" "51" "52" "67" "99" "100" "101" "105" "108" "111" "124" "128" "133" "168" "172" "185" "189" "195" "203" "233" "239" "242" "245" "262" "263" "275" "291")
# declare -a subsUganda=("20011" "20012" "20031" "40038" "40042" "40059" "40067" "40086" "40088" "40095" "40104" "40106" "40113" "40152" "40154" "40156" "40164" "40177" "40178" "40206" "40241" "40243" "40250" "40274" "40280" "40284" "40285" "40309" "40345" "40364" "40381" "40409" "40452")
# declare -a subsUganda2=("20020" "40122" "40248" "40337") 
# declare -a subsBonn=("h1" "h65" "h108")
# declare -a subsHype=("00" "01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19")
# declare -a subsBad=("1208" "1266")
# declare -a subsDolphin=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25" "26" "27" "28" "29" "30" "31" "32" "33" "34" "35" "36" "37" "38" "39" "40" "41" "42" "43" "44" "45")
declare -a subsMICCAI=("POCEMR088" "POCEMR092" "POCEMR093" "POCEMR095" "POCEMR097" "POCEMR098" "POCEMR099" "POCEMR101" "POCEMR102" "POCEMR104")
# declare -a subsMICCAIval=("POCEMR013" "POCEMR020" "POCEMR027" "POCEMR040" "POCEMR044" "POCEMR046" "POCEMR058" "POCEMR072" "POCEMR105" "POCEMR106")

for i in "${subsMICCAI[@]}"; do

	echo "subject${i}"

	# mkdir -p /media/hdd/levibaljer/MICCAI_challenge/Validation_data/outputs_test1/${i}
	# mkdir -p /media/hdd/levibaljer/MICCAI_challenge/Validation_data/outputs_test1/${i}/Enhanced

	# python test.py --image /media/hdd/levibaljer/Combined_data/Fold1/test/images/${i}.nii.gz --result /media/hdd/levibaljer/Combined_data/Fold1/test/swin_unet_48/${i}_swin48.nii.gz
	# python test.py --image /media/hdd/levibaljer/Combined_data/Fold1/val/images/${i}.nii.gz --result /media/hdd/levibaljer/Combined_data/Fold1/val/gambas/9block/${i}_gambas9.nii.gz
	# python test.py --image /media/hdd/levibaljer/Uganda/ULF_AXI/${i}/AXIWarped.nii.gz --result /media/hdd/levibaljer/Uganda/gambas_final/${i}_gambas_final.nii.gz
	# python test.py --image /media/hdd/levibaljer/Bonn/registered/${i}/ULFWarped.nii.gz --result /media/hdd/levibaljer/Bonn/gambas3/${i}_gambas.nii.gz
	# python test.py --image /media/hdd/levibaljer/Combined_data/badExample/images/${i}_axi_Warped.nii.gz --result /media/hdd/levibaljer/Combined_data/badExample/results/${i}_gambas.nii.gz
	python test.py --image /media/hdd/levibaljer/MICCAI_challenge/images_T2/${i}_T2.nii.gz --result /media/hdd/levibaljer/MICCAI_challenge/ablations_results/resnet/${i}_T1.nii.gz
	# python test.py --image /media/hdd/levibaljer/MICCAI_challenge/Validation_data/outputs_t2_input/${i}/Enhanced/${i}_T2.nii.gz --result /media/hdd/levibaljer/MICCAI_challenge/Validation_data/outputs_realSynth_T1/${i}_T1.nii.gz
	# python test.py --image /media/hdd/levibaljer/MICCAI_challenge/Validation_data/T2/${i}_T2.nii.gz --result /media/hdd/levibaljer/MICCAI_challenge/Validation_data/outputs_test1/${i}/Enhanced/${i}_FLAIR.nii.gz

done