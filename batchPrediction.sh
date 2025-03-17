#!/bin/bash
# Script to test segmentation of Hyperfine scans (as well as scans from other scanners).
# Levente Baljer (k2035837@kcl.ac.uk)

# declare -a subsTest=("2" "4" "5" "17" "47" "48" "62" "63" "70" "76" "77" "83" "88" "89" "94" "97" "117" "121" "146" "150" "167" "176" "181" "190" "201" "211" "212" "213" "217" "247" "248" "258" "259" "260" "266" "267" "270" "279" "287" "292" "296" "297" "303")
# declare -a subsTestFold2=("9" "15" "23" "24" "25" "54" "60" "69" "73" "78" "79" "80" "82" "86" "90" "91" "92" "99" "104" "114" "115" "132" "155" "169" "182" "192" "193" "208" "209" "222" "223" "228" "231" "232" "242" "250" "252" "274" "276" "281" "283" "293" "302")
# declare -a subsVal=("9" "11" "13" "19" "23" "26" "27" "44" "49" "51" "52" "67" "99" "100" "101" "105" "108" "111" "124" "128" "133" "168" "172" "185" "189" "195" "203" "233" "239" "242" "245" "262" "263" "275" "291")
# declare -a subsUganda=("20011" "20012" "20031" "40038" "40042" "40059" "40067" "40086" "40088" "40095" "40104" "40106" "40113" "40152" "40154" "40156" "40164" "40177" "40178" "40206" "40241" "40243" "40250" "40274" "40280" "40284" "40285" "40309" "40345" "40364" "40381" "40409" "40452")
declare -a subsUganda2=("20020" "40122" "40248" "40337") 
# declare -a subsBonn=("h1" "h65" "h108")
# declare -a subsHype=("00" "01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19")
# declare -a subsBad=("1208" "1266")

for i in "${subsUganda2[@]}"; do

	echo "subject${i}"

	# python test.py --image /media/hdd/levibaljer/Combined_data/Fold1/test/images/${i}.nii.gz --result /media/hdd/levibaljer/Combined_data/Fold1/test/swin_unet_48/${i}_swin48.nii.gz
	# python test.py --image /media/hdd/levibaljer/Combined_data/Fold1/val/images/${i}.nii.gz --result /media/hdd/levibaljer/Combined_data/Fold1/val/gambas/9block/${i}_gambas9.nii.gz
	python test.py --image /media/hdd/levibaljer/Uganda/ULF_AXI/${i}/AXIWarped.nii.gz --result /media/hdd/levibaljer/Uganda/gambas_final/${i}_gambas_final.nii.gz
	# python test.py --image /media/hdd/levibaljer/Bonn/registered/${i}/ULFWarped.nii.gz --result /media/hdd/levibaljer/Bonn/gambas3/${i}_gambas.nii.gz
	# python test.py --image /media/hdd/levibaljer/Combined_data/badExample/images/${i}_axi_Warped.nii.gz --result /media/hdd/levibaljer/Combined_data/badExample/results/${i}_gambas.nii.gz

done