#!/bin/bash
#for file in models-OBJ/*.binvox
#do
	#mv $file "models-custom-binvox/$(basename "$file")"
#done

#for file in models-OBJ/*.obj; do
#if ! test -f "models-custom-binvox/$(basename "$file" .obj).binvox"; then 
	#../binvox.exe -aw -cb -d 128 "$file"
	#mv "models-OBJ/$(basename "$file" .obj).binvox" "models-custom-binvox/$(basename "$file" .obj).binvox"
#fi; done


while IFS="," read -r column1 column2; do
shape_code=${column1//[$'\t\r\n ']}
rotation=${column2//[$'\t\r\n ']}
resolution=256
if ! test -f "tables-binvox-$resolution/$shape_code.binvox"; then
	if [ $rotation == '1' ]; then
		../binvox.exe -rotz -e -aw -cb -d $resolution "models-OBJ/$shape_code.obj"
		mv "models-OBJ/$shape_code.binvox" "tables-binvox-$resolution/$shape_code.binvox"
	elif [ $rotation == '2' ]; then
		../binvox.exe -rotz -rotz -e -aw -cb -d $resolution "models-OBJ/$shape_code.obj"
		mv "models-OBJ/$shape_code.binvox" "tables-binvox-$resolution/$shape_code.binvox"
	elif [ $rotation == '3' ]; then
		../binvox.exe -rotz -rotz -rotz -e -aw -cb -d $resolution "models-OBJ/$shape_code.obj"
		mv "models-OBJ/$shape_code.binvox" "tables-binvox-$resolution/$shape_code.binvox"
	else
		../binvox.exe -e -aw -cb -d $resolution "models-OBJ/$shape_code.obj"
		mv "models-OBJ/$shape_code.binvox" "tables-binvox-$resolution/$shape_code.binvox"
	fi
fi; done < Table_voxelization.csv
