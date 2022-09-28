for file in ../../data/youtube/hrvideo/hr_png/1*.png
do
	name=${file##*/}
	python random_crop.py --input_image $name --lr_dir '../../data/youtube/lrvideo/lr_png' --hlr_dir '../../data/youtube/hrvideo/hlr_png' --hr_dir '../../data/youtube/hrvideo/hr_png' --patchsize 128 --scale 3 --output_dir './youtubedb'
done
for file in ../../data/youtube/hrvideo/hr_png/2*.png
do
	name=${file##*/}
	python random_crop.py --input_image $name --lr_dir '../../data/youtube/lrvideo/lr_png' --hlr_dir '../../data/youtube/hrvideo/hlr_png' --hr_dir '../../data/youtube/hrvideo/hr_png' --patchsize 128 --scale 3 --output_dir './youtubedb'
done
for file in ../../data/youtube/hrvideo/hr_png/3*.png
do
	name=${file##*/}
	python random_crop.py --input_image $name --lr_dir '../../data/youtube/lrvideo/lr_png' --hlr_dir '../../data/youtube/hrvideo/hlr_png' --hr_dir '../../data/youtube/hrvideo/hr_png' --patchsize 128 --scale 3 --output_dir './youtubedb'
done

