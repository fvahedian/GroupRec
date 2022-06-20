#!/bin/sh
touch record.txt
for i in 1 2 3 5 7 8 9 10;
do
	echo ${i}
	EST_TIME=$(TZ="EST" date)
	echo "Start - $EST_TIME\n" >> record.txt
​
	python3 main.py --input ../data/ml_small_mix/ --model mpe --mpth 0.40 --logname mpe_40_deep64_v${i}
	EST_TIME=$(TZ="EST" date)
	echo "Finish - $EST_TIME\n" >> record.txt
done
rm record.txt
