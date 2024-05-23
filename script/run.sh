dts=cifar10
md=cnn
sr=0.5

python ../main.py fedavg --suffix $1 --dataset $dts --model $md --sr $sr
python ../main.py fedasync --suffix $1 --dataset $dts --model $md --sr $sr