dts=cifar10
md=cnncifar
sr=0.5

python ../main.py --alg fedavg --suffix $1 --dataset $dts --model $md --sr $sr
#python ../main.py fedasync --suffix $1 --dataset $dts --model $md --sr $sr