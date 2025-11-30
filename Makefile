setup:
	pip install -r requirements.txt

train:
	python train.py --epochs 50 --batch 16 --lr 5e-4 --mosaic 1.0

train-fast:
	python train.py --epochs 10 --batch 16 --lr 1e-3 --mosaic 0.5

run:
	python predict.py

docker-build:
	docker build -t french-cards-detector .

docker-run:
	docker run -it --rm -p 9696:9696 french-cards-detector
