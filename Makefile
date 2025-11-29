setup:
	pip install -r requirements.txt

train:
	python train.py

run:
	python predict.py

docker-build:
	docker build -t french-cards-detector .

docker-run:
	docker run -it --rm -p 9696:9696 french-cards-detector
