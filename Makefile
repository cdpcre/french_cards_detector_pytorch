setup:
	pip install -r requirements.txt

train:
	python train.py --epochs 50 --batch 16 --lr 5e-4 --mosaic 1.0

train-fast:
	python train_fast.py --epochs 10 --batch 16 --device mps

run:
	python predict.py

test:
	pytest tests/ -v --ignore=tests/integration_test.py

test-unit:
	pytest tests/unit/ tests/test_predict.py -v

test-all:
	pytest tests/ -v

integration-test:
	python tests/integration_test.py


docker-build:
	docker build -t french-cards-detector .

docker-run:
	docker run -it --rm -p 9696:9696 -v $(PWD)/models:/app/models -e MODEL_PATH=models/best.pt french-cards-detector

docker-compose-up:
	docker-compose up --build
