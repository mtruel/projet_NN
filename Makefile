install:
	python3.9 -m venv .env && . .env/bin/activate && pip install -r requirements.txt

gen_database:
	python3.9 src/database_gen.py

clean_data:
	rm -rf data/*

