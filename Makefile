.PHONY: install ingest index run test clean lint

# Installation and setup
install:
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt')"

# Data processing pipeline
ingest:
	python -m app.ingest_data "database and model/blog.sql"

# Build vector index
index:
	python -m app.build_index

# Run the application
run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Testing
test:
	pytest tests/ -v

# Development tools
lint:
	black app/ tests/
	isort app/ tests/

# Clean generated files
clean:
	rm -rf app/storage/blog_data.*
	rm -rf app/storage/chroma/
	rm -rf app/storage/suppressed_links.log

# Setup from scratch
setup: install
	cp .env.example .env
	@echo "Please edit .env with your LLM API keys"
	@echo "Then run: make ingest && make index && make run"