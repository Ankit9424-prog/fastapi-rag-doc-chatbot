.PHONY: run test lint format docker-up docker-down install

# Install dependencies
install:
	pip install -r requirements.txt

# Start infrastructure services
docker-up:
	docker-compose up -d

# Stop infrastructure services
docker-down:
	docker-compose down

# Run the application
run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
test:
	pytest tests/ -v

# Run linter
lint:
	ruff check app/ tests/

# Format code
format:
	ruff format app/ tests/
