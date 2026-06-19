.PHONY: run test lint format docker-up docker-down docker-build install migrate clean

# Install dependencies
install:
	pip install -r requirements.txt

# Start infrastructure services (Qdrant, Redis, PostgreSQL)
docker-up:
	docker compose up -d qdrant redis postgres

# Stop infrastructure services
docker-down:
	docker compose down

# Build Docker image for the application
docker-build:
	docker compose build app

# Run database migrations
migrate:
	alembic upgrade head

# Create a new migration
migration:
	alembic revision --autogenerate -m "$(msg)"

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

# Remove build artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/
