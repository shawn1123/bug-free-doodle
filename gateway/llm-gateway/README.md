# README.md

# LLM Gateway

## Overview

LLM Gateway is a FastAPI application that serves as a gateway for users to connect to OpenAI-compatible endpoint LLMs. It supports chat completions and completions endpoints, ensuring secure access through API key validation.

## Features

- API key validation in request headers
- Logging of user prompts and responses in a database
- Options for static response or streaming output
- Well-structured codebase with clear separation of concerns

## Project Structure

```
llm-gateway
├── src
│   ├── main.py          # Entry point of the FastAPI application
│   ├── api              # Contains API routes and middleware
│   ├── core             # Core configuration and security utilities
│   ├── db               # Database models and CRUD operations
│   ├── services         # Functions interacting with LLMs
│   └── schemas          # Request and response schemas
├── alembic              # Database migration scripts
├── tests                # Unit tests for the application
├── alembic.ini         # Alembic configuration file
├── pyproject.toml      # Project configuration file
└── requirements.txt     # Required Python packages
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd llm-gateway
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up the database and run migrations:
   ```
   alembic upgrade head
   ```

## Usage

To run the FastAPI application, execute:
```
uvicorn src.main:app --reload
```

Visit `http://localhost:8000/docs` to access the interactive API documentation.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.