llm-gateway
├── src
│   ├── main.py
│   ├── api
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── middleware.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── security.py
│   ├── db
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── crud.py
│   ├── services
│   │   ├── __init__.py
│   │   └── llm.py
│   └── schemas
│       ├── __init__.py
│       └── requests.py
├── alembic
│   ├── versions
│   │   └── __init__.py
│   ├── env.py
│   └── script.py.mako
├── tests
│   ├── __init__.py
│   ├── conftest.py
│   └── test_api.py
├── alembic.ini
├── pyproject.toml
├── requirements.txt
└── README.md