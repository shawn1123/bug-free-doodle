import sqlite3
from datetime import datetime

def create_database():
    """Create the SQLite database and required tables"""
    try:
        # Connect to SQLite database (creates it if it doesn't exist)
        conn = sqlite3.connect('llm_logs.db')
        cursor = conn.cursor()

        # Create the logs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user TEXT,
            model TEXT NOT NULL,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            endpoint TEXT NOT NULL
        )
        ''')

        # Create indexes for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON logs(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user ON logs(user)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_model ON logs(model)')

        # Insert some test data
        test_data = (
            datetime.now().isoformat(),
            'test_user',
            'gpt-3.5-turbo',
            'Hello, world!',
            '{"response": "Hi there!"}',
            'completions'
        )
        cursor.execute('''
        INSERT INTO logs (timestamp, user, model, prompt, response, endpoint)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', test_data)

        # Commit changes and close connection
        conn.commit()
        print("Database created successfully with test data!")

    except sqlite3.Error as e:
        print(f"Error creating database: {e}")

    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    create_database()