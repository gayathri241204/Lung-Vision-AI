import mysql.connector
from mysql.connector import Error

def get_db_connection():
    """MySQL lungvision_db database connection setup"""
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='lungvision_db',
            user='root',
            password='pulmo@2026'
        )
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

# Quick Connection Test
if __name__ == "__main__":
    conn = get_db_connection()
    if conn:
        print("Connected to lungvision_db successfully!")
        conn.close()
    else:
        print("Connection failed. Check your password and MySQL server status.")