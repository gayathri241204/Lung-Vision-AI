import mysql.connector
from mysql.connector import Error

def get_db_connection():
    """TiDB Cloud lungvision_db database connection setup"""
    try:
        connection = mysql.connector.connect(
            host='gateway01.ap-northeast-1.prod.aws.tidbcloud.com',
            port=4000,
            user='2B13KN8vRi2Xikn.root',
            password='Xj0LWOQ2MuO7pzjc',
            database='lungvision_db',
            ssl_verify_cert=False
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