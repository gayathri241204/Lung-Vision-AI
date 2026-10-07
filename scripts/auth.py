import bcrypt
from scripts.db_connection import get_db_connection

def register_user(name, email, password, role):
    conn = get_db_connection()
    if not conn:
        return False, "Database connection failed."
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT user_id FROM users WHERE email = %s", (email,))
        if cur.fetchone():
            return False, "Email already registered."
        
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
        
        insert_query = "INSERT INTO users (name, email, password_hash, role) VALUES (%s, %s, %s, %s)"
        cur.execute(insert_query, (name, email, hashed, role))
        conn.commit()
        return True, "Registration successful!"
    except Exception as e:
        return False, str(e)
    finally:
        cur.close()
        conn.close()

def login_user(email, password):
    conn = get_db_connection()
    if not conn:
        return None, "Database connection failed."
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            return user, "Login successful."
        return None, "Invalid email or password."
    except Exception as e:
        return None, str(e)
    finally:
        cur.close()
        conn.close()

def get_emails_by_role(role):
    """MySQL டேட்டாபேஸிலிருந்து ரோல் வாரியாக ஈமெயில்களை எடுக்கும்"""
    try:
        from scripts.db_connection import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # MySQL-ல் '?' கிடையாது, '%s' மட்டுமே பயன்படுத்த வேண்டும்:
        query = "SELECT email FROM users WHERE LOWER(role) = LOWER(%s)"
        cursor.execute(query, (role.strip(),))
        rows = cursor.fetchall()
        
        emails = []
        for r in rows:
            if isinstance(r, dict):
                emails.append(r.get('email'))
            elif isinstance(r, (tuple, list)) and len(r) > 0:
                emails.append(r[0])
                
        cursor.close()
        conn.close()
        return emails
    except Exception as e:
        print("Error fetching emails by role:", e)
        return []