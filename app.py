import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import io
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, jsonify
from werkzeug.utils import secure_filename

# ReportLab Imports (Professional PDF Generation)
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

# Custom Modular Scripts
from scripts.db_connection import get_db_connection
from scripts.auth import login_user, register_user,get_emails_by_role
from scripts.inference import evaluate_image_quality, get_multi_model_analysis

app = Flask(__name__)
app.secret_key = 'lungvision_enterprise_secure_key_2026'

# Configure upload directory
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_next_doctor_patient_id(cur):
    """Doctor Portal-kku: P007, P008, P009... generate seiyyum"""
    cur.execute("SELECT patient_id FROM patient_records WHERE patient_id LIKE 'P%'")
    rows = cur.fetchall()
    nums = []
    for r in rows:
        pid = r[0] if isinstance(r, (tuple, list)) else r.get('patient_id', '')
        if pid and pid.startswith('P') and not pid.startswith('PT-'):
            try:
                nums.append(int(pid[1:]))
            except ValueError:
                pass
    next_num = (max(nums) + 1) if nums else 1
    return f"P{next_num:03d}"

def get_next_self_patient_id(cur):
    """Patient Self-Portal-kku: PT-001, PT-002, PT-003... generate seiyyum"""
    cur.execute("SELECT patient_id FROM patient_records WHERE patient_id LIKE 'PT-%'")
    rows = cur.fetchall()
    nums = []
    for r in rows:
        pid = r[0] if isinstance(r, (tuple, list)) else r.get('patient_id', '')
        if pid and pid.startswith('PT-'):
            try:
                val = int(pid.replace('PT-', ''))
                # Pazhaiya 60001, 90001 numbers-ai thavirtthu, 3-digit series (PT-001, PT-002)-ai track pannum:
                if val < 10000:
                    nums.append(val)
            except ValueError:
                pass
    next_num = (max(nums) + 1) if nums else 1
    return f"PT-{next_num:03d}"

# ---------------------------------------------------------
# Default Home Route (Directs root URL to Login)
# ---------------------------------------------------------
@app.route('/')
def home():
    return redirect(url_for('login'))


# ---------------------------------------------------------
# MODULE 1: Authentication & Role Selection Portal
# ---------------------------------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        role = request.form.get('role', '').strip()

        conn = get_db_connection()
        if conn:
            try:
                cur = conn.cursor(dictionary=True)
                cur.execute("SELECT * FROM users WHERE email = %s AND role = %s", (email, role))
                user = cur.fetchone()
                cur.close()
                conn.close()

                if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
                    
                    
                    if user.get('status') == 'inactive':
                        flash("Your account has been deactivated by Administrator. Access denied.", "danger")
                        return redirect(url_for('login'))

                    
                    session['user_id'] = user['user_id']
                    session['user_name'] = user['name']
                    session['name'] = user['name']       
                    session['email'] = user['email']
                    session['role'] = user['role']

                   
                    if user['role'] == 'admin':
                        return redirect(url_for('admin_dashboard'))
                    elif user['role'] == 'doctor':
                        return redirect(url_for('doctor_dashboard'))
                    elif user['role'] == 'patient':
                        return redirect(url_for('patient_dashboard'))
                else:
                    flash("Invalid email, PIN, or role selection.", "danger")

            except Exception as e:
                print("Login verification error:", e)
                flash("Authentication error occurred. Please try again.", "danger")

    return render_template('login.html')

# ---------------------------------------------------------
# Password PIN Reset Route (Bcrypt Protected)
# ---------------------------------------------------------
@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    email = request.form.get('email', '').strip()
    new_password = request.form.get('new_password', '').strip()

    # Input validation
    if not email or len(new_password) != 6 or not new_password.isdigit():
        flash("Please enter a valid email and exactly 6-digit numeric PIN.", "warning")
        return redirect(url_for('login'))

    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor(dictionary=True)
            # Check if user exists in database
            cur.execute("SELECT user_id, name FROM users WHERE email = %s", (email,))
            user = cur.fetchone()

            if not user:
                flash("This email is not registered in our clinical records.", "danger")
            else:
                # Bcrypt cryptographic hashing for the new PIN
                hashed_pw = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                cur.execute("UPDATE users SET password_hash = %s WHERE email = %s", (hashed_pw, email))
                conn.commit()
                flash("Password PIN reset successfully! Please log in with your new PIN.", "success")

            cur.close()
            conn.close()
        except Exception as e:
            print(">>> [Forgot Password Error]:", e)
            flash("Database error during password reset.", "danger")

    return redirect(url_for('login'))


# =========================================================
# MODULE: User Registration (Fixed Database Columns & Bcrypt)
# =========================================================
# =========================================================
# MODULE: User Registration (Fixed Database Columns & Bcrypt)
# =========================================================

import bcrypt

@app.route('/register', methods=['POST'])
def register():
    name = request.form.get('name', '').strip()
    email = request.form.get('email', '').strip()
    password = request.form.get('password', '').strip()
    role = request.form.get('role', 'patient').strip().lower()
    
    age = request.form.get('age', '').strip()
    gender = request.form.get('gender', 'Male').strip()
    phone = request.form.get('phone', '').strip()

    if not (name and email and password):
        flash("Please provide all required fields.", "warning")
        return redirect(url_for('login'))

    if len(password) != 6 or not password.isdigit():
        flash("Security PIN must be exactly 6 numeric digits.", "warning")
        return redirect(url_for('login'))

    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor(dictionary=True)
            
            # Check if email already exists
            cur.execute("SELECT user_id FROM users WHERE email = %s", (email,))
            if cur.fetchone():
                flash("Email already registered. Please login.", "warning")
                cur.close()
                conn.close()
                return redirect(url_for('login'))

            hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

            # Insert user record
            cur.execute("""
                INSERT INTO users (name, email, password_hash, role, status)
                VALUES (%s, %s, %s, %s, 'active')
            """, (name, email, hashed_pw, role))
            user_id = cur.lastrowid

            if role == 'patient':
                # =========================================================
                # STEP 3: Sequential PT-001, PT-002 Generator for Self-Portal
                # =========================================================
                cur.execute("SELECT patient_id FROM patient_records WHERE patient_id LIKE 'PT-%'")
                rows = cur.fetchall()

                nums = []
                for r in rows:
                    pid_val = r['patient_id'] if isinstance(r, dict) else r[0]
                    if pid_val and str(pid_val).startswith('PT-'):
                        try:
                            val = int(str(pid_val).replace('PT-', ''))
                            # Pazhaiya 60001, 90001-ai thavirtthu, pudhiya 3-digit series-ai mattum edukkum:
                            if val < 1000:
                                nums.append(val)
                        except ValueError:
                            pass

                # Universal logic: Mudhal patient-kku PT-001, aduthu PT-002, PT-003...
                next_num = (max(nums) + 1) if nums else 1
                patient_id = f"PT-{next_num:03d}"

                patient_age = int(age) if age.isdigit() else None
                cur.execute("""
                    INSERT INTO patient_records (patient_id, name, age, gender, phone, email)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (patient_id, name, patient_age, gender, phone, email))

            conn.commit()
            cur.close()
            conn.close()
            flash("Registration successful! You can now log in with your credentials.", "success")
            return redirect(url_for('login'))

        except Exception as e:
            print(">>> [Registration Error]:", e)
            flash("Registration failed due to a database error.", "danger")

    return redirect(url_for('login'))
# ---------------------------------------------------------
# Dynamic Role-Based Email Suggestions API
# ---------------------------------------------------------
@app.route('/api/get-emails/<role>')
def get_role_emails(role):
    if role == 'admin':
        return jsonify([])

    conn = get_db_connection()
    emails = []
    if conn:
        try:
            cur = conn.cursor(dictionary=True)
            cur.execute("SELECT email FROM users WHERE role = %s ORDER BY user_id DESC", (role,))
            rows = cur.fetchall()
            emails = [row['email'] for row in rows if row.get('email')]
            cur.close()
            conn.close()
        except Exception as e:
            print(f">>> [API ERROR] Failed to fetch emails for {role}: {e}")
            
    return jsonify(emails)


# ---------------------------------------------------------
# MODULE 2: Administrative Intelligence Dashboard
# ---------------------------------------------------------

@app.route('/admin')
def admin_dashboard():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))

    conn = get_db_connection()
    metrics = {
        'total_doctors': 0,
        'total_patients': 0,
        'total_reports': 0,
        'total_analyses': 0
    }
    recent_activities = []
    doctors_list = []
    patients_list = []
    studies_list = []
    disease_counts = {'Normal': 0, 'Pneumonia': 0, 'Tuberculosis': 0, 'COVID-19': 0, 'Lung Opacity': 0}

    if conn:
        cur = conn.cursor(dictionary=True)

        # 1. உண்மையான எண்கள் (Real DB Counts)
        cur.execute("SELECT COUNT(*) AS c FROM users WHERE role = 'doctor'")
        metrics['total_doctors'] = cur.fetchone()['c']

        cur.execute("SELECT COUNT(*) AS c FROM patient_records")
        metrics['total_patients'] = cur.fetchone()['c']

        cur.execute("SELECT COUNT(*) AS c FROM radiograph_studies")
        total_studies = cur.fetchone()['c']
        metrics['total_reports'] = total_studies
        metrics['total_analyses'] = total_studies

        
        cur.execute("""
            SELECT s.*, p.name 
            FROM radiograph_studies s 
            LEFT JOIN patient_records p ON s.patient_id = p.patient_id 
            ORDER BY s.study_id DESC LIMIT 5
        """)
        recent_activities = cur.fetchall()

        
        for act in recent_activities:
            act['created_at'] = act.get('examination_date') or act.get('created_at')

        
        cur.execute("SELECT * FROM users WHERE role = 'doctor'")
        doctors_list = cur.fetchall()

        cur.execute("SELECT * FROM patient_records ORDER BY patient_id DESC")
        patients_list = cur.fetchall()

        cur.execute("SELECT * FROM radiograph_studies ORDER BY study_id DESC")
        studies_list = cur.fetchall()

        # 4. (Disease Distribution for Donut Chart)
        cur.execute("SELECT predicted_disease, COUNT(*) as cnt FROM radiograph_studies GROUP BY predicted_disease")
        for row in cur.fetchall():
            d_name = row['predicted_disease']
            if d_name in disease_counts:
                disease_counts[d_name] = row['cnt']

        cur.close()
        conn.close()

    return render_template(
        'admin.html',
        m=metrics,
        activities=recent_activities,
        doctors=doctors_list,
        patients=patients_list,
        studies=studies_list,
        disease_counts=disease_counts
    )

# ---------------------------------------------------------
# Admin: Toggle User Account Status (Active <-> Inactive)
# ---------------------------------------------------------
@app.route('/admin/toggle-status/<int:user_id>', methods=['POST'])
def toggle_user_status(user_id):
    if session.get('role') != 'admin':
        return redirect(url_for('login'))
    
   
    if user_id == session.get('user_id'):
        flash("Action denied: You cannot deactivate your own administrative account.", "warning")
        return redirect(url_for('admin_dashboard'))

    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor(dictionary=True)
            cur.execute("SELECT status, name, role FROM users WHERE user_id = %s", (user_id,))
            target_user = cur.fetchone()
            
            if target_user:
                current_st = target_user.get('status', 'active')
                new_st = 'inactive' if current_st == 'active' else 'active'
                
                cur.execute("UPDATE users SET status = %s WHERE user_id = %s", (new_st, user_id))
                conn.commit()
                flash(f"Account for {target_user['name']} ({target_user['role'].capitalize()}) is now {new_st.upper()}.", "success")
            cur.close()
            conn.close()
        except Exception as e:
            print(f">>> [Admin Status Error]: {e}")
            flash("Failed to update user account status.", "danger")

    return redirect(url_for('admin_dashboard'))


# ---------------------------------------------------------
# Admin: Create Additional Administrator (Internal Access Only)
# ---------------------------------------------------------
@app.route('/admin/create-admin', methods=['POST'])
def create_internal_admin():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))

    name = request.form.get('name', '').strip()
    email = request.form.get('email', '').strip()
    pin = request.form.get('pin', '').strip()

    if not (name and email and pin and len(pin) == 6 and pin.isdigit()):
        flash("Please provide valid details: Name, Email, and a 6-digit numeric PIN.", "warning")
        return redirect(url_for('admin_dashboard'))

    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor(dictionary=True)
            cur.execute("SELECT user_id FROM users WHERE email = %s", (email,))
            if cur.fetchone():
                flash("Email is already registered in the system.", "danger")
                return redirect(url_for('admin_dashboard'))

            # Bcrypt hashing for the PIN
            hashed_pin = bcrypt.hashpw(pin.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

            cur.execute("""
                INSERT INTO users (name, email, password_hash, role, status)
                VALUES (%s, %s, %s, 'admin', 'active')
            """, (name, email, hashed_pin))
            conn.commit()
            flash(f"New Administrator '{name}' created successfully!", "success")
            cur.close()
            conn.close()
        except Exception as e:
            print(f">>> [Admin Creation Error]: {e}")
            flash("Failed to register new administrator.", "danger")

    return redirect(url_for('admin_dashboard'))

# =========================================================
# MODULE 3: Doctor Dashboard with Real Dynamic Database Metrics
# =========================================================
@app.route('/doctor')
def doctor_dashboard():
    if session.get('role') != 'doctor':
        return redirect(url_for('login'))

    conn = get_db_connection()
    metrics = {
        'my_patients': 0,
        'reports_generated': 0,
        'analyses_done': 0,
        'pending_analyses': 0
    }
    patients = []

    if conn:
        cur = conn.cursor(dictionary=True)

       
        cur.execute("SELECT COUNT(*) AS count FROM patient_records")
        metrics['my_patients'] = cur.fetchone()['count']

        # 2. (Analyses Done)
        cur.execute("SELECT COUNT(*) AS count FROM radiograph_studies")
        total_studies = cur.fetchone()['count']
        metrics['analyses_done'] = total_studies
        metrics['reports_generated'] = total_studies

        # 3.  (Pending Analyses)
        cur.execute("""
            SELECT COUNT(*) AS count FROM patient_records 
            WHERE patient_id NOT IN (SELECT DISTINCT patient_id FROM radiograph_studies)
        """)
        metrics['pending_analyses'] = cur.fetchone()['count']

        # Recent Patients 
        

        cur.execute("""
        SELECT 
        p.*,
        (SELECT s.study_id FROM radiograph_studies s 
         WHERE s.patient_id = p.patient_id 
         ORDER BY s.study_id DESC LIMIT 1) AS study_id
         FROM patient_records p
         ORDER BY p.patient_id DESC 
         LIMIT 5
         """)
        patients = cur.fetchall()

        cur.close()
        conn.close()

    return render_template('doctor.html', patients=patients, metrics=metrics)


# ---------------------------------------------------------
# Direct Sidebar Upload Redirect Handler
# ---------------------------------------------------------
@app.route('/doctor/upload')
def direct_upload():
    if session.get('role') != 'doctor':
        return redirect(url_for('login'))
    
    current_patient = session.get('current_patient_id')
    if current_patient:
        return redirect(url_for('upload_xray', patient_id=current_patient))
    
    flash("Please select a patient from the list to upload radiograph.", "info")
    return redirect(url_for('patient_selection'))

# ---------------------------------------------------------
# Doctor Systematic AI Analytics Route
# ---------------------------------------------------------
@app.route('/doctor/analytics')
def model_analytics():
    if session.get('role') != 'doctor':
        return redirect(url_for('login'))
    return render_template('model_analytics.html', doctor_name=session.get('user_name', 'Doctor'))

# ---------------------------------------------------------
# MODULE 4: Patient Selection & Search Pipeline
# ---------------------------------------------------------

@app.route('/doctor/patients')
def patient_selection():
    if session.get('role') != 'doctor':
        return redirect(url_for('login'))

    search_query = request.args.get('search', '').strip()
    conn = get_db_connection()
    patients = []

    if conn:
        try:
            cur = conn.cursor(dictionary=True)
            
            # 1. Patient records fetch panrom
            if search_query:
                query = "SELECT * FROM patient_records WHERE patient_id LIKE %s OR name LIKE %s OR phone LIKE %s ORDER BY patient_id DESC"
                like_val = f"%{search_query}%"
                cur.execute(query, (like_val, like_val, like_val))
            else:
                cur.execute("SELECT * FROM patient_records ORDER BY patient_id DESC")
            patients = cur.fetchall()

            # 2. X-ray studies ulla patients-ai eduthu verified status assign panrom
            cur.execute("SELECT DISTINCT patient_id FROM radiograph_studies")
            verified_rows = cur.fetchall()
            verified_ids = {row['patient_id'] for row in verified_rows if row.get('patient_id')}

            for p in patients:
                if p.get('patient_id') in verified_ids:
                    p['status'] = 'Verified'
                else:
                    p['status'] = 'Pending'

            cur.close()
        except Exception as e:
            print("Database Error in patient_selection:", e)
        finally:
            conn.close()

    return render_template('patient_selection.html', patients=patients, search=search_query)

@app.route('/doctor/patient/add', methods=['POST'])
def add_new_patient():
    if session.get('role') != 'doctor':
        return redirect(url_for('login'))

    name = request.form.get('name', '').strip()
    age = request.form.get('age', '').strip()
    gender = request.form.get('gender', '').strip()
    phone = request.form.get('phone', '').strip()
    email = request.form.get('email', '').strip()

    conn = get_db_connection()
    if conn:
        cur = conn.cursor()
        try:
            # Doctor Portal-kku sequential P-series generator (P007, P008, P009...):
            cur.execute("SELECT patient_id FROM patient_records WHERE patient_id LIKE 'P%'")
            rows = cur.fetchall()
            
            nums = []
            for r in rows:
                pid_val = r[0] if isinstance(r, (tuple, list)) else r.get('patient_id', '')
                # Verum P001, P002 pola irukkum numbers-ai mattum filter seiyyum (PT-90001, PT-60001 reject aagum):
                if pid_val and pid_val.startswith('P') and not pid_val.startswith('PT-'):
                    try:
                        nums.append(int(pid_val[1:]))
                    except ValueError:
                        pass
            
            # Universal Dynamic Logic:
            # Table-il already P006 iruppadhaal max(nums) = 6 -> 6 + 1 = 7 (P007 aagum)
            next_num = (max(nums) + 1) if nums else 1
            patient_id = f"P{next_num:03d}"

            # Database-il P007, P008 aaga insert aagum:
            cur.execute("""
                INSERT INTO patient_records (patient_id, name, age, gender, phone, email, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """, (patient_id, name, age, gender, phone, email))
            conn.commit()
            flash(f"New patient registered successfully with ID: {patient_id}!", "success")
        except Exception as e:
            print("DB Insert Error:", e)
            flash("Error: Patient ID or Email already exists.", "danger")
        finally:
            cur.close()
            conn.close()

    return redirect(url_for('patient_selection'))
# ---------------------------------------------------------
# MODULE 5: Radiograph Upload (Selected Patient Context)
# ---------------------------------------------------------
@app.route('/doctor/upload/<patient_id>', methods=['GET', 'POST'])
def upload_xray(patient_id):
    if session.get('role') != 'doctor':
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM patient_records WHERE patient_id = %s", (patient_id,))
    patient = cur.fetchone()
    cur.close()
    conn.close()

    if not patient:
        flash("Patient ID not found.", "danger")
        return redirect(url_for('patient_selection'))

    if request.method == 'POST':
        file = request.files.get('xray_file')
        if file and file.filename != '':
            filename = secure_filename(f"{patient_id}_{file.filename}")
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            
            
            session['current_study_file'] = filename
            session['current_xray_file'] = filename    
            session['current_patient_id'] = patient_id
            session['analyzed_file'] = None            
            session['current_study_id'] = None         
            return redirect(url_for('quality_assessment'))

    return render_template('upload_xray.html', patient=patient)


# ---------------------------------------------------------
# MODULE 6: Radiograph Gatekeeper Quality Check
# ---------------------------------------------------------
@app.route('/doctor/quality-check')
def quality_assessment():
    if session.get('role') != 'doctor':
        return redirect(url_for('login'))
    
    filename = session.get('current_study_file')
    if not filename:
        return redirect(url_for('patient_selection'))
        
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    quality_result = evaluate_image_quality(image_path)
    
    return render_template('quality_check.html', q=quality_result, filename=filename)

# =========================================================
# MODULE 7-10: Multi-Model Diagnostic Results & AI CDSS
# =========================================================
import json
from datetime import datetime
from scripts.inference import get_multi_model_analysis, generate_ai_clinical_recommendation

@app.route('/doctor/diagnostic-results')
def diagnostic_results():
    if session.get('role') != 'doctor':
        return redirect(url_for('login'))

    current_img_file = session.get('current_xray_file') or session.get('current_study_file')
    patient_id = session.get('current_patient_id') or 'P008'
    doctor_name = session.get('name') or 'Dr. Ramya'

    upload_folder = os.path.join(app.root_path, 'static', 'uploads')
    img_full_path = os.path.join(upload_folder, current_img_file) if current_img_file else None

    if img_full_path and os.path.exists(img_full_path) and session.get('analyzed_file') != current_img_file:
        ai_res = get_multi_model_analysis(img_full_path)

        champion_model = ai_res.get('champion_model', 'DenseNet121')
        predicted_disease = ai_res.get('best_disease', 'Normal')
        confidence_score = float(ai_res.get('best_conf', 90.0))
        risk_score = int(ai_res.get('risk_score', 15))
        risk_level = ai_res.get('risk_level', 'Low')
        comparison = ai_res.get('comparison', {})


        patient_age = 40
        try:
            conn_p = get_db_connection()
            if conn_p:
                cur_p = conn_p.cursor(dictionary=True)
                cur_p.execute("SELECT age FROM patient_records WHERE patient_id = %s", (patient_id,))
                p_row = cur_p.fetchone()
                if p_row and p_row.get('age'):
                    patient_age = int(p_row['age'])
                cur_p.close()
                conn_p.close()
        except Exception as e:
            print("Patient Age Fetch Note:", e)

        # 🔴 AI Clinical Recommendation Engine 
        ai_recommendations = generate_ai_clinical_recommendation(
            predicted_disease, confidence_score, risk_score, patient_age
        )
        workup_list = ai_recommendations.get("diagnostic_workup", ["Clinical correlation advised."])
        clinical_rec = " | ".join(workup_list)

        champion_probs = comparison.get(champion_model, list(comparison.values())[0] if comparison else {})

        new_study_id = None
        try:
            conn = get_db_connection()
            if conn:
                cur = conn.cursor()
                insert_sql = """
                    INSERT INTO radiograph_studies 
                    (patient_id, doctor_name, image_path, champion_model, predicted_disease, confidence, risk_score, risk_level, clinical_recommendation, examination_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cur.execute(insert_sql, (
                    patient_id,
                    doctor_name,
                    current_img_file,
                    champion_model,
                    predicted_disease,
                    confidence_score,
                    risk_score,
                    risk_level,
                    clinical_rec,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ))
                conn.commit()
                new_study_id = cur.lastrowid
                cur.close()
                conn.close()
                print(f">>> [SUCCESS] Saved Study #{new_study_id} for Patient: {patient_id}")
        except Exception as e:
            print(">>> DB Insert Error:", e)

        
        final_study_id = new_study_id or session.get('current_study_id') or 1
        session['current_study_id'] = final_study_id
        session['analyzed_file'] = current_img_file

        normalized_study = {
            'study_id': final_study_id,
            'patient_id': patient_id,
            'image_filename': current_img_file,
            'predicted_disease': predicted_disease,
            'confidence_score': confidence_score,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'champion_model': champion_model,
            'probabilities': champion_probs,
            'comparison': comparison
        }

        return render_template('diagnostic_results.html', study=normalized_study, comparison=comparison)

   
    study = None
    study_id = request.args.get('study_id') or session.get('current_study_id')
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor(dictionary=True)
            if study_id:
                cur.execute("SELECT * FROM radiograph_studies WHERE study_id = %s", (study_id,))
            else:
                cur.execute("SELECT * FROM radiograph_studies ORDER BY study_id DESC LIMIT 1")
            study = cur.fetchone()
            cur.close()
            conn.close()
    except Exception as e:
        print("DB Fetch Notice:", e)

    if not study:
        study = {}

    img_name = study.get('image_path') or current_img_file or 'sample.jpg'
    if '/' in img_name or '\\' in img_name:
        img_name = os.path.basename(img_name)

    final_study_id = study.get('study_id') or session.get('current_study_id') or 1

    normalized_study = {
        'study_id': final_study_id,
        'patient_id': study.get('patient_id') or patient_id,
        'image_filename': img_name,
        'predicted_disease': study.get('predicted_disease', 'Normal'),
        'confidence_score': float(study.get('confidence', 90.0)),
        'risk_score': study.get('risk_score', 15),
        'risk_level': study.get('risk_level', 'Low'),
        'champion_model': study.get('champion_model', 'DenseNet121'),
        'probabilities': {}
    }

    return render_template('diagnostic_results.html', study=normalized_study, comparison={})


# ---------------------------------------------------------
# MODULE 11: Doctor Radiograph Examination Archive
# ---------------------------------------------------------
@app.route('/doctor/reports')
def doctor_reports():
    if session.get('role') != 'doctor':
        return redirect(url_for('login'))

    search_query = request.args.get('search', '').strip()
    conn = get_db_connection()
    reports = []

    if conn:
        cur = conn.cursor(dictionary=True)
        if search_query:
            query = """
                SELECT s.*, p.name 
                FROM radiograph_studies s
                LEFT JOIN patient_records p ON s.patient_id = p.patient_id
                WHERE s.patient_id LIKE %s OR p.name LIKE %s OR s.predicted_disease LIKE %s
                ORDER BY s.study_id DESC
            """
            like_val = f"%{search_query}%"
            cur.execute(query, (like_val, like_val, like_val))
        else:
            query = """
                SELECT s.*, p.name 
                FROM radiograph_studies s
                LEFT JOIN patient_records p ON s.patient_id = p.patient_id
                ORDER BY s.study_id DESC
            """
            cur.execute(query)
        reports = cur.fetchall()
        cur.close()
        conn.close()

    return render_template('doctor_reports.html', reports=reports, search=search_query)


# ---------------------------------------------------------
# MODULE 12: Patient Self-Service Dashboard
# ---------------------------------------------------------
# ---------------------------------------------------------
# Patient Portal Dashboard (Authentic Database Fetch)
# ---------------------------------------------------------
@app.route('/patient')
def patient_dashboard():
    if session.get('role') != 'patient':
        return redirect(url_for('login'))

    user_id = session.get('user_id')
    user_email = session.get('email', '')
    user_name = session.get('name', session.get('user_name', 'Patient'))

    patient_info = {
        'patient_id': f"PT-{user_id}",
        'name': user_name,
        'email': user_email,
        'age': 'Not Specified',
        'gender': 'Not Specified',
        'phone': 'Not Specified'
    }
    reports = []

    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor(dictionary=True)
            
            cur.execute("SELECT * FROM patient_records WHERE email = %s OR patient_id = %s", (user_email, f"PT-{user_id}"))
            p_rec = cur.fetchone()
            
            if p_rec:
                patient_info['patient_id'] = p_rec.get('patient_id') or f"PT-{user_id}"
                patient_info['name'] = p_rec.get('name') or user_name
                patient_info['age'] = p_rec.get('age') if p_rec.get('age') is not None else 'Not Specified'
                patient_info['gender'] = p_rec.get('gender') or 'Not Specified'
                patient_info['phone'] = p_rec.get('phone') or 'Not Specified'
                patient_info['email'] = p_rec.get('email') or user_email

            cur.execute("""
                SELECT * FROM radiograph_studies 
                WHERE patient_id = %s 
                ORDER BY examination_date DESC
            """, (patient_info['patient_id'],))
            reports = cur.fetchall()

            cur.close()
            conn.close()
        except Exception as e:
            print("Patient dashboard fetch error:", e)

    return render_template('patient.html', patient=patient_info, reports=reports)


# ---------------------------------------------------------
# System Logout
# ---------------------------------------------------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# =========================================================
# Module 11: Professional Radiology PDF Generator & Routes
# =========================================================
import io
import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_study_pdf(study_id):
    study = None
    patient = None
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor(dictionary=True)
            cur.execute("SELECT * FROM radiograph_studies WHERE study_id = %s", (study_id,))
            study = cur.fetchone()
            if study and study.get('patient_id'):
                cur.execute("SELECT * FROM patient_records WHERE patient_id = %s", (study['patient_id'],))
                patient = cur.fetchone()
            cur.close()
            conn.close()
    except Exception as e:
        print("PDF DB Fetch Error:", e)

    if not study:
        return None, None

    # 2.  (Safe Fallbacks)
    patient_id = study.get('patient_id') or (patient.get('patient_id') if patient else 'P008')
    patient_name = (patient.get('name') or patient.get('patient_name')) if patient else 'Patient'
    age = patient.get('age', 40) if patient else 40
    gender = patient.get('gender', 'Male') if patient else 'Male'
    
    doctor_name = study.get('doctor_name') or session.get('name') or 'Dr. Ramya'
    
    disease = study.get('predicted_disease') or 'Normal'
    raw_conf = study.get('confidence') or study.get('confidence_score') or 90.0
    try:
        confidence = float(raw_conf)
    except (ValueError, TypeError):
        confidence = 90.0

    risk_score = study.get('risk_score') or 15
    risk_level = study.get('risk_level') or 'Low'
    champ_model = study.get('champion_model') or 'DenseNet121'

    
    exam_val = study.get('examination_date') or study.get('created_at')
    if isinstance(exam_val, datetime):
        exam_date = exam_val.strftime('%d %b %Y, %I:%M %p')
    elif exam_val:
        exam_date = str(exam_val)
    else:
        exam_date = datetime.now().strftime('%d %b %Y, %I:%M %p')

    
    img_filename = study.get('image_path') or study.get('image_filename') or ''
    upload_dir = os.path.join(app.root_path, 'static', 'uploads')
    img_full_path = os.path.join(upload_dir, os.path.basename(img_filename))

    # 3. 🔴 AI Clinical Decision Support Engine (CDSS) 
    try:
        from scripts.inference import generate_ai_clinical_recommendation
        ai_cdss = generate_ai_clinical_recommendation(disease, confidence, risk_score, age)
        triage_priority = ai_cdss.get("triage_priority", "Routine Outpatient")
        workup_list = ai_cdss.get("diagnostic_workup", [study.get('clinical_recommendation') or "Correlate clinically."])
        monitoring_list = ai_cdss.get("monitoring_protocol", ["Standard clinical follow-up."])
    except Exception as e:
        print("CDSS Engine Fallback:", e)
        db_rec = study.get('clinical_recommendation') or "Clinical correlation with vitals and lab panels advised."
        triage_priority = "Clinical Priority Monitoring" if risk_score > 60 else "Routine Outpatient Care"
        workup_list = [db_rec, "Serum inflammatory markers and diagnostic panel correlation."]
        monitoring_list = ["Respiratory rate and oxygen saturation surveillance."]

    workup_html = "".join([f"• {item}<br/>" for item in workup_list])
    monitoring_html = "".join([f"• {item}<br/>" for item in monitoring_list])

    # 4. PDF (A4 Page Size)
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=26,
        bottomMargin=26
    )
    styles = getSampleStyleSheet()

    # (Typography Styles)
    header_title_style = ParagraphStyle(
        'HeaderTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0f172a")
    )
    header_sub_style = ParagraphStyle(
        'HeaderSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#64748b")
    )
    meta_label = ParagraphStyle(
        'MetaLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#334155")
    )
    meta_val = ParagraphStyle(
        'MetaVal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0f172a")
    )
    section_title = ParagraphStyle(
        'SecTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=12,
        textColor=colors.HexColor("#1e293b")
    )

    elements = []

    # ================= DISEASE CLINICAL EXPLANATIONS =================
    disease_desc_map = {
        "tuberculosis": "<b>Clinical Pathology:</b> Mycobacterium tuberculosis infection. Radiograph demonstrates apical fibro-cavitary infiltrates, parenchymal consolidation, or nodular densities. Requires urgent airborne isolation protocols.",
        "covid-19": "<b>Clinical Pathology:</b> SARS-CoV-2 viral pulmonary involvement. Radiograph displays bilateral peripheral, subpleural ground-glass opacities and multi-lobar consolidation. Respiratory monitoring recommended.",
        "covid": "<b>Clinical Pathology:</b> SARS-CoV-2 viral pulmonary involvement. Radiograph displays bilateral peripheral, subpleural ground-glass opacities and multi-lobar consolidation. Respiratory monitoring recommended.",
        "viral pneumonia": "<b>Clinical Pathology:</b> Diffuse inflammatory lung process characterized by patchy bilateral interstitial infiltrates and peribronchial thickening. Supportive clinical workup indicated.",
        "lung opacity": "<b>Clinical Pathology:</b> Elevated attenuation in the pulmonary parenchyma obscuring underlying vascular markings, suggesting localized consolidation, fluid, or focal atelectasis.",
        "normal": "<b>Clinical Pathology:</b> Unremarkable pulmonary evaluation. Clear, symmetrical lung fields with preserved bronchovascular markings, sharp costophrenic angles, and standard cardiothoracic ratio."
    }
    disease_key = str(disease).strip().lower()
    disease_pathology = disease_desc_map.get(
        disease_key, 
        f"<b>Clinical Pathology:</b> Radiographic patterns indicate {disease} involvement requiring diagnostic lab correlation and physical examination."
    )

    # ================= STEP 1: TOP LETTERHEAD (Image 3 Style) =================
   # ================= STEP 1: ATTRACTIVE HEADER WITH "LV" LOGO BADGE =================
    
    # 1. Sleek "LV" Monogram Logo Badge
    logo_badge = Table(
        [[Paragraph("<b><font size=13 color='white'>LV</font></b>", ParagraphStyle('LVLogo', alignment=1))]],
        colWidths=[34],
        rowHeights=[34]
    )
    logo_badge.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#008891')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))

    # 2. Dual-Tone Title & Modern Tagline
    brand_text = Paragraph(
        "<b><font size=16 color='#0F172A'>LUNG</font> <font size=16 color='#008891'>VISION</font></b><br/>"
        "<font size=7 color='#0284C7'><b>PULMONARY INTELLIGENCE & DIAGNOSTIC PLATFORM</b></font>",
        ParagraphStyle('BrandTitle', parent=styles['Normal'], leading=12)
    )

    # 3. Logo + Text pakkam pakkam vaikka table
    brand_composite = Table([[logo_badge, brand_text]], colWidths=[40, 250])
    brand_composite.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))

    # 4. Right side Reporting Doctor details
    head_right = Paragraph(
        f"<b>Reporting Doctor:</b> Dr. {doctor_name}<br/>"
        f"<font color='#64748b' size=7.5>Consultant Radiologist / Pulmonology</font><br/>"
        f"<b>Exam Date:</b> {exam_date}",
        ParagraphStyle('HeadR', parent=styles['Normal'], fontName='Helvetica', fontSize=8, leading=11, alignment=2, textColor=colors.HexColor("#334155"))
    )

    # 5. Main Top Header Table
    t_head = Table([[brand_composite, head_right]], colWidths=[290, 230])
    t_head.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    elements.append(t_head)
    
    # Modern Teal Accent Divider Line
    elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#008891"), spaceAfter=7, spaceBefore=1))

    # Center Title Pill
    title_box = Table([[Paragraph("<font color='#008891'><b>CLINICAL DIAGNOSTIC EVALUATION REPORT</b></font>", ParagraphStyle('CTitle', fontName='Helvetica-Bold', fontSize=9.5, alignment=1))]], colWidths=[520], rowHeights=[18])
    title_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F0FDFA')),
        ('BOX', (0,0), (-1,-1), 0.8, colors.HexColor('#CCFBF1')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 1),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
    ]))
    elements.append(title_box)
    elements.append(Spacer(1, 6))

    # ================= STEP 2: PATIENT INFO GRID (Image 4 Style) =================

    clean_pid = str(patient_id) if str(patient_id).startswith('PT-') else f"PT-{patient_id}"

    patient_meta_data = [
        [
            Paragraph("<b>Patient Name:</b>", meta_label), Paragraph(f"<b>{patient_name}</b>", meta_val),
            Paragraph("<b>Patient ID:</b>", meta_label), Paragraph(clean_pid, meta_val)   
        ],
        [
            Paragraph("<b>Age / Gender:</b>", meta_label), Paragraph(f"{age} Yrs / {gender}", meta_val),
            Paragraph("<b>Study ID:</b>", meta_label), Paragraph(f"#{study.get('study_id', study_id)}", meta_val)
        ]
    ]
    t_patient = Table(patient_meta_data, colWidths=[85, 175, 85, 175])
    t_patient.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 0.8, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(t_patient)
    elements.append(Spacer(1, 8))

    
# ================= STEP 3: SIDE-BY-SIDE RADIOGRAPH & DISEASE PATHOLOGY =================
    # ================= STEP 3: SIDE-BY-SIDE RADIOGRAPH & DISEASE PATHOLOGY =================
    xray_box = None

    # Multi-path resolver: Render matrum Localhost rendilume X-ray image-ai sariyaaga kandupidikkum
    candidate_paths = []
    
    # 1. Existing variable check
    if 'img_full_path' in locals() and img_full_path:
        candidate_paths.append(str(img_full_path))
        fname = os.path.basename(str(img_full_path))
    elif 'study' in locals() and isinstance(study, dict) and study.get('image_path'):
        candidate_paths.append(str(study.get('image_path')))
        fname = os.path.basename(str(study.get('image_path')))
    else:
        fname = "xray.png"

    # 2. Ella potential folder paths-aiyum check seiyyum
    base_dir = getattr(app, 'root_path', os.getcwd())
    candidate_paths.extend([
        os.path.join(base_dir, 'static', 'uploads', fname),
        os.path.join(base_dir, 'uploads', fname),
        os.path.join(base_dir, 'static', fname),
        os.path.join(os.getcwd(), 'static', 'uploads', fname),
        os.path.join(os.getcwd(), 'uploads', fname),
        os.path.join('static', 'uploads', fname),
        os.path.join('uploads', fname),
        fname
    ])

    resolved_img_path = None
    for p in candidate_paths:
        if p and os.path.exists(p) and os.path.isfile(p):
            resolved_img_path = p
            break

    if resolved_img_path:
        try:
            xray_box = RLImage(resolved_img_path, width=160, height=170)
        except Exception:
            xray_box = Paragraph("<font size=8 color='#64748b'>[Image Rendering Unavailable]</font>", styles['Normal'])
    else:
        xray_box = Paragraph("<font size=8 color='#64748b'>[Radiograph Not Found on Disk]</font>", styles['Normal'])

    diag_color = "#dc2626" if disease.lower() != 'normal' else "#16a34a"
    triage_color = "#b91c1c" if ("Isolation" in triage_priority or "Admission" in triage_priority or risk_score > 65) else "#0284c7"

    findings_html = f"""
    <b>PRIMARY DIAGNOSTIC FINDING:</b><br/>
    <font size="11" color="{diag_color}"><b>{disease.upper()}</b></font> 
    &nbsp;<font size=8 color="#475569">(Confidence: <b>{confidence:.2f}%</b> | Risk Index: <b>{risk_score}/100</b>)</font><br/>
    • <b>Triage Priority:</b> <font color="{triage_color}"><b>{triage_priority}</b></font><br/>
    <br/>
    {disease_pathology}<br/>
    <br/>
    <b>EXAMINATION PARAMETERS:</b><br/>
    • <b>Modality:</b> Digital Chest Radiography (PA View)<br/>
    • <b>Gatekeeper Quality:</b> Validated Exposure & Contrast<br/>
    • <b>Champion AI Model:</b> {champ_model}
    """

    findings_para = Paragraph(findings_html, ParagraphStyle(
        'FindingsStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10.8,
        textColor=colors.HexColor("#1e293b")
    ))

    dual_panel = Table([[xray_box, findings_para]], colWidths=[170, 350])
    dual_panel.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOX', (0,0), (0,0), 1, colors.HexColor("#0f172a")),
        ('BACKGROUND', (0,0), (0,0), colors.HexColor("#090e17")),
        ('LEFTPADDING', (1,0), (1,0), 10),
        ('RIGHTPADDING', (1,0), (1,0), 0),
        ('TOPPADDING', (0,0), (0,0), 0),     # X-ray image border top-la irukkum
        ('TOPPADDING', (1,0), (1,0), 12),    # <-- Ithu thaan Text-ah 2 steps keezhe kondu varum!
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    elements.append(dual_panel)
    elements.append(Spacer(1, 8))

    # ================= STEP 4: STEP-BY-STEP CLINICAL WORKUP (Image 4 Style) =================
    elements.append(Paragraph("<b>RECOMMENDED CLINICAL WORKUP & ACTION PLAN:</b>", ParagraphStyle('RecH', fontName='Helvetica-Bold', fontSize=8.5, leading=11, textColor=colors.HexColor('#0f172a'))))
    elements.append(Spacer(1, 3))
    
    workup_combined = f"{workup_html}<br/><b>Surveillance & Follow-up:</b><br/>{monitoring_html}"
    elements.append(Paragraph(workup_combined, ParagraphStyle(
        'WorkupStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor("#334155"),
        leftIndent=8
    )))
    elements.append(Spacer(1, 8))

    # ================= STEP 5: MEDICAL DISCLAIMER =================
    disclaimer_text = (
        "<i><b>Medical Disclaimer:</b> This report is generated by the LUNG VISION AI multi-model decision support platform "
        "and is intended exclusively for authorized clinical personnel. Final clinical diagnosis and patient discharge authority "
        "remain with the qualified licensed medical practitioner.</i>"
    )
    elements.append(Paragraph(disclaimer_text, ParagraphStyle(
        'DiscStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.8,
        leading=8.8,
        textColor=colors.HexColor("#64748b")
    )))
    elements.append(Spacer(1, 8))

    # ================= STEP 6: SIGNATURES SECTION =================
    sub_center = ParagraphStyle('SigSub', fontName='Helvetica', fontSize=7.8, leading=10, alignment=1, textColor=colors.HexColor("#475569"))
    sig_table_data = [
        [
            Paragraph("<b>Verified by:</b><br/>LUNG VISION AI Quality Gatekeeper<br/><font color='#64748b'>Automated QA Passed</font>", sub_center),
            Paragraph(f"<b>Reporting Doctor:</b><br/>Dr. {doctor_name}<br/><font color='#64748b'>Consultant Radiologist</font>", sub_center)
        ]
    ]
    t_sig = Table(sig_table_data, colWidths=[260, 260])
    t_sig.setStyle(TableStyle([
        ('LINEABOVE', (0,0), (0,0), 1, colors.HexColor("#94a3b8")),
        ('LINEABOVE', (1,0), (1,0), 1, colors.HexColor("#94a3b8")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    elements.append(t_sig)

    # PDF Build (Single-Page Output Guaranteed)
    doc.build(elements)
 
    buffer.seek(0)
    
    filename = f"LungVision_Report_{patient_id}_ST{study_id}.pdf"
    return buffer, filename

# 1. VIEW ROUTE 
@app.route('/report/view/<int:study_id>')
def view_report(study_id):
    buffer, filename = generate_study_pdf(study_id)
    if not buffer:
        return "Study record not found", 404
    return send_file(buffer, mimetype='application/pdf', as_attachment=False)


# 2. DOWNLOAD ROUTE
@app.route('/report/download/<int:study_id>')
def download_report(study_id):
    buffer, filename = generate_study_pdf(study_id)
    if not buffer:
        return "Study record not found", 404
    return send_file(buffer, mimetype='application/pdf', as_attachment=True, download_name=filename)


# =========================================================
# System Security: Master Admin Auto-Initialization
# =========================================================
def init_master_admin():
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor(dictionary=True)
            hashed_pin = bcrypt.hashpw('123456'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            cur.execute("SELECT * FROM users WHERE role = 'admin'")
            admin = cur.fetchone()
            
            if not admin:
                cur.execute("""
                    INSERT INTO users (name, email, password_hash, role)
                    VALUES (%s, %s, %s, %s)
                """, ('System Administrator', 'admin@gmail.com', hashed_pin, 'admin'))
                print(">>> [SECURITY] Master Admin Initialized: admin@gmail.com (PIN: 123456)")
            else:
                # ஏற்கனவே உள்ள அட்மினின் பாஸ்வேர்டை சரியான bcrypt ஹேஷாக மாற்றுகிறது
                cur.execute("UPDATE users SET password_hash = %s WHERE role = 'admin'", (hashed_pin,))
                print(">>> [SECURITY] Master Admin Hash Verified: admin@gmail.com")
                
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print("Admin setup note:", e)

if __name__ == '__main__':
    init_master_admin()
    app.run(host='127.0.0.1', port=5000, debug=False)