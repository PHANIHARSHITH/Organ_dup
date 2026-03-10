
from flask import Flask, request, render_template
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler # Import necessary scikit-learn classes

app = Flask(__name__)

# --- Model Loading Configuration ---
MODELS = {}
MODEL_CONFIG = {
    "heart": {"path": "pickel_files/heart_transplant.sav"},
    "kidney": {
        "path": "pickel_files/kidney/kidney_transplant.sav",
        "label_encoder_path": "pickel_files/kidney/label_encoders.pkl",
        "scaler_path": "pickel_files/kidney/scaler.pkl"
    },
    "liver": {"path": "pickel_files/liver_transplant.sav"},
    "lung": {"path": "pickel_files/lung_transplant.sav"},
}

def load_models():
    """Loads all specified pickle models and preprocessors into memory at startup."""
    print("Loading models...")
    for model_name, config in MODEL_CONFIG.items():
        model_path = config["path"]
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    MODELS[model_name] = pickle.load(f)
                print(f"-> Successfully loaded '{model_name}' model.")
                
                if model_name == "kidney":
                    # Load associated preprocessors for kidney
                    le_path = config.get("label_encoder_path")
                    scaler_path = config.get("scaler_path")
                    
                    if le_path and os.path.exists(le_path):
                        with open(le_path, 'rb') as f:
                            MODELS[f"{model_name}_label_encoders"] = pickle.load(f)
                        print(f"-> Successfully loaded '{model_name}' label encoders.")
                    else:
                        print(f"-> WARNING: Label encoder file not found for '{model_name}' at {le_path}.")
                        MODELS[f"{model_name}_label_encoders"] = None

                    if scaler_path and os.path.exists(scaler_path):
                        with open(scaler_path, 'rb') as f:
                            MODELS[f"{model_name}_scaler"] = pickle.load(f)
                        print(f"-> Successfully loaded '{model_name}' scaler.")
                    else:
                        print(f"-> WARNING: Scaler file not found for '{model_name}' at {scaler_path}.")
                        MODELS[f"{model_name}_scaler"] = None

            except Exception as e:
                print(f"-> ERROR loading '{model_name}' model or its components from {model_path}: {e}")
                MODELS[model_name] = None
        else:
            print(f"-> WARNING: Model file not found for '{model_name}' at {model_path}.")
            MODELS[model_name] = None

# Load models when the Flask app starts
with app.app_context():
    load_models()

# --- Frontend Rendering Routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/heart")
def heart_form():
    return render_template("heart.html")

@app.route("/kidney")
def kidney_form():
    return render_template("kidney.html")

@app.route("/liver")
def liver_form():
    return render_template("liver.html")

@app.route("/lung")
def lung_form():
    return render_template("lung.html")

@app.route("/loading")
def loading_page():
    return render_template("loading.html")

@app.route("/result")
def result_page():
    return render_template("result.html", result="No prediction data received.", organ="System")


# --- Prediction API Routes ---

@app.route("/predict/heart", methods=["POST"])
def predict_heart():
    organ = "Heart"
    print(f"DEBUG: MODELS dictionary keys: {list(MODELS.keys())}")
    print(f"DEBUG: MODELS['heart'] value: {MODELS.get('heart')}")
    model = MODELS.get("heart")
    if not model:
        print(f"ERROR: Heart model is None or missing!")
        return render_template("result.html", result="Heart model is not available.", organ=organ)
        
    try:
        # --- Feature Extraction & Preprocessing (Target: 41 features) ---
        
        # 1. Create a DataFrame from form data
        form_data = request.form.to_dict()
        input_df = pd.DataFrame([form_data])

        # 2. Define expected categorical columns and their possible values
        # This ensures the one-hot encoded DataFrame has a consistent structure
        categorical_cols = {
            'diagnosis': ['CONGENITAL', 'FAILED OHT', 'HCM', 'ICM', 'NICM', 'OTHER/UNKNOWN', 'RESTRICTIVE', 'VALVULAR'],
            'mcs': ['ECMO', 'IABP', 'bivad/tah', 'dischargeable VAD', 'left endo device', 'non-dischargeable VAD', 'none', 'right endo device'],
            'abo': ['A', 'AB', 'B', 'O'],
            'CODDON': ['Anoxia/Asphyx', 'Cardiovascular', 'Drowning', 'Drug Intoxication', 'IntracranHem/Stroke/Seiz', 'Natural Causes', 'Trauma'],
            'HIST_MI': ['No', 'Yes'],
            'diabetes': ['No', 'Yes']
        }

        # 3. Process numerical features
        numerical_features = [
            'AGE', 'AGE_DON', 'CREAT_TRR', 'CREAT_DON', 'BMI_CALC', 'BMI_DON_CALC',
            'DAYSWAIT_CHRON', 'medcondition', 'ABOMAT', 'DISTANCE', 'TX_YEAR'
        ]
        for col in numerical_features:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

        # 4. One-hot encode categorical features
        processed_dfs = [input_df[numerical_features]]
        for col, categories in categorical_cols.items():
            # Set the column as a categorical type with all possible categories
            input_df[col] = pd.Categorical(input_df[col], categories=categories)
            # Create dummy variables
            dummies = pd.get_dummies(input_df[col], prefix=col, drop_first=(col == 'HIST_MI')) # drop_first to match 41 features
            processed_dfs.append(dummies)
        
        # 5. Combine all features
        final_df = pd.concat(processed_dfs, axis=1)
        
        # Ensure final feature vector has exactly 41 features, adding zeros for any missing
        # This is a safeguard if a category was missed.
        expected_feature_count = 41
        final_features = final_df.to_numpy().flatten()
        if len(final_features) < expected_feature_count:
            final_features = np.pad(final_features, (0, expected_feature_count - len(final_features)))
        
        features = [final_features[:expected_feature_count]]
        
        # --- Prediction ---
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] # Probability of "MATCH"
        result_text = "MATCH" if prediction == 1 else "NO MATCH"
        
        return render_template("result.html", result=result_text, probability=probability, organ=organ)

    except Exception as e:
        print(f"ERROR during heart prediction: {e}")
        return render_template("result.html", result=f"Error processing data: {e}", organ=organ)


@app.route("/predict/kidney", methods=["POST"])
def predict_kidney():
    organ = "Kidney"
    model = MODELS.get("kidney")
    label_encoders = MODELS.get("kidney_label_encoders")
    scaler = MODELS.get("kidney_scaler")

    if not model or not label_encoders or not scaler:
        missing = [name for name, obj in [("model", model), ("encoders", label_encoders), ("scaler", scaler)] if not obj]
        return render_template("result.html", result=f"Kidney {', '.join(missing)} not available.", organ=organ)
    
    try:
        # --- Feature Extraction & Preprocessing (Target: 36 features) ---
        form_data = request.form.to_dict()
        
        # Define which columns are categorical and numerical
        categorical_cols = [
            'recipient_sex', 'recipient_blood_type', 'recipient_rh', 'recipient_cmv', 'recipient_ebv', 
            'donor_type', 'donor_blood_type', 'donor_rh', 'crossmatch_result'
        ]
        numerical_cols = [
            'recipient_age', 'recipient_years_on_dialysis', 'recipient_creatinine', 
            'recipient_egfr', 'recipient_pra', 'donor_age', 'donor_bmi', 'donor_cold_ischemia_time'
        ]
        
        feature_dict = {}
        # Process categorical features with label encoders
        for col in categorical_cols:
            le = label_encoders.get(col)
            val = form_data.get(col, '')
            if le:
                # Handle unseen labels by defaulting to 0
                try:
                    feature_dict[col] = le.transform([val])[0]
                except ValueError:
                    feature_dict[col] = 0 
            else:
                feature_dict[col] = 0

        # Process numerical features
        for col in numerical_cols:
            feature_dict[col] = float(form_data.get(col) or 0)
        
        # Separate numeric and categorical values so scaling only applies to numbers
        num_values = [feature_dict[col] for col in numerical_cols]
        cat_values = [feature_dict[col] for col in categorical_cols]

        # scale numeric portion; scaler was trained on the pure numeric set
        expected_numeric = getattr(scaler, 'n_features_in_', None)
        if expected_numeric is not None and len(num_values) != expected_numeric:
            print(f"WARNING: Kidney scaler expects {expected_numeric} numeric features but received {len(num_values)}. Adjusting input array.")
            if len(num_values) > expected_numeric:
                num_values = num_values[:expected_numeric]
            else:
                num_values = num_values + [0] * (expected_numeric - len(num_values))

        scaled_nums = scaler.transform([num_values])[0]

        # Reassemble the full feature vector (numeric followed by categorical)
        features_list = list(scaled_nums) + cat_values

        # The RandomForest model was trained on 36 features; pad/truncate as needed.
        # Our current form provides only a subset, so zeros fill the rest.
        expected_feature_count = 36
        if len(features_list) < expected_feature_count:
            features_list.extend([0] * (expected_feature_count - len(features_list)))
        elif len(features_list) > expected_feature_count:
            features_list = features_list[:expected_feature_count]

        features = np.array(features_list).reshape(1, -1)
        
        # already scaled numeric parts; use features directly
        scaled_features = features

        # --- Prediction ---
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]
        result_text = "MATCH" if prediction == 1 else "NO MATCH"
        
        return render_template("result.html", result=result_text, probability=probability, organ=organ)
    except Exception as e:
        print(f"ERROR during kidney prediction: {e}")
        return render_template("result.html", result=f"Error processing data: {e}", organ=organ)


@app.route("/predict/liver", methods=["POST"])
def predict_liver():
    organ = "Liver"
    model = MODELS.get("liver")
    if not model:
        return render_template("result.html", result="Liver model is not available.", organ=organ)
    
    try:
        # --- Feature Extraction & Preprocessing (Target: 14 features) ---
        form_data = request.form.to_dict()
        input_df = pd.DataFrame([form_data])

        # Define categories for blood type to one-hot encode
        blood_types = ['O+', 'O-', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-']
        input_df['blood_type'] = pd.Categorical(input_df['blood_type'], categories=blood_types)
        blood_type_dummies = pd.get_dummies(input_df['blood_type'], prefix='blood')

        # Process numerical features
        numerical_cols = ['age', 'bmi', 'meld_score', 'bilirubin', 'inr', 'creatinine']
        for col in numerical_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

        # Combine features
        final_df = pd.concat([input_df[numerical_cols], blood_type_dummies], axis=1)
        features = [final_df.to_numpy().flatten()]

        # --- Prediction ---
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        result_text = "MATCH" if prediction == 1 else "NO MATCH"
        
        return render_template("result.html", result=result_text, probability=probability, organ=organ)
    except Exception as e:
        print(f"ERROR during liver prediction: {e}")
        return render_template("result.html", result=f"Error processing data: {e}", organ=organ)


@app.route("/predict/lung", methods=["POST"])
def predict_lung():
    organ = "Lung"
    model = MODELS.get("lung")
    if not model:
        return render_template("result.html", result="Lung model is not available.", organ=organ)
        
    try:
        # --- Feature Extraction & Preprocessing (Target: 11 features) ---
        form_data = request.form.to_dict()
        input_df = pd.DataFrame([form_data])

        # Define categories for one-hot encoding
        diagnoses = ['COPD', 'IPF', 'CF', 'PAH', 'Other']
        input_df['primary_diagnosis'] = pd.Categorical(input_df['primary_diagnosis'], categories=diagnoses)
        diagnosis_dummies = pd.get_dummies(input_df['primary_diagnosis'], prefix='diag')

        # Encode binary feature
        input_df['smoking_history'] = input_df['smoking_history'].apply(lambda x: 1 if x == 'Yes' else 0)

        # Process numerical features
        numerical_cols = ['age', 'bmi', 'fev1', 'oxygen_requirement', 'pulmonary_pressure']
        for col in numerical_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
            
        # Combine all features
        final_df = pd.concat([
            input_df[numerical_cols],
            input_df[['smoking_history']],
            diagnosis_dummies
        ], axis=1)
        features = [final_df.to_numpy().flatten()]

        # --- Prediction ---
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        result_text = "MATCH" if prediction == 1 else "NO MATCH"

        return render_template("result.html", result=result_text, probability=probability, organ=organ)
    except Exception as e:
        print(f"ERROR during lung prediction: {e}")
        return render_template("result.html", result=f"Error processing data: {e}", organ=organ)

if __name__ == "__main__":
    app.run(debug=True)
