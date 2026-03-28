from flask import Flask, request, render_template
import pickle
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler 


def _repair_random_forest_for_sklearn(model):
    if model is None:
        return
    for est in getattr(model, "estimators_", []):
            est.monotonic_cst = None


app = Flask(__name__)

MODELS = {}
MODEL_CONFIG = {
    "heart": {"path": "pickel_files/heart_transplant.sav"},
    "kidney": {
        "path": "pickel_files/kidney/kidney_transplant.sav",
        "label_encoder_path": "pickel_files/kidney/label_encoders.pkl",
        "scaler_path": "pickel_files/kidney/scaler.pkl"
    },
    "liver": {
        "path": "",
        "meta_path": "pickel_files/liver/liver_model_meta.pkl"
    },
    "lung": {"path": "pickel_files/lung_transplant.sav"},
}

def _apply_sklearn_compat_patches():
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, '_RemainderColsList'):
        class _RemainderColsList(list): pass
        _ct._RemainderColsList = _RemainderColsList

    from sklearn.impute import _base as _ib
    if not hasattr(_ib.SimpleImputer, '_patched_fill_dtype'):
        _orig_transform = _ib.SimpleImputer.transform
        def _patched_transform(self, X, **kw):
            if not hasattr(self, '_fill_dtype') and hasattr(self, '_fit_dtype'):
                self._fill_dtype = self._fit_dtype
            return _orig_transform(self, X, **kw)
        _ib.SimpleImputer.transform = _patched_transform
        _ib.SimpleImputer._patched_fill_dtype = True


def load_models():
    """Loads all specified pickle models and preprocessors into memory at startup."""
    _apply_sklearn_compat_patches()
    print("Loading models...")
    for model_name, config in MODEL_CONFIG.items():
        model_path = config["path"]
        if model_name == "liver":
            meta_path = config.get("meta_path", "")
            if meta_path and os.path.exists(meta_path):
                try:
                    MODELS["liver_meta"] = joblib.load(meta_path)
                except:
                    with open(meta_path, 'rb') as f:
                        MODELS["liver_meta"] = pickle.load(f)
                MODELS["liver"] = MODELS["liver_meta"].get("model")
                print("-> Successfully loaded liver model from metadata bundle.")
            else:
                print(f"-> WARNING: Liver meta file not found at {meta_path}.")
                MODELS["liver_meta"] = None
                MODELS["liver"] = None
            continue
        if os.path.exists(model_path):
            try:
                try:
                    MODELS[model_name] = joblib.load(model_path)
                    print(f"-> Successfully loaded '{model_name}' model with joblib.")
                except Exception as e:
                    print(f"-> Joblib load failed, trying pickle: {e}")
                    with open(model_path, 'rb') as f:
                        MODELS[model_name] = pickle.load(f)
                    print(f"-> Successfully loaded '{model_name}' model with pickle.")

                if model_name == "kidney":
                    _repair_random_forest_for_sklearn(MODELS.get(model_name))
                    le_path     = config.get("label_encoder_path")
                    scaler_path = config.get("scaler_path")
                    if le_path and os.path.exists(le_path):
                        try:    MODELS["kidney_label_encoders"] = joblib.load(le_path)
                        except: 
                            with open(le_path, 'rb') as f: MODELS["kidney_label_encoders"] = pickle.load(f)
                        print("-> Successfully loaded 'kidney' label encoders.")
                    else:
                        print(f"-> WARNING: Label encoder not found at {le_path}.")
                        MODELS["kidney_label_encoders"] = None
                    if scaler_path and os.path.exists(scaler_path):
                        try:    MODELS["kidney_scaler"] = joblib.load(scaler_path)
                        except:
                            with open(scaler_path, 'rb') as f: MODELS["kidney_scaler"] = pickle.load(f)
                        print("-> Successfully loaded 'kidney' scaler.")
                    else:
                        print(f"-> WARNING: Scaler not found at {scaler_path}.")
                        MODELS["kidney_scaler"] = None

                elif model_name == "liver":
                    meta_path = config.get("meta_path", "")
                    if meta_path and os.path.exists(meta_path):
                        try:    MODELS["liver_meta"] = joblib.load(meta_path)
                        except:
                            with open(meta_path, 'rb') as f: MODELS["liver_meta"] = pickle.load(f)
                        MODELS["liver"] = MODELS["liver_meta"].get("model")
                        print("-> Successfully loaded liver model metadata.")
                    else:
                        print(f"-> WARNING: Liver meta file not found at {meta_path}.")
                        MODELS["liver_meta"] = None
                        MODELS["liver"] = None

            except Exception as e:
                print(f"-> ERROR loading '{model_name}': {e}")
                MODELS[model_name] = None
        else:
            print(f"-> WARNING: Model file not found for '{model_name}' at {model_path}.")
            MODELS[model_name] = None

with app.app_context():
    load_models()

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
        
        form_data = request.form.to_dict()
        input_df = pd.DataFrame([form_data])

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
           
            input_df[col] = pd.Categorical(input_df[col], categories=categories)
            
            dummies = pd.get_dummies(input_df[col], prefix=col, drop_first=(col == 'HIST_MI')) 
            processed_dfs.append(dummies)
        
 
        final_df = pd.concat(processed_dfs, axis=1)
        
        
        expected_feature_count = 41
        final_features = final_df.to_numpy().flatten()
        if len(final_features) < expected_feature_count:
            final_features = np.pad(final_features, (0, expected_feature_count - len(final_features)))
        
        features = [final_features[:expected_feature_count]]
        
       
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] 
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

    # repair again in case someone replaced the model object after startup
    _repair_random_forest_for_sklearn(model)

    if not model or not label_encoders or not scaler:
        missing = [name for name, obj in [("model", model), ("encoders", label_encoders), ("scaler", scaler)] if not obj]
        return render_template("result.html", result=f"Kidney {', '.join(missing)} not available.", organ=organ)
    
    try:
        
        form_data = request.form.to_dict()

        # original dataset column order used during training
        expected_columns = [
            'Donor_ID', 'Donor_Age', 'Donor_Gender', 'Donor_Blood_Type',
            'Donor_HLA_A', 'Donor_HLA_B', 'Donor_HLA_DR', 'Donor_Creatinine_Level',
            'Donor_BMI', 'Donor_Medical_History', 'Recipient_ID', 'Recipient_Age',
            'Recipient_Gender', 'Recipient_Blood_Type', 'Recipient_HLA_A',
            'Recipient_HLA_B', 'Recipient_HLA_DR', 'Recipient_Creatinine_Level',
            'Recipient_BMI', 'Recipient_Urgency_Level', 'Recipient_Medical_History',
            'Compatibility_Score'
        ]

        # initialise with zeros
        feature_dict = {col: 0 for col in expected_columns}
        def get_num(key):
            try:
                return float(form_data.get(key) or 0)
            except Exception:
                return 0.0

        # fill numeric values from form where available
        mapping = {
            'donor_age': 'Donor_Age',
            'donor_bmi': 'Donor_BMI',
            'recipient_age': 'Recipient_Age',
            'recipient_creatinine': 'Recipient_Creatinine_Level',
            # other numeric features from train (donor_creatinine, Recipient_BMI,
            # Compatibility_Score) are not collected in the form and remain 0
        }
        for form_k, col_n in mapping.items():
            feature_dict[col_n] = get_num(form_k)

        # encode categorical columns using stored label encoders (preserves order)
        for col_name, le in label_encoders.items():
            form_key = col_name.lower()
            val = form_data.get(form_key, '')
            if le and val:
                try:
                    feature_dict[col_name] = le.transform([str(val)])[0]
                except Exception:
                    feature_dict[col_name] = 0
            else:
                feature_dict[col_name] = 0

        # scale numeric columns in place
        numeric_cols = list(getattr(scaler, 'feature_names_in_', []))
        if numeric_cols:
            raw_nums = [feature_dict.get(col, 0) for col in numeric_cols]
            scaled_nums = scaler.transform([raw_nums])[0]
            for i, col in enumerate(numeric_cols):
                feature_dict[col] = scaled_nums[i]

        # final feature vector in the same column order used during training
        features = np.array([feature_dict[col] for col in expected_columns]).reshape(1, -1)
        scaled_features = features

        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]
        # protect against corrupted/inconsistent models by clipping probability
        probability = float(np.clip(probability, 0.0, 1.0))
        result_text = "MATCH" if prediction == 1 else "NO MATCH"
        
        return render_template("result.html", result=result_text, probability=probability, organ=organ)
    except Exception as e:
        print(f"ERROR during kidney prediction: {e}")
        return render_template("result.html", result=f"Error processing data: {e}", organ=organ)


@app.route("/predict/liver", methods=["POST"])
def predict_liver():
    organ = "Liver"
    model = MODELS.get("liver")
    meta  = MODELS.get("liver_meta")

    if not model:
        return render_template("result.html", result="Liver model is not available.", organ=organ)
    if not meta:
        return render_template("result.html", result="Liver model metadata not available.", organ=organ)

    try:
        form_data      = request.form.to_dict()
        feature_names  = meta["feature_names"]   # exact training column order (31 features)
        num_cols       = meta["num_cols"]
        label_encoders = meta["label_encoders"]
        num_imputer    = meta.get("num_imputer")  # if needed for missing numeric values

        feature_list = []
        for col in feature_names:
            val = form_data.get(col, "").strip()
            if col in num_cols:
                try:    feature_list.append(float(val) if val else np.nan)
                except: feature_list.append(np.nan)
            else:
                le = label_encoders.get(col)
                if le and val:
                    try:    feature_list.append(int(le.transform([val])[0]))
                    except: feature_list.append(0)
                else:
                    feature_list.append(0)

        features    = np.array(feature_list, dtype=float).reshape(1, -1)
        if num_imputer:
            num_indices = [feature_names.index(c) for c in num_cols if c in feature_names]
            features[0, num_indices] = num_imputer.transform(
                features[:, num_indices]
            )[0]
        prediction  = model.predict(features)[0]
        probability = float(np.clip(model.predict_proba(features)[0][1], 0.0, 1.0))
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
        form_data = request.form.to_dict()

        def get_float(key, default=0.0):
            try:    return float(form_data.get(key, default) or default)
            except: return default

        # ── Computed fields ───────────────────────────────────────────────────
        donor_age     = get_float("Donor_Age")
        recipient_age = get_float("Recipient_Age")
        donor_cap     = get_float("Donor_Lung_Capacity")
        recipient_cap = get_float("Recipient_Lung_Capacity")

        hla_a  = form_data.get("Donor_HLA_A",  "")
        hla_b  = form_data.get("Donor_HLA_B",  "")
        hla_dr = form_data.get("Donor_HLA_DR", "")

        hla_a_match  = 1 if hla_a  == form_data.get("Recipient_HLA_A",  "") else 0
        hla_b_match  = 1 if hla_b  == form_data.get("Recipient_HLA_B",  "") else 0
        hla_dr_match = 1 if hla_dr == form_data.get("Recipient_HLA_DR", "") else 0
        hla_total    = hla_a_match + hla_b_match + hla_dr_match

        d_bt = form_data.get("Donor_Blood_Type",     "")
        r_bt = form_data.get("Recipient_Blood_Type", "")
        blood_type_match = 1 if d_bt == r_bt else 0

        age_diff     = abs(donor_age - recipient_age)
        lung_cap_diff = abs(donor_cap - recipient_cap)

        # ── Build DataFrame in exact training column order (26 cols) ──────────
        row = {
            "Donor_Age":              donor_age,
            "Donor_Gender":           form_data.get("Donor_Gender", ""),
            "Donor_Blood_Type":       d_bt,
            "Donor_HLA_A":            hla_a,
            "Donor_HLA_B":            hla_b,
            "Donor_HLA_DR":           hla_dr,
            "Donor_Smoking_History":  form_data.get("Donor_Smoking_History", ""),
            "Donor_Medical_History":  form_data.get("Donor_Medical_History", ""),
            "Donor_Lung_Capacity":    donor_cap,
            "Recipient_Age":          recipient_age,
            "Recipient_Gender":       form_data.get("Recipient_Gender", ""),
            "Recipient_Blood_Type":   r_bt,
            "Recipient_HLA_A":        form_data.get("Recipient_HLA_A", ""),
            "Recipient_HLA_B":        form_data.get("Recipient_HLA_B", ""),
            "Recipient_HLA_DR":       form_data.get("Recipient_HLA_DR", ""),
            "Recipient_Medical_History":  form_data.get("Recipient_Medical_History", ""),
            "Recipient_Oxygen_Support":   form_data.get("Recipient_Oxygen_Support", ""),
            "Recipient_Lung_Capacity":    recipient_cap,
            "Recipient_Urgency_Level":    form_data.get("Recipient_Urgency_Level", ""),
            "HLA_A_Match":            hla_a_match,
            "HLA_B_Match":            hla_b_match,
            "HLA_DR_Match":           hla_dr_match,
            "HLA_Total_Match":        hla_total,
            "Blood_Type_Match":       blood_type_match,
            "Age_Diff":               age_diff,
            "Lung_Cap_Diff":          lung_cap_diff,
        }
        input_df = pd.DataFrame([row])

        prediction  = model.predict(input_df)[0]
        probability = float(np.clip(model.predict_proba(input_df)[0][1], 0.0, 1.0))
        result_text = "MATCH" if prediction == 1 else "NO MATCH"

        return render_template("result.html", result=result_text, probability=probability, organ=organ)
    except Exception as e:
        print(f"ERROR during lung prediction: {e}")
        return render_template("result.html", result=f"Error processing data: {e}", organ=organ)

if __name__ == "__main__":
    app.run(debug=True)