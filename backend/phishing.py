#!/usr/bin/env python3
"""
Enhanced train_models_full.py with advanced feature engineering for >90% accuracy

Usage:
  python train_models_full.py --csv malicious_phish.csv --max-samples 8000

Set --max-samples to an integer to run a stratified sample (fast).
Set --max-samples to 0 or omit to process the entire dataset (may be slow).
"""

import argparse, os, json, time, re
from urllib.parse import urlparse
import pandas as pd, numpy as np
from collections import Counter
import string
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, BaggingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
from pathlib import Path

# optional packages
try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb

    LGB_AVAILABLE = True
except:
    LGB_AVAILABLE = False

try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False

# Enhanced label mapping
LABEL_MAP = {"benign": 0, "phishing": 1, "defacement": 2, "malware": 3}

# Comprehensive feature keys
FEATURE_KEYS = [
    # Basic URL features
    'url_length', 'num_dots', 'num_hyphens', 'num_at', 'num_question',
    'num_equal', 'num_slash', 'num_digits', 'ratio_digits',
    'has_ip', 'host_length', 'num_subdomains',

    # Advanced URL features
    'num_params', 'path_length', 'query_length', 'fragment_length',
    'num_uppercase', 'num_lowercase', 'ratio_uppercase',
    'num_special_chars', 'ratio_special_chars',

    # Suspicious patterns
    'has_suspicious_words', 'num_suspicious_words', 'entropy',
    'longest_word_length', 'avg_word_length', 'num_words',
    'has_shortening_service', 'has_redirect_words',
    'num_hexadecimal', 'ratio_hexadecimal',

    # Domain features
    'is_ip_address', 'has_port', 'tld_length', 'has_common_tld',
    'domain_token_count', 'subdomain_depth',

    # Path and query features
    'path_token_count', 'query_token_count', 'has_executable_extension',
    'has_suspicious_extension', 'num_encoded_chars',

    # Statistical features
    'char_frequency_variance', 'digit_concentration',
    'consonant_vowel_ratio', 'repeated_char_ratio'
]

# Suspicious keywords and patterns
SUSPICIOUS_WORDS = [
    'secure', 'account', 'update', 'confirm', 'verify', 'login', 'signin',
    'bank', 'paypal', 'ebay', 'amazon', 'microsoft', 'apple', 'google',
    'facebook', 'twitter', 'instagram', 'netflix', 'urgent', 'suspended',
    'click', 'here', 'now', 'limited', 'offer', 'free', 'winner', 'congratulations'
]

SHORTENING_SERVICES = [
    'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd',
    'buff.ly', 'adf.ly', 'short.link', 'tiny.cc'
]

REDIRECT_WORDS = ['redirect', 'r', 'url', 'link', 'goto', 'forward', 'proxy']

COMMON_TLDS = ['.com', '.org', '.net', '.edu', '.gov', '.mil', '.int']

SUSPICIOUS_EXTENSIONS = ['.exe', '.scr', '.bat', '.cmd', '.pif', '.zip', '.rar']


def calculate_entropy(text):
    """Calculate Shannon entropy of text"""
    if not text:
        return 0
    counter = Counter(text)
    length = len(text)
    entropy = -sum((count / length) * np.log2(count / length) for count in counter.values())
    return entropy


def normalize_url(u):
    """Enhanced URL normalization"""
    if not isinstance(u, str) or u.strip() == "":
        return None
    u = u.strip()
    if not u.lower().startswith(("http://", "https://")):
        u = "http://" + u
    try:
        parsed = urlparse(u)
        # Remove common tracking parameters
        return f"{parsed.scheme}://{parsed.netloc.lower()}{parsed.path}"
    except:
        return None


def extract_advanced_features(url):
    """Extract comprehensive features from URL"""
    try:
        if not isinstance(url, str):
            url = ""

        parsed = urlparse(url)
        host = parsed.netloc or ""
        path = parsed.path or ""
        query = parsed.query or ""
        fragment = parsed.fragment or ""

        # Basic features
        features = {
            'url_length': len(url),
            'num_dots': url.count('.'),
            'num_hyphens': url.count('-'),
            'num_at': url.count('@'),
            'num_question': url.count('?'),
            'num_equal': url.count('='),
            'num_slash': url.count('/'),
            'num_digits': sum(ch.isdigit() for ch in url),
            'ratio_digits': sum(ch.isdigit() for ch in url) / (len(url) + 1),
            'has_ip': 1 if re.search(r"(?:\d{1,3}\.){3}\d{1,3}", url) else 0,
            'host_length': len(host),
            'num_subdomains': host.count('.'),
        }

        # Advanced URL structure features
        features.update({
            'num_params': len(query.split('&')) if query else 0,
            'path_length': len(path),
            'query_length': len(query),
            'fragment_length': len(fragment),
            'num_uppercase': sum(ch.isupper() for ch in url),
            'num_lowercase': sum(ch.islower() for ch in url),
            'ratio_uppercase': sum(ch.isupper() for ch in url) / (len(url) + 1),
            'num_special_chars': sum(ch in string.punctuation for ch in url),
            'ratio_special_chars': sum(ch in string.punctuation for ch in url) / (len(url) + 1),
        })

        # Suspicious pattern features
        url_lower = url.lower()
        suspicious_count = sum(word in url_lower for word in SUSPICIOUS_WORDS)
        features.update({
            'has_suspicious_words': 1 if suspicious_count > 0 else 0,
            'num_suspicious_words': suspicious_count,
            'entropy': calculate_entropy(url),
            'has_shortening_service': 1 if any(service in url_lower for service in SHORTENING_SERVICES) else 0,
            'has_redirect_words': 1 if any(word in url_lower for word in REDIRECT_WORDS) else 0,
        })

        # Text analysis features
        words = re.findall(r'[a-zA-Z]+', url)
        if words:
            features.update({
                'longest_word_length': max(len(word) for word in words),
                'avg_word_length': sum(len(word) for word in words) / len(words),
                'num_words': len(words),
            })
        else:
            features.update({
                'longest_word_length': 0,
                'avg_word_length': 0,
                'num_words': 0,
            })

        # Hexadecimal detection
        hex_chars = sum(ch in '0123456789abcdefABCDEF' for ch in url)
        features.update({
            'num_hexadecimal': hex_chars,
            'ratio_hexadecimal': hex_chars / (len(url) + 1),
        })

        # Domain-specific features
        features.update({
            'is_ip_address': 1 if re.match(r'^\d+\.\d+\.\d+\.\d+', host) else 0,
            'has_port': 1 if ':' in host and not host.startswith('[') else 0,
            'has_common_tld': 1 if any(tld in url_lower for tld in COMMON_TLDS) else 0,
            'domain_token_count': len(host.split('.')),
            'subdomain_depth': max(0, host.count('.') - 1),
        })

        # TLD analysis
        tld = ""
        if '.' in host:
            tld = host.split('.')[-1]
        features['tld_length'] = len(tld)

        # Path and query analysis
        path_tokens = [token for token in path.split('/') if token]
        query_tokens = [token for token in query.split('&') if token]
        features.update({
            'path_token_count': len(path_tokens),
            'query_token_count': len(query_tokens),
            'has_executable_extension': 1 if any(ext in url_lower for ext in SUSPICIOUS_EXTENSIONS) else 0,
            'has_suspicious_extension': 1 if any(ext in path.lower() for ext in SUSPICIOUS_EXTENSIONS) else 0,
            'num_encoded_chars': url.count('%'),
        })

        # Statistical features
        if url:
            char_counts = Counter(url)
            char_frequencies = np.array(list(char_counts.values()))
            features['char_frequency_variance'] = np.var(char_frequencies)

            # Digit concentration (how clustered digits are)
            digit_positions = [i for i, ch in enumerate(url) if ch.isdigit()]
            if len(digit_positions) > 1:
                digit_gaps = np.diff(digit_positions)
                features['digit_concentration'] = 1 / (np.mean(digit_gaps) + 1)
            else:
                features['digit_concentration'] = 0

            # Consonant to vowel ratio
            vowels = sum(ch.lower() in 'aeiou' for ch in url if ch.isalpha())
            consonants = sum(ch.isalpha() for ch in url) - vowels
            features['consonant_vowel_ratio'] = consonants / (vowels + 1)

            # Repeated character patterns
            repeated_chars = sum(1 for i in range(len(url) - 1) if url[i] == url[i + 1])
            features['repeated_char_ratio'] = repeated_chars / (len(url) + 1)
        else:
            features.update({
                'char_frequency_variance': 0,
                'digit_concentration': 0,
                'consonant_vowel_ratio': 0,
                'repeated_char_ratio': 0,
            })

        # Ensure all expected keys are present
        for key in FEATURE_KEYS:
            if key not in features:
                features[key] = 0

        return features

    except Exception as e:
        print(f"Error extracting features from URL: {url}, Error: {e}")
        return {k: 0 for k in FEATURE_KEYS}


def create_ensemble_models(random_state=42):
    """Create advanced ensemble models"""
    models = {}

    # Optimized Random Forest
    models['random_forest_optimized'] = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        n_jobs=-1,
        random_state=random_state,
        class_weight='balanced'
    )

    # Gradient Boosting
    models['gradient_boosting'] = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=random_state
    )

    # Enhanced Neural Network
    models['mlp_enhanced'] = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state
    )

    # Support Vector Machine with RBF kernel
    models['svm_optimized'] = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True,
        class_weight='balanced',
        random_state=random_state
    )

    # Logistic Regression with regularization
    models['logistic_regression'] = LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='liblinear',
        max_iter=1000,
        class_weight='balanced',
        random_state=random_state
    )

    # XGBoost if available
    if XGB_AVAILABLE:
        models['xgboost_optimized'] = XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=random_state
        )

    # LightGBM if available
    if LGB_AVAILABLE:
        models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight='balanced',
            random_state=random_state,
            verbose=-1
        )

    # CatBoost if available
    if CATBOOST_AVAILABLE:
        models['catboost'] = CatBoostClassifier(
            iterations=300,
            depth=8,
            learning_rate=0.1,
            l2_leaf_reg=3,
            class_weights='Balanced',
            random_seed=random_state,
            verbose=False
        )

    return models


def main(csv_path, models_dir, max_samples=None, random_state=42):
    os.makedirs(models_dir, exist_ok=True)
    print("=" * 60)
    print("ENHANCED ML PIPELINE FOR URL CLASSIFICATION")
    print("=" * 60)
    print("Reading", csv_path)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    print(f"Initial dataset size: {len(df)} rows")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"After deduplication: {len(df)} rows")

    # Detect columns
    if 'URL' not in df.columns:
        url_candidates = [c for c in df.columns if 'url' in c.lower() or 'link' in c.lower()]
        if url_candidates:
            df = df.rename(columns={url_candidates[0]: 'URL'})
            print(f"Using '{url_candidates[0]}' as URL column")
        else:
            print("Error: No URL column found in dataset")
            return

    if 'Type' not in df.columns:
        type_candidates = [c for c in df.columns if c.lower() in ('type', 'label', 'class', 'y')]
        if type_candidates:
            df = df.rename(columns={type_candidates[0]: 'Type'})
            print(f"Using '{type_candidates[0]}' as Type column")
        else:
            print("Error: No Type/Label column found in dataset")
            return

    # Enhanced label mapping
    df['Type_mapped'] = df['Type'].apply(lambda x: LABEL_MAP.get(str(x).lower().strip()) if pd.notna(x) else None)
    before = len(df)
    df = df.dropna(subset=['Type_mapped']).reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows with missing/unmapped labels. Remaining: {len(df)}")

    # Show class distribution
    class_dist = df['Type_mapped'].value_counts().sort_index()
    print("\nClass distribution:")
    for label, count in class_dist.items():
        label_name = [k for k, v in LABEL_MAP.items() if v == label][0]
        print(f"  {label_name}: {count} ({count / len(df) * 100:.1f}%)")

    if len(df) == 0:
        print("Error: No valid data remaining after filtering")
        return

    # Smart sampling to maintain class balance
    if max_samples and max_samples > 0 and len(df) > max_samples:
        print(f"\nSampling {max_samples} rows with stratified approach...")
        df_sampled = df.groupby('Type_mapped', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max_samples // len(class_dist)), random_state=random_state)
        ).reset_index(drop=True)

        # If we still need more samples, add randomly
        if len(df_sampled) < max_samples:
            remaining = max_samples - len(df_sampled)
            additional = df[~df.index.isin(df_sampled.index)].sample(
                min(remaining, len(df) - len(df_sampled)), random_state=random_state
            )
            df_sampled = pd.concat([df_sampled, additional]).reset_index(drop=True)

        df = df_sampled
        print(f"Sampled dataset size: {len(df)} rows")

    # Enhanced URL processing
    print("\nProcessing URLs...")
    df['URL'] = df['URL'].apply(normalize_url)
    before_url_filter = len(df)
    df = df.dropna(subset=['URL']).reset_index(drop=True)
    print(f"Dropped {before_url_filter - len(df)} invalid URLs. Remaining: {len(df)}")

    if len(df) == 0:
        print("Error: No valid URLs remaining")
        return

    # Advanced feature extraction with progress tracking
    print("\nExtracting advanced features...")
    feature_list = []
    batch_size = 1000

    for i in range(0, len(df), batch_size):
        batch_end = min(i + batch_size, len(df))
        print(f"Processing batch {i // batch_size + 1}/{(len(df) - 1) // batch_size + 1} "
              f"(URLs {i + 1}-{batch_end})")

        batch_features = []
        for j in range(i, batch_end):
            batch_features.append(extract_advanced_features(df.iloc[j]['URL']))

        feature_list.extend(batch_features)

    # Create comprehensive feature dataset
    print("Creating feature dataset...")
    feats = pd.DataFrame(feature_list, columns=FEATURE_KEYS)

    # Handle any missing values
    for col in FEATURE_KEYS:
        if col not in feats.columns:
            feats[col] = 0
        feats[col] = pd.to_numeric(feats[col], errors='coerce').fillna(0)

    # Combine with original data
    dff = df.reset_index(drop=True).copy()
    for col in FEATURE_KEYS:
        dff[col] = feats.reset_index(drop=True)[col]

    dff['Type'] = dff['Type_mapped'].astype(int)

    print(f"\nFinal dataset: {len(dff)} rows × {len(FEATURE_KEYS)} features")
    print(f"Feature statistics:")
    print(f"  Non-zero features per sample: {(dff[FEATURE_KEYS] != 0).sum(axis=1).mean():.1f}")
    print(f"  Feature value ranges: {dff[FEATURE_KEYS].min().min():.3f} to {dff[FEATURE_KEYS].max().max():.3f}")

    # Prepare data with advanced preprocessing
    X = dff[FEATURE_KEYS].values
    y = dff['Type'].values

    print(f"\nPreparing train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Advanced feature scaling
    print("Applying robust scaling...")
    scaler = RobustScaler()  # More robust to outliers than StandardScaler
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Feature selection
    print("Selecting best features...")
    selector = SelectKBest(score_func=f_classif, k=min(40, len(FEATURE_KEYS)))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    selected_features = [FEATURE_KEYS[i] for i in selector.get_support(indices=True)]
    print(f"Selected {len(selected_features)} most informative features")

    # Create and train advanced models
    print("\n" + "=" * 60)
    print("TRAINING ADVANCED MODELS")
    print("=" * 60)

    models = create_ensemble_models(random_state)
    results = {}
    trained_models = {}

    for name, clf in models.items():
        print(f"\nTraining {name}...")
        t0 = time.time()

        try:
            # Use selected features for training
            clf.fit(X_train_selected, y_train)
            t1 = time.time()
            print(f"Training completed in {t1 - t0:.1f}s")

            # Cross-validation score
            cv_scores = cross_val_score(clf, X_train_selected, y_train, cv=5, scoring='accuracy')
            print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

            # Test set predictions
            y_pred = clf.predict(X_test_selected)

            # Calculate comprehensive metrics
            r = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision_macro': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
                'recall_macro': float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
                'f1_macro': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
                'cv_accuracy_mean': float(cv_scores.mean()),
                'cv_accuracy_std': float(cv_scores.std()),
            }

            # AUC metrics for probability-based models
            if hasattr(clf, 'predict_proba'):
                try:
                    y_proba = clf.predict_proba(X_test_selected)
                    y_test_bin = pd.get_dummies(y_test)
                    r['roc_auc_macro'] = float(roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr'))
                    r['pr_auc_macro'] = float(average_precision_score(y_test_bin, y_proba, average='macro'))
                except Exception as e:
                    print(f"Warning: AUC calculation failed: {e}")
                    r['roc_auc_macro'] = None
                    r['pr_auc_macro'] = None
            else:
                r['roc_auc_macro'] = None
                r['pr_auc_macro'] = None

            r['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
            r['classification_report'] = classification_report(y_test, y_pred, zero_division=0)

            results[name] = r
            trained_models[name] = clf

            print(f"Accuracy: {r['accuracy']:.3f} | F1-Macro: {r['f1_macro']:.3f}")

            # Save individual model
            model_artifact = {
                'model': clf,
                'scaler': scaler,
                'feature_selector': selector,
                'selected_features': selected_features,
                'all_features': FEATURE_KEYS
            }

            model_path = os.path.join(models_dir, f"{name}.joblib")
            joblib.dump(model_artifact, model_path)

        except Exception as e:
            print(f"Error training {name}: {e}")
            continue

    if not results:
        print("Error: No models were successfully trained")
        return

    # Create ensemble of best models
    print(f"\n{'=' * 60}")
    print("CREATING ENSEMBLE MODEL")
    print("=" * 60)

    # Select top 3 models for ensemble
    top_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:3]
    print(f"Top models for ensemble: {[name for name, _ in top_models]}")

    ensemble_estimators = [(name, trained_models[name]) for name, _ in top_models]
    ensemble = VotingClassifier(estimators=ensemble_estimators, voting='soft')

    print("Training ensemble...")
    ensemble.fit(X_train_selected, y_train)
    y_pred_ensemble = ensemble.predict(X_test_selected)

    # Evaluate ensemble
    ensemble_results = {
        'accuracy': float(accuracy_score(y_test, y_pred_ensemble)),
        'precision_macro': float(precision_score(y_test, y_pred_ensemble, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_test, y_pred_ensemble, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_test, y_pred_ensemble, average='macro', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, y_pred_ensemble).tolist(),
        'classification_report': classification_report(y_test, y_pred_ensemble, zero_division=0)
    }

    results['ensemble'] = ensemble_results

    # Save ensemble model
    ensemble_artifact = {
        'model': ensemble,
        'scaler': scaler,
        'feature_selector': selector,
        'selected_features': selected_features,
        'all_features': FEATURE_KEYS
    }
    joblib.dump(ensemble_artifact, os.path.join(models_dir, "ensemble.joblib"))

    print(f"Ensemble Accuracy: {ensemble_results['accuracy']:.3f}")

    # Save comprehensive results
    results_path = os.path.join(models_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Create detailed summary
    summary_data = []
    for model_name, metrics in results.items():
        summary_data.append({
            'model': model_name,
            'accuracy': metrics['accuracy'],
            'f1_macro': metrics['f1_macro'],
            'precision_macro': metrics['precision_macro'],
            'recall_macro': metrics['recall_macro'],
            'cv_accuracy': metrics.get('cv_accuracy_mean', 'N/A')
        })

    summary_df = pd.DataFrame(summary_data).sort_values('accuracy', ascending=False)
    summary_path = os.path.join(models_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # Final summary
    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE - RESULTS SUMMARY")
    print("=" * 60)
    print(f"Models saved to: {models_dir}")
    print(f"Best performing models:")
    for i, (model, acc) in enumerate([(row['model'], row['accuracy']) for _, row in summary_df.head(3).iterrows()], 1):
        print(f"  {i}. {model}: {acc:.3f} accuracy")

    # Check if we achieved >90% accuracy
    best_accuracy = summary_df.iloc[0]['accuracy']
    if best_accuracy >= 0.90:
        print(f"\n🎉 SUCCESS: Achieved {best_accuracy:.3f} accuracy (>90% target met!)")
    else:
        print(f"\n⚠️  Best accuracy: {best_accuracy:.3f} (Target: >90%)")
        print("Consider: More data, feature engineering, or hyperparameter tuning")

    return results


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    default_csv = script_dir/"malicious_phish_clean.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=str(default_csv))
    parser.add_argument('--models-dir', default=str(script_dir/"models"))
    parser.add_argument('--max-samples', type=int, default=8000, help='0 or omit to use full dataset')
    args = parser.parse_args()
    main(args.csv, args.models_dir, max_samples=(args.max_samples or None))