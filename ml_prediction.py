import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def prepare_features(df):
    """
    Prepare features for ML model using technical indicators
    """
    # Create features from technical indicators
    features = pd.DataFrame(index=df.index)
    
    # Price-based features
    features['price_change'] = df['Close'].pct_change()
    features['volume_change'] = df['Volume'].pct_change()
    features['price_volatility'] = df['Close'].rolling(window=10).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # Moving averages
    features['ma5'] = df['Close'].rolling(window=5).mean()
    features['ma20'] = df['Close'].rolling(window=20).mean()
    features['ma_ratio'] = features['ma5'] / features['ma20']
    
    # Target variable (1 if price goes up, 0 if down)
    features['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Drop NaN values
    features = features.dropna()
    
    return features

def train_model(features):
    """
    Train Random Forest model for trend prediction
    """
    # Prepare data
    X = features.drop('target', axis=1)
    y = features['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, accuracy, (X_test_scaled, y_test, y_pred)

def get_feature_importance(model, features):
    """
    Get feature importance from the model
    """
    feature_names = features.drop('target', axis=1).columns
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    return importance.sort_values('importance', ascending=False)

def predict_trend(model, scaler, latest_features):
    """
    Make prediction for the latest data point
    """
    scaled_features = scaler.transform(latest_features)
    prediction = model.predict_proba(scaled_features)
    return prediction[0]  # Returns probability of price going up/down
