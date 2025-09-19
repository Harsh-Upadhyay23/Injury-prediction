#!/usr/bin/env python3
"""
Analysis script to identify the best performing models for injury prediction
"""

import pandas as pd
import joblib
import os

def analyze_best_models():
    """Analyze all model results to identify the best performers"""
    
    print("🔍 Analyzing Best Models for Injury Prediction")
    print("=" * 60)
    
    # Load all leaderboards
    try:
        # Advanced ensembles
        adv_severity = pd.read_csv("results/updated_advanced_hybrid/updated_ensembles_severity_leaderboard.csv")
        adv_location = pd.read_csv("results/updated_advanced_hybrid/updated_ensembles_location_leaderboard.csv")
        
        # Basic models
        basic_severity = pd.read_csv("results/severity_model_leaderboard.csv")
        basic_location = pd.read_csv("results/location_model_leaderboard.csv")
        
        print("✅ All leaderboards loaded successfully")
        
    except FileNotFoundError as e:
        print(f"❌ Error loading leaderboards: {e}")
        return
    
    # Combine all models for comparison
    print("\n📊 SEVERITY PREDICTION - All Models Comparison")
    print("-" * 60)
    
    # Prepare severity data
    adv_sev = adv_severity.copy()
    adv_sev['Category'] = 'Advanced Ensemble'
    
    basic_sev = basic_severity.copy()
    basic_sev['Category'] = 'Basic/Simple'
    
    all_severity = pd.concat([adv_sev, basic_sev], ignore_index=True)
    all_severity = all_severity.sort_values('F1 Score', ascending=False)
    
    print("Top 5 Severity Models:")
    for i, (_, row) in enumerate(all_severity.head().iterrows()):
        print(f"{i+1}. {row['Model']} ({row['Category']}) - F1: {row['F1 Score']:.4f}")
    
    print("\n📊 LOCATION PREDICTION - All Models Comparison")
    print("-" * 60)
    
    # Prepare location data
    adv_loc = adv_location.copy()
    adv_loc['Category'] = 'Advanced Ensemble'
    
    basic_loc = basic_location.copy()
    basic_loc['Category'] = 'Basic/Simple'
    
    all_location = pd.concat([adv_loc, basic_loc], ignore_index=True)
    all_location = all_location.sort_values('F1 Score', ascending=False)
    
    print("Top 5 Location Models:")
    for i, (_, row) in enumerate(all_location.head().iterrows()):
        print(f"{i+1}. {row['Model']} ({row['Category']}) - F1: {row['F1 Score']:.4f}")
    
    # Identify best models
    best_severity = all_severity.iloc[0]
    best_location = all_location.iloc[0]
    
    print("\n🏆 BEST MODELS IDENTIFIED")
    print("=" * 60)
    print(f"🥇 Best Severity Model: {best_severity['Model']}")
    print(f"   Category: {best_severity['Category']}")
    print(f"   F1-Score: {best_severity['F1 Score']:.4f}")
    print(f"   Accuracy: {best_severity['Accuracy']:.4f}")
    
    print(f"\n🥇 Best Location Model: {best_location['Model']}")
    print(f"   Category: {best_location['Category']}")
    print(f"   F1-Score: {best_location['F1 Score']:.4f}")
    print(f"   Accuracy: {best_location['Accuracy']:.4f}")
    
    # Check if models exist
    print("\n🔍 Checking Model Availability")
    print("-" * 60)
    
    # Check severity model
    severity_model_path = None
    if best_severity['Category'] == 'Advanced Ensemble':
        model_name = best_severity['Model'].replace(' ', '_').lower()
        severity_model_path = f"results/updated_advanced_hybrid/{model_name}_severity_model.pkl"
    else:
        model_name = best_severity['Model'].replace(' ', '_').lower()
        severity_model_path = f"results/models/{model_name}_severity_model.pkl"
    
    if os.path.exists(severity_model_path):
        print(f"✅ Severity model found: {severity_model_path}")
    else:
        print(f"❌ Severity model not found: {severity_model_path}")
    
    # Check location model
    location_model_path = None
    if best_location['Category'] == 'Advanced Ensemble':
        model_name = best_location['Model'].replace(' ', '_').lower()
        location_model_path = f"results/updated_advanced_hybrid/{model_name}_location_model.pkl"
    else:
        model_name = best_location['Model'].replace(' ', '_').lower()
        location_model_path = f"results/models/{model_name}_location_model.pkl"
    
    if os.path.exists(location_model_path):
        print(f"✅ Location model found: {location_model_path}")
    else:
        print(f"❌ Location model not found: {location_model_path}")
    
    # Save best model info
    best_models_info = {
        'severity': {
            'model_name': best_severity['Model'],
            'category': best_severity['Category'],
            'f1_score': best_severity['F1 Score'],
            'accuracy': best_severity['Accuracy'],
            'model_path': severity_model_path
        },
        'location': {
            'model_name': best_location['Model'],
            'category': best_location['Category'],
            'f1_score': best_location['F1 Score'],
            'accuracy': best_location['Accuracy'],
            'model_path': location_model_path
        }
    }
    
    # Save to file
    import json
    with open('best_models_info.json', 'w') as f:
        json.dump(best_models_info, f, indent=2)
    
    print(f"\n💾 Best models info saved to: best_models_info.json")
    
    return best_models_info

if __name__ == "__main__":
    analyze_best_models()
