"""
Fix the prediction classification thresholds in ensemble_predictor.py
"""
import re

file_path = 'ml_models/ensemble_predictor.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the DIRECTION_THRESHOLDS dictionary
old_thresholds = r"DIRECTION_THRESHOLDS = \{[^}]+\}"

new_thresholds = """DIRECTION_THRESHOLDS = {
    '5m': 0.05,   # 0.05% for 5-minute (very sensitive)
    '15m': 0.08,  # 0.08% for 15-minute
    '1h': 0.10,   # 0.10% for 1-hour
    '4h': 0.15,   # 0.15% for 4-hour
    '1d': 0.25,   # 0.25% for daily
}"""

if re.search(old_thresholds, content):
    content = re.sub(old_thresholds, new_thresholds, content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Successfully updated prediction thresholds!")
    print("\nNew DIRECTION_THRESHOLDS:")
    print("  5m:  0.05% (was 0.15%)")
    print("  15m: 0.08% (was 0.20%)")
    print("  1h:  0.10% (was 0.20%)")
    print("  4h:  0.15% (was 0.30%)")
    print("  1d:  0.25% (was 0.50%)")
    print("\n🔄 Now you need to regenerate predictions:")
    print("   python scripts/generate_predictions.py")
    print("\n📊 Then test again:")
    print("   python test_agent_quick.py")
else:
    print("❌ Could not find DIRECTION_THRESHOLDS in ensemble_predictor.py")
    print("Manual edit needed!")