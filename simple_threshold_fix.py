"""
Simple line-by-line replacement
"""
import re

with open('crypto_agent/tools.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the thresholds using regex
replacements = [
    (r'if weighted_price_change > 0\.01:  # > 1%', 
     'if weighted_price_change > 0.003:  # > 0.3%'),
    
    (r'elif weighted_price_change > 0\.005:  # > 0\.5%',
     'elif weighted_price_change > 0.001:  # > 0.1%'),
    
    (r'elif weighted_price_change < -0\.01:  # < -1%',
     'elif weighted_price_change < -0.003:  # < -0.3%'),
    
    (r'elif weighted_price_change < -0\.005:  # < -0\.5%',
     'elif weighted_price_change < -0.001:  # < -0.1%'),
]

count = 0
for old, new in replacements:
    if re.search(old, content):
        content = re.sub(old, new, content)
        count += 1

with open('crypto_agent/tools.py', 'w', encoding='utf-8') as f:
    f.write(content)

print(f"✅ Replaced {count}/4 threshold lines")
if count == 4:
    print("\n🎉 All thresholds updated successfully!")
    print("\nNew thresholds:")
    print("  STRONG_BUY:  > 0.3%  (was > 1%)")
    print("  BUY:         > 0.1%  (was > 0.5%)")  
    print("  SELL:        < -0.1% (was < -0.5%)")
    print("  STRONG_SELL: < -0.3% (was < -1%)")
    print("\n▶️  Now run: python test_agent_quick.py")
else:
    print(f"\n⚠️  Only {count}/4 lines updated. Check manually.")