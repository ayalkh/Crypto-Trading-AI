import os
import sys
import subprocess
import glob

def run_tests():
    test_files = glob.glob('tests/test_*.py')
    test_files.sort()
    
    # Filter out this script if it matches pattern (it won't because it's run_all_tests.py)
    
    print(f"Found {len(test_files)} test scripts.")
    print("="*60)
    
    passed = []
    failed = []
    
    for test_file in test_files:
        print(f"\nRunning {test_file}...")
        print("-" * 40)
        
        try:
            # Run the test script with output streaming
            result = subprocess.run(
                [sys.executable, "-u", test_file],
                capture_output=False,
                text=True
            )
            
            # Output is already streamed
            
            if result.returncode == 0:
                print(f"\n✅ {test_file} PASSED")
                passed.append(test_file)
            else:
                print(f"\n❌ {test_file} FAILED (Exit code: {result.returncode})")
                failed.append(test_file)
                
        except Exception as e:
            print(f"\n❌ {test_file} FAILED TO EXECUTE: {e}")
            failed.append(test_file)
            
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {len(test_files)}")
    print(f"Passed:      {len(passed)}")
    print(f"Failed:      {len(failed)}")
    
    if failed:
        print("\nFailed Tests:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\n🎉 All tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    # Ensure we are running from project root
    if not os.path.exists('tests'):
        print("Error: Please run this script from the project root directory.")
        sys.exit(1)
    run_tests()
