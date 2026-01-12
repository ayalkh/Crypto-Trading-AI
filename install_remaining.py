
import subprocess
import sys

def read_installed():
    try:
        # Read pyproject.toml manually to see what's there
        # We look for lines in 'dependencies = [...]'
        with open("pyproject.toml", "r") as f:
            content = f.read()
        # Very naive parsing
        import re
        installed = set()
        for match in re.finditer(r'"([a-zA-Z0-9_\-]+)\s*\(', content):
            installed.add(match.group(1).lower())
        return installed
    except Exception as e:
        print(e)
        return set()

def read_requirements(filename):
    packages = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # extract name
            parts = line.split('==')
            name = parts[0]
            if '[' in name: # specific extras
                 name = name.split('[')[0] 
            packages.append((name, line))
    return packages

def install_packages(packages, installed_names, chunk_size=1):
    to_install = []
    for name, line in packages:
        if name.lower() not in installed_names and name.lower().replace('_', '-') not in installed_names:
            to_install.append(line)
    
    print(f"Found {len(to_install)} packages to install.")
    
    for i in range(0, len(to_install), chunk_size):
        chunk = to_install[i:i+chunk_size]
        print(f"Installing chunk {i//chunk_size + 1}: {chunk}")
        try:
            # -vv to see output if possible, but might be too much.
            subprocess.check_call(["poetry", "add"] + chunk)
        except subprocess.CalledProcessError:
            print(f"Error installing chunk: {chunk}")
            sys.exit(1)

if __name__ == "__main__":
    installed = read_installed()
    # explicitly add 'requests' if missed by regex
    installed.add("requests") 
    pkgs = read_requirements("requirements.txt")
    install_packages(pkgs, installed)
