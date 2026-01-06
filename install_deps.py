
import subprocess
import sys

def read_requirements(filename):
    packages = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Remove anything that might confuse poetry if ambiguous, generally 'pkg==ver' is fine.
            # However, lines like 'pkg @ url' or '-e .' need handling.
            # The shown requirements.txt looks like 'pkg==ver'.
            packages.append(line)
    return packages

def install_packages(packages, chunk_size=20):
    for i in range(0, len(packages), chunk_size):
        chunk = packages[i:i+chunk_size]
        print(f"Installing chunk {i//chunk_size + 1}/{len(packages)//chunk_size + 1}: {chunk}")
        try:
            subprocess.check_call(["poetry", "add"] + chunk)
        except subprocess.CalledProcessError as e:
            print(f"Error installing chunk: {chunk}")
            print(e)
            # Continue or stop? Better stop to let user debug, or try individual?
            # Stopping is safer to avoid partial messed up state that scrolls away.
            sys.exit(1)

if __name__ == "__main__":
    pkgs = read_requirements("requirements.txt")
    install_packages(pkgs)
