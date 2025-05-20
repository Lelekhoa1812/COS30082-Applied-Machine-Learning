# Installation
1. Create your virtual environment to ensure Python version compatibility wouldn't violate your current setup
```bash
python3 -m venv cos30082-env
source cos30082-env/bin/activate
```
- Note: `python3` on MacOS and `python` on Windows.
- Use `deactivate` to terminate the virtual environment.

2. Upgrade pip to avoid legacy issue:
```bash
pip install --upgrade pip
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Optional:** If you want to re-clone the full repo, please follow:
> Clone Anti-spoofing repo:
```bash
git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git
```

>  Copy the following core folders from the repo into the project's directory:
- Model evaluation Python script `src` folder.
- The model folder with model weights `resources`.