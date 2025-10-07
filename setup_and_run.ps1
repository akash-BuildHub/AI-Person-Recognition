# setup_and_run.ps1

# Step 1: Upgrade pip
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

# Step 2: Install all required packages
Write-Host "Installing dependencies from requirements.txt..."
pip install -r .\requirements.txt

# Step 3: Confirm installation
Write-Host "Listing installed packages..."
pip list

# Step 4: Run the Streamlit app
Write-Host "Launching Streamlit app..."
streamlit run .\streamlit_app.py
