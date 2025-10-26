### Install Python 3.10.10

.\venv\Scripts\activate
(Virtual Environment Ensures no collisions between libraries and correct python version is running)

python.exe -m pip install --upgrade pip

pip3 install torch~=2.2.1 --index-url https://download.pytorch.org/whl/cpu

python -m pip install mlagents==1.1.0

### Unity Setup

Go to Window -> Package Manage -> Package Manager -> Unity Registry

Seach com.unity.ml-agents and install it