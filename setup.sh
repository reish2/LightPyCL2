sudo apt install ocl-icd-opencl-dev python3-pip
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
pip3 install --upgrade setuptools
pip3 install -r requirements.txt

echo "Setup done."
echo "Now activate your fresh virtualenv:"
echo "source venv/bin/activate"