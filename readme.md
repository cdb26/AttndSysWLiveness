Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
py -3.10 -m venv venv

venv\Scripts\activate
pip install opencv-python opencv-contrib-python numpy