
# TianGong AI Cyclone

## Env Preparing

Setup `venv`:

```bash

sudo apt-get install python3.12-dev
sudo apt-get install nvidia-cuda-toolkit

python3.12 -m venv .venv
source .venv/bin/activate
```

Install requirements:

```bash
python.exe -m pip install --upgrade pip

pip install --upgrade pip

pip install --upgrade pip
pip install -r requirements.txt

pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt --upgrade

pip install -r requirements_freeze.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

pip freeze > requirements_freeze.txt


aws s3 ls --no-sign-request --region us-east-1 s3://noaa-oar-mlwp-data/
```

Auto lint:
```bash
black .
```

## Run with PM2

```bash

npm i -g pm2

pm2 start ecosystem.config.json

pm2 start ecosystem.quatro.json

pm2 restart all

pm2 status

pm2 restart unstructured-gunicorn
pm2 stop unstructured-gunicorn
pm2 delete unstructured-gunicorn

pm2 logs unstructured-gunicorn
```
