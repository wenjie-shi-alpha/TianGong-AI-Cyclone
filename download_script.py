import os
import pandas as pd
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import NoCredentialsError

# 创建data目录
os.makedirs('data', exist_ok=True)

# 读取CSV文件
df = pd.read_csv('output/nc_file_urls.csv')

# 选择三个模型的文件
# 使用CSV中存在的文件
selected_files = {
    'pangu': 'PANG_v100_IFS_2022032900_f000_f240_06.nc',
    'aurora': 'AURO_v100_IFS_2025061000_f000_f240_06.nc',
    'fourcast': 'FOUR_v200_GFS_2020093012_f000_f240_06.nc'
}

# S3配置
bucket_name = 'noaa-oar-mlwp-data'

# 初始化S3客户端，使用匿名访问
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED, region_name='us-east-1'))

def download_file(s3_key, local_filename):
    try:
        s3.download_file(bucket_name, s3_key, f'data/{local_filename}')
        print(f'下载成功: {local_filename}')
    except NoCredentialsError:
        print('AWS凭证错误')
    except Exception as e:
        print(f'下载失败 {local_filename}: {e}')

# 下载文件
for model, filename in selected_files.items():
    # 从CSV中找到对应的key
    row = df[df['key'].str.contains(filename)]
    if not row.empty:
        s3_key = row.iloc[0]['key']
        download_file(s3_key, filename)
    else:
        print(f'未找到文件: {filename}')

print('下载完成')