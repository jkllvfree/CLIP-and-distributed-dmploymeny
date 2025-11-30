import mysql.connector
from dbutils.pooled_db import PooledDB
import os
from dotenv import load_dotenv

load_dotenv()

SERVER_IP = os.getenv("SERVER_IP")
SERVER_PORT = int(os.getenv("SERVER_PORT"))


#数据库的配置
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "charset": "utf8mb4"
}

OFFLOAD_CONFIG = {
    'visual_attn': os.getenv("OFFLOAD_VISUAL_ATTN", 'false').lower() == 'true',
    'visual_mlp': os.getenv("OFFLOAD_VISUAL_MLP", 'false').lower() == 'true',
    'text_attn': os.getenv("OFFLOAD_TEXT_ATTN", 'false').lower() == 'true',
    # ... 可根据需要扩展
}



