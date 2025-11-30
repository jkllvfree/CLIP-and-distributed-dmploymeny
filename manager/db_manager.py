import mysql.connector
from dbutils.pooled_db import PooledDB
from flask import jsonify

from utils.config import DB_CONFIG # 导入刚才写好的配置

POOL = PooledDB(
    creator=mysql.connector,  # 使用的驱动
    maxconnections=10,        # 连接池允许的最大连接数
    mincached=2,              # 初始化时，连接池中至少创建的空闲连接
    maxcached=5,              # 连接池中最多闲置的连接
    blocking=True,            # 连接池中如果没有可用连接后，是否阻塞等待
    ping=0,                   # 0 = None = never, 1 = default = whenever it is requested, 2 = when a cursor is created, 4 = when a query is executed, 7 = always
    **DB_CONFIG               # 使用 ** 语法自动解包上面的字典配置
)


def get_db_connection():
    return POOL.connection()

#还可以写其他数据库的常用操作函数
def execute_query(query, params=None, logger=None):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(query, params)
        results = cursor.fetchall()
        return results
    except Exception as e:
        if logger:
            logger.error(f"数据库查询失败: {e}")
        return jsonify({"code": 0, "msg": f"数据库查询失败：{str(e)}"}), 500

    finally:
        cursor.close()
        conn.close()

def execute_insert(query, params=None, logger=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.executemany(query, params)
        conn.commit()
        if logger:
            logger.info(f"[BackgroundTask] 成功插入 {len(params)} 条记录")
    except Exception as e:
        conn.rollback()
        if logger:
            logger.error(f"数据库插入失败: {e}")
    #资源释放
    finally:
        cursor.close()
        conn.close()