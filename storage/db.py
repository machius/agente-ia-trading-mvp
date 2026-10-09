import os
import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver
load_dotenv()
DATABASE_URL = os.environ["DATABASE_URL"]



def build_checkpointer() -> PostgresSaver:
    conn = psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()  # crea las tablas si no existen; seguro de llamar siempre
    return checkpointer