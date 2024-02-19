import llama_index.core
import wandb
from dotenv import load_dotenv
from llama_index.core import set_global_handler
import os
load_dotenv("/home/dai/35/AyurSanvaad/src/rag/.env")
wandb.login()