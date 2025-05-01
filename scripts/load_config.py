"""
Usage example:

from load_config import initialize_config, get_env_variable

config = initialize_config()
api_key = get_env_variable("OPENAI_API_KEY")

embedding_model = config["openai"]["embedding_model"]
top_k = config["rag"]["top_k"]
"""

import os
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_secrets():
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            return dict(st.secrets)
    except ImportError:
        pass
    return {}

def get_env_variable(key, default=None):
    # First check Streamlit secrets, then .env/os
    secrets = load_secrets()
    return secrets.get(key) or os.getenv(key, default)

def initialize_config():
    from dotenv import load_dotenv
    load_dotenv()
    return load_config()
