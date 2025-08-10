import os
from dotenv import load_dotenv
load_dotenv()

# KPI targets (override these via environment variables on Render)
TARGET_ROAS = float(os.getenv("TARGET_ROAS", 2.54))
BREAKEVEN_ROAS = float(os.getenv("BREAKEVEN_ROAS", 1.54))
TARGET_CPA = float(os.getenv("TARGET_CPA", 39.30))
BREAKEVEN_CPA_TRUE_AOV = float(os.getenv("BREAKEVEN_CPA_TRUE_AOV", 81.56))
BREAKEVEN_CPA_BUY1 = float(os.getenv("BREAKEVEN_CPA_BUY1", 65.50))
TARGET_CVR = float(os.getenv("TARGET_CVR", 4.2))

# Storage (SQLite file in the app directory by default)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///erto_agent.db")

# Provider keys (kept empty here; set in Render env)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Modes
EXPERT_MODE = os.getenv("EXPERT_MODE", "ON").upper() == "ON"
