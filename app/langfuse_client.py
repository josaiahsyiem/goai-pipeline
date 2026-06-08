from langfuse import Langfuse
import os

# Langfuse 4.x reads LANGFUSE_HOST for the OTel exporter.
# Ensure it's set from LANGFUSE_BASE_URL if not already present.
os.environ.setdefault(
    "LANGFUSE_HOST",
    os.environ.get("LANGFUSE_BASE_URL", "http://host.docker.internal:3001")
)

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)
