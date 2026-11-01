# Downloads and verifies the Kronos model into KRONOS_HOME (used when building the Docker image,
# so a hosted server doesn't re-download 407 MB on every restart).
import sys

import service as S

S.download_model()
if S.model_state["status"] != "ready":
    sys.exit(f"model download failed: {S.model_state['error']}")
print("model ready in", S.MODELS)
