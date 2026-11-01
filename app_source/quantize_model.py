# Builds the 8-bit copy of Kronos-base used on small servers (KRONOS_INT8=1). Run after fetch_model.py.
import kronos_paths as kp
import service as S

kp.quantize(S.MODELS)
print("int8 model saved in", S.MODELS)
