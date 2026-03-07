

# TEST - Retrieve default anda ll models
from models import MODELS, DefaultModel

print("Default model:", DefaultModel())
print("First 10 available:", MODELS.list()[:10])
for m in MODELS.list_with_meta()[:3]:
    print(m["id"], m["created"], m["owned_by"])

MODELS.list()  # ensure cache