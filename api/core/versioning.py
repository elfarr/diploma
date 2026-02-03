from dataclasses import dataclass

@dataclass(frozen=True)
class Versions:
    model_version: str
    schema_version: str

def get_versions(model_version: str, schema_version: str) -> Versions:
    return Versions(model_version=model_version, schema_version=schema_version)
