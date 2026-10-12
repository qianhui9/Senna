"""Identity comes exclusively from official instance_token, never proximity."""
import numpy as np


def instance_map(annotations):
    result = {}
    for annotation in annotations:
        instance = annotation["instance_token"]
        if not instance or instance in result:
            raise ValueError("Missing or duplicate instance_token within one sample")
        result[instance] = annotation
    return result


def match_instances(annotations0, annotations1):
    a, b = instance_map(annotations0), instance_map(annotations1)
    return [(a[i], b[i]) for i in sorted(a.keys() & b.keys())]


def persistence_counts(a, b):
    a, b = set(a), set(b)
    return {"objects_t0": len(a), "objects_t1": len(b), "persistent_objects": len(a & b),
            "new_objects": len(b-a), "disappeared_objects": len(a-b),
            "persistent_ratio": len(a & b)/len(a) if a else np.nan,
            "denominator_valid": bool(a)}
