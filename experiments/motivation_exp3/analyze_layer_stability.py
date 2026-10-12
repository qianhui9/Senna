"""Layer comparison uses the same valid comparison IDs at all extracted layers."""
import pandas as pd
from .utils.common import read, save, standalone, write_json
from .utils.statistics import grouped


def common_layer_cohort(df):
    valid = df[df.numeric_status == "valid"]
    keys = ["object_pair_id", "comparison_group", "aggregation"]
    layers = set(df.layer_id.unique())
    sizes = valid.groupby(keys).layer_id.nunique()
    ids = sizes[sizes == len(layers)].reset_index()[keys]
    return valid.merge(ids, on=keys, how="inner", validate="many_to_one")


def run(cfg, out):
    data = read(out, "feature_similarity")
    common = common_layer_cohort(data)
    save(out, "layer_stability", grouped(common, ["layer_id", "comparison_group", "aggregation"], cfg))
    write_json(out/"metadata/layer_cohort.json", {"rows_before": len(data), "common_valid_rows": len(common),
        "definition": "Identical object/control pair IDs valid at every extracted layer, separately by aggregation/group"})


if __name__ == "__main__":
    standalone("layer-analysis")
