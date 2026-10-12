import unittest
import numpy as np
from experiments.motivation_exp2.utils.annotation_utils import match_instances, persistence_counts, instance_map


class InstanceTests(unittest.TestCase):
    def test_same_instance_different_annotation(self):
        a = dict(instance_token="I",token="ann0",category_name="car")
        b = dict(instance_token="I",token="ann1",category_name="car")
        self.assertEqual(match_instances([a],[b]),[(a,b)])

    def test_same_category_different_instance_not_matched(self):
        a = dict(instance_token="A",token="ann0",category_name="car",translation=[0,0,0])
        b = dict(instance_token="B",token="ann1",category_name="car",translation=[0,0,0])
        self.assertEqual(match_instances([a],[b]),[])

    def test_empty_persistence_is_nan(self):
        result = persistence_counts([], ["x"])
        self.assertTrue(np.isnan(result["persistent_ratio"]))
        self.assertFalse(result["denominator_valid"])

    def test_exact_counts(self):
        result = persistence_counts(["a","b"],["b","c"])
        self.assertEqual(result["persistent_ratio"],.5)
        self.assertEqual(result["new_objects"],1)
        self.assertEqual(result["disappeared_objects"],1)

    def test_duplicate_instance_rejected(self):
        with self.assertRaises(ValueError):
            instance_map([dict(instance_token="I"),dict(instance_token="I")])
