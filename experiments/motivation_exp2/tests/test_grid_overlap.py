import unittest
import numpy as np
from experiments.motivation_exp2.utils.geometry_utils import grid_cells, grid_overlap


class GridTests(unittest.TestCase):
    def test_identical_sets(self):
        m = grid_overlap({(0,0),(1,1)},{(0,0),(1,1)})
        self.assertEqual(m["grid_overlap"],1)
        self.assertEqual(m["old_grid_retention"],1)

    def test_disjoint_sets(self):
        m = grid_overlap({(0,0)},{(1,1)})
        self.assertEqual(m["grid_overlap"],0)
        self.assertTrue(m["completely_migrated"])

    def test_empty_sets_nan(self):
        m = grid_overlap(set(),set())
        self.assertTrue(np.isnan(m["grid_overlap"]))
        self.assertTrue(np.isnan(m["old_grid_retention"]))
        self.assertIsNone(m["completely_migrated"])

    def test_retention_separate_from_jaccard(self):
        m = grid_overlap({(0,0)},{(0,0),(1,1)})
        self.assertEqual(m["old_grid_retention"],1)
        self.assertEqual(m["grid_overlap"],.5)

    def test_boundary_touch_not_area_intersection(self):
        self.assertEqual(grid_cells([0,0,50,50],100,100,[2,2]),{(0,0)})

    def test_three_rules(self):
        b = [49,0,51,50]
        self.assertEqual(grid_cells(b,100,100,[2,2],"any"),{(0,0),(0,1)})
        self.assertEqual(grid_cells(b,100,100,[2,2],"coverage",.25),set())
        self.assertEqual(grid_cells(b,100,100,[2,2],"center"),set())

    def test_partial_outside_and_nonsquare_grid(self):
        self.assertEqual(grid_cells([-10,-10,30,30],90,60,[2,3]),{(0,0)})
