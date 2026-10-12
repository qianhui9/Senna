import unittest
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import BoxVisibility
from pyquaternion import Quaternion

from experiments.motivation_exp2.utils.geometry_utils import project_box, world_to_camera, relative_ego_motion

CFG = dict(near_plane_m=.1, min_bbox_width_px=8, min_bbox_height_px=8, min_visible_area_ratio=.2)
K = np.array([[500,0,320],[0,500,240],[0,0,1]], float)


class ProjectionTests(unittest.TestCase):
    def box(self, center=(0,0,10), size=(2,2,2)):
        return Box(center, size, Quaternion(), token="annotation")

    def test_identical_pose_static_projection(self):
        pose = dict(rotation=[1,0,0,0], translation=[0,0,0])
        c = self.box().corners()
        a = project_box(world_to_camera(c, pose, pose), K,640,480,CFG)
        b = project_box(world_to_camera(c, pose, pose), K,640,480,CFG)
        self.assertEqual(a,b)
        self.assertEqual(a["projection_status"],"valid")
        self.assertAlmostEqual(a["box_center_x"],320)
        self.assertAlmostEqual(a["box_center_y"],240)

    def test_static_object_different_ego_pose(self):
        p0 = dict(rotation=[1,0,0,0], translation=[0,0,0])
        p1 = dict(rotation=[1,0,0,0], translation=[1,0,0])
        c = self.box().corners()
        a,b = [project_box(world_to_camera(c,p,p0),K,640,480,CFG) for p in [p0,p1]]
        self.assertLess(b["box_center_x"], a["box_center_x"])

    def test_behind_camera(self):
        result = project_box(self.box((0,0,-5)).corners(),K,640,480,CFG)
        self.assertEqual(result["projection_status"],"behind_or_inside_near_plane")

    def test_near_plane_crossing_explicitly_excluded(self):
        result = project_box(self.box((0,0,.5)).corners(),K,640,480,CFG)
        self.assertEqual(result["projection_status"],"near_plane_crossing_excluded")
        self.assertTrue(result["near_plane_crossing"])

    def test_boundary_clipping(self):
        result = project_box(self.box((5,0,10)).corners(),K,640,480,CFG)
        self.assertEqual(result["projection_status"],"valid")
        self.assertTrue(result["geometric_clipped"])
        self.assertLess(result["geometric_in_image_area_ratio"],1)
        self.assertEqual(result["box_xmax"],640)
        self.assertGreater(result["raw_box_xmax"],640)

    def test_outside_image(self):
        result = project_box(self.box((100,0,10)).corners(),K,640,480,CFG)
        self.assertEqual(result["projection_status"],"outside_image")

    def test_nonfinite_and_degenerate(self):
        c = self.box().corners()
        c[0,0] = np.nan
        self.assertEqual(project_box(c,K,640,480,CFG)["projection_status"],"nonfinite_geometry")
        c = np.tile([[0],[0],[10]],(1,8))
        self.assertEqual(project_box(c,K,640,480,CFG)["projection_status"],"degenerate_projection")

    def test_official_transform_agreement(self):
        global_box = self.box((9,4,20))
        pose = dict(rotation=Quaternion(axis=[0,0,1],angle=.3).elements.tolist(), translation=[2,1,0])
        cal = dict(rotation=Quaternion(axis=[1,0,0],angle=.2).elements.tolist(), translation=[.4,.2,1],sensor_token="sensor",camera_intrinsic=K.tolist())
        records = {("sample_data","sd"): dict(calibrated_sensor_token="cal",ego_pose_token="pose",width=640,height=480),
                   ("calibrated_sensor","cal"):cal, ("ego_pose","pose"):pose, ("sensor","sensor"):dict(modality="camera")}
        class DB:
            def get(self,table,token): return records[table,token]
            def get_sample_data_path(self,token): return "image.jpg"
            def get_boxes(self,token): return [global_box.copy()]
        _, boxes,_ = NuScenes.get_sample_data(DB(),"sd",box_vis_level=BoxVisibility.NONE)
        np.testing.assert_allclose(world_to_camera(global_box.corners(),pose,cal),boxes[0].corners(),atol=1e-12)

    def test_yaw_wrap_and_quaternion_sign(self):
        poses = [dict(rotation=Quaternion(axis=[0,0,1],degrees=angle).elements,translation=[0,0,0]) for angle in [179,-179]]
        m = relative_ego_motion(*poses,.5)
        self.assertAlmostEqual(m["ego_rotation_deg"],2,places=8)
        self.assertAlmostEqual(m["ego_yaw_change_deg"],2,places=8)
        poses[1]["rotation"] = -poses[0]["rotation"]
        self.assertAlmostEqual(relative_ego_motion(*poses,.5)["ego_rotation_deg"],0,places=8)

    def test_initial_ego_translation_frame(self):
        p0 = dict(rotation=Quaternion(axis=[0,0,1],degrees=90).elements,translation=[0,0,0])
        p1 = dict(rotation=p0["rotation"],translation=[0,2,0])
        m = relative_ego_motion(p0,p1,1)
        self.assertAlmostEqual(m["ego_translation_x_m"],2)
        self.assertAlmostEqual(m["ego_translation_y_m"],0)

    def test_invalid_time_rejected(self):
        with self.assertRaises(ValueError):
            relative_ego_motion({}, {}, 0)
