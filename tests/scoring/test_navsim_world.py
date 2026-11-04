from __future__ import annotations
from types import SimpleNamespace as NS
import unittest
import numpy as np
from shapely.geometry import box
from nuplan.common.maps.maps_datatypes import SemanticMapLayer as Layer
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType as Kind
from reactbench.scoring.navsim_world import build_world


class Map:
    def get_available_map_objects(self):
        return [Layer.ROADBLOCK,Layer.INTERSECTION,Layer.CARPARK_AREA]

    def get_proximal_map_objects(self, point, radius, layers):
        return {layer:[NS(polygon=box(-100,-100,100,100))] if layer==Layer.ROADBLOCK else [] for layer in layers}


class Scene:
    map_api=Map()
    initial_ego_state=NS(rear_axle=NS(x=10.,y=20.,heading=np.pi/2),
        car_footprint=NS(vehicle_parameters=NS(rear_length=1.,front_length=3.,width=2.)),
        dynamic_car_state=NS(rear_axle_velocity_2d=NS(x=0.,y=5.),rear_axle_acceleration_2d=NS(x=0.,y=-1.)))

    def get_tracked_objects_at_iteration(self, i):
        objects=[NS(track_token="self",tracked_object_type=Kind.EGO)]
        if i!=2:
            objects.append(NS(track_token="actor",tracked_object_type=Kind.VEHICLE,
                           box=NS(center=NS(x=10.,y=25.+i,heading=np.pi/2),length=4.,width=2.)))
        return NS(tracked_objects=objects)


class WorldTests(unittest.TestCase):
    def test_frames_track_gaps_ego_filter_and_polygon_coverage(self):
        w=build_world(Scene(),{"scene":"recorded"})
        self.assertEqual(w["map_status"],"available")
        self.assertEqual(w["unpopulated_map_layers"],["DRIVABLE_AREA"])
        self.assertEqual(len(w["tracks"]),2)
        self.assertEqual(w["tracks"][0]["times_s"],[0.,.5])
        self.assertEqual(w["tracks"][1]["times_s"],[1.5,2.,2.5,3.,3.5,4.])
        np.testing.assert_allclose(w["tracks"][0]["poses"][0],[5.,0.,0.],atol=1e-12)
        self.assertEqual(w["geometry"],{"x_min":-1.,"x_max":3.,"y_min":-1.,"y_max":1.})

    def test_missing_map_keeps_physical_tracks_but_dac_unavailable(self):
        scene=Scene();scene.map_api=NS(get_available_map_objects=lambda: (_ for _ in ()).throw(FileNotFoundError("missing map")))
        w=build_world(scene,{})
        self.assertEqual(w["map_status"],"unavailable")
        self.assertIn("missing map",w["map_error"])
        self.assertEqual(len(w["tracks"]),2)


if __name__=="__main__":unittest.main()
