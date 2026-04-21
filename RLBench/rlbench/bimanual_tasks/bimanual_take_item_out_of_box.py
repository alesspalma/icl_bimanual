from typing import List, Tuple
import numpy as np
from collections import defaultdict
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.conditions import NothingGrasped
from rlbench.backend.task import BimanualTask

class BimanualTakeItemOutOfBox(BimanualTask):

    def init_task(self) -> None:
        item = Shape('item')
        lid = Shape('box_lid')
        self.register_graspable_objects([item])
        success_sensor = ProximitySensor('success_out_box')
        grab_lid = ProximitySensor('grab_lid')
        self.register_success_conditions([
            DetectedCondition(item, success_sensor),
            NothingGrasped(self.robot.right_gripper),])
            # DetectedCondition(self.robot.left_arm.get_tip(), grab_lid),
            # DetectedCondition(lid, grab_lid)])
        
        self.waypoint_mapping = defaultdict(lambda: 'right')
        for i in range(4):
            self.waypoint_mapping[f'waypoint{i}'] = 'left'

    def init_episode(self, index: int) -> List[str]:
        return ['take item out of box',
                'open the box and take the item out',
                'put the item found inside the box on the table',
                'set the item down on the table',
                'pick up the item from the box and put it down',
                'grasp the edge of the box lid to open it, then grasp the item'
                ', lifting up out of the box and leaving it down on the '
                'table']

    def variation_count(self) -> int:
        return 1
    
    # def is_static_workspace(self):
    #     return True
    
    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, -np.pi / 8], [0, 0, np.pi / 8]
