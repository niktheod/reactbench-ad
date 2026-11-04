"""ReCogDrive agent input from a NAVSIM annotation and one front-camera image."""
from pathlib import Path

import numpy as np
from navsim.common.dataclasses import AgentInput, Camera, Cameras, EgoStatus, Lidar


def make_recogdrive_agent_input(
    image_path,
    velocities,            # list of 4 [vx, vy]
    accelerations,         # list of 4 [ax, ay]
    history_trajectories,  # list of 4 [x, y, heading]
    navigation_command,    # one-hot, e.g. [0, 1, 0]
):
    assert len(velocities) == 4
    assert len(accelerations) == 4
    assert len(history_trajectories) == 4
    assert len(navigation_command) == 4

    ego_statuses = []
    cameras = []
    lidars = []

    for i in range(4):
        ego_statuses.append(
            EgoStatus(
                ego_pose=np.array(history_trajectories[i], dtype=np.float32),
                ego_velocity=np.array(velocities[i], dtype=np.float32),
                ego_acceleration=np.array(accelerations[i], dtype=np.float32),
                driving_command=np.array(navigation_command, dtype=np.int64),
                in_global_frame=False,
            )
        )

        # ReCogDrive only uses cameras[-1].cam_f0.image
        cameras.append(
            Cameras(
                cam_f0=Camera(image=Path(image_path)),
                cam_l0=Camera(),
                cam_l1=Camera(),
                cam_l2=Camera(),
                cam_r0=Camera(),
                cam_r1=Camera(),
                cam_r2=Camera(),
                cam_b0=Camera(),
            )
        )

        lidars.append(Lidar())

    return AgentInput(
        ego_statuses=ego_statuses,
        cameras=cameras,
        lidars=lidars,
    )
