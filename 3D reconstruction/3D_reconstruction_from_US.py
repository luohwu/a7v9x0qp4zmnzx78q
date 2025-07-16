import os
import time

import pandas as pd
from skimage.io import imread

from Tools.converter import *
from multiprocessing import Pool



def construct_pcd_from_df(df_chunk):
    df_chunk = df_chunk.reset_index(drop=True)
    label_pcd_merged = []
    for index, row in df_chunk.iterrows():
        label = imread(row['label_path'], as_gray=True)
        if 255 not in np.unique(label):
            continue

        # There are two types of tracking data in `tracking.csv`: original (i.e., x) and optimized (i.e., x_optimized). Both are already temporally synchronized, so they
        # can be used directly

        # using original tracking data
        T_tracking = vectorToMatrix([row['x'], row['y'], row['z']],
                                    [row['euler_x'], row['euler_y'], row['euler_z']])

        # using optimized tracking data
        # T_tracking = vectorToMatrix([row['x_optimized'], row['y_optimized'], row['z_optimized']],
        #                             [row['euler_x_optimized'], row['euler_y_optimized'], row['euler_z_optimized']])

        # the coordinate system of the ultrasound image is defined at the top-left corner
        T_scale = np.eye(4)
        T_scale[0, 0] = row['scale_X']
        T_scale[1, 1] = row['scale_Y']

        # keep only bone pixels
        rows_index, cols_index = np.where(abs(label) > 0)
        if len(rows_index) < 2:
            return None

        xyz_pixels = np.stack([cols_index + 0.5, rows_index + 0.5, np.zeros_like(rows_index), np.ones_like(rows_index)],
                              axis=1).T # homogenous coordinates with [x,y,z,1]. z is always 0 for the image plane

        calibrationT = vectorToMatrix(row['calibration_t'], row['calibration_euler'])


        # T_scale @ xyz_pixels: convert pixels to points (unit:mm) in the ImagePlane coor
        # calibrationT @ T_scale @ xyz_pixels: pixels(ImagePlane) -> Points (ImagePlane) -> Points (Marker space)
        # T_tracking @ calibrationT @ T_scale @ xyz_pixels: pixels(ImagePlane) -> Points (ImagePlane) -> Points (Marker space) -> Points (CT space)
        T_img_2_world = T_tracking @ calibrationT @ T_scale
        xyz_world = (T_img_2_world @ xyz_pixels)


        label_pcd_merged.append(xyz_world.T)
    if label_pcd_merged:
        return np.concatenate(label_pcd_merged)
    return None

def main_foot_ankle():
    specimen_names = [
        "specimen00",
        "specimen01",
        "specimen02",
        "specimen03",
        "specimen04",
        "specimen05",
        "specimen06",
        "specimen07",
        "specimen08",
        "specimen09",
        "specimen10",
        "specimen11",
        "specimen12",
        "specimen13",
        "specimen14"
    ]


    # calibration results. Note that the special details are the same for all sweeps.
    calibration_t = [26.44694442, -0.52572229, 128.00100047] # unit: mm
    calibration_euler = [92.48865621, -0.46874914, 179.2277322] # euler angles in degrees
    scale_X = 0.05392 # mm/pixel
    scale_Y = 0.05392 # mm/pixel

    dataset_root_folder = "../data/AI_Ultrasound_dataset"
    specimens_involved = [1,3,4,5,6,9,10,11,12,13,14]
    for specimen_id in specimens_involved:
        specimen_name = specimen_names[specimen_id]

        # read CT model data
        CT_model_mesh=o3d.io.read_triangle_mesh(os.path.join(dataset_root_folder,specimen_name,"CT_bone_model.stl"))
        CT_model_pcd=CT_model_mesh.sample_points_uniformly(1000000)

        for record_id in range(1,15):
            dataFolder=os.path.join(dataset_root_folder,specimen_name,f"record{record_id:02d}")
            assert os.path.isdir(dataFolder),"dataset folder does not exist"

            label_folder = os.path.join(dataFolder, 'Labels')
            timestamps_target = [int(file.split('_')[0]) for file in os.listdir(label_folder) if
                                 file.endswith("_label.png")]
            tracking_df = pd.read_csv(os.path.join(dataFolder, 'tracking.csv'))


            tracking_df = tracking_df[tracking_df['timestamp'].isin(timestamps_target)]

            # update the details for each frame
            tracking_df['label_path'] = tracking_df['timestamp'].apply(
                lambda x: os.path.join(label_folder, f"{x}_label.png"))
            tracking_df['calibration_t'] = [calibration_t] * len(tracking_df)
            tracking_df['calibration_euler'] = [calibration_euler] * len(tracking_df)
            tracking_df['scale_X'] = [scale_X] * len(tracking_df)
            tracking_df['scale_Y'] = [scale_Y] * len(tracking_df)

            # Chunking the DataFrame
            num_processes = 6
            pool = Pool(processes=num_processes)
            chunk_size = int(np.ceil(len(tracking_df) / num_processes))
            df_chunks = [tracking_df.iloc[i:i + chunk_size] for i in range(0, len(tracking_df), chunk_size)]

            # Processing in parallel
            results = pool.map(construct_pcd_from_df, df_chunks)
            pool.close()
            pool.join()

            # Combining results
            xyz_with_feature=np.concatenate(results)
            xyz = xyz_with_feature[:,:3]
            reconstruction_pcd = o3d.geometry.PointCloud()
            reconstruction_pcd.points = o3d.utility.Vector3dVector(xyz)


            dis=np.asarray(reconstruction_pcd.compute_point_cloud_distance(CT_model_pcd))
            print(f"CD distance from US-reconstruction to CT-pcd: {dis.mean()}")
            o3d.visualization.draw_geometries([reconstruction_pcd,CT_model_mesh])



if __name__ == '__main__':
    main_foot_ankle()
