
# with open('/home/eva_sean/src/edge-ai/lab4/DepthAnythingV2/metric_depth/dataset/splits/kitti/val.txt', 'r') as f:
#     lines = f.readlines()

# with open('kitti_val.txt', 'w') as f:
#     new_lines = []    
#     for line in lines:
#         line = line.strip()
#         line = line.replace('/mnt/bn/liheyang/Kitti/', '')
#         f.write(line + '\n')
        
with open('kitti_val.txt', 'r') as f:
    new_lines = f.readlines()
    scene_set = set()
    for line in new_lines:
        scene = '/'.join(line.split(' ')[0].split('/')[1:3])
        scene_set.add(scene)   
    scene_list = sorted(list(scene_set))
    
with open('kitti_val_scene_list.txt', 'w') as f:
    for scene in scene_list:
        f.write(scene + '\n')
    
    
    