import numpy as np
import pandas as pd
from glob import glob
import os
from tqdm import tqdm
import time

'''
视频路径（按地区分别，例如目前路径为sheyang）
'''
video_root_directory = 'D://Datasets//facial_videos//sheyang//'
'''
特征提取后放置路径
'''
original_features_dir = 'D://Datasets//original_features//sheyang//'
'''
1.指定FeatureExtraction.exe的根目录（win）
2.指定FeatureExtraction的根目录（类Unix）
'''
openface_executive_root_dir = "D://OpenFace_2.2.0_win_x64//"

'''
阅读任务视频的路径
'''
reading_dir = 'reading_videos//'

'''
阅读任务视频原始特征的最终路径
'''
reading_out_dir = original_features_dir + reading_dir

reading_cust_id_list = []
'''
1.注意视频名字的匹配问题，例如read_video_*.mp4
2.由于每次筛选活动存在的命名问题，建议先手动筛选无效视频、cust_id重复视频以及测试视频
3.这一步耗费的时间取决于总视频数量与长短，还和cpu性能有关
'''
for g in tqdm(glob(video_root_directory + reading_dir + 'read_video_*.mp4')):
    file = g.split('\\')[-1]  # 取出文件名以及后缀
    file_name = file.split('.mp4')[0]  # 获取文件名
    cust_id = file_name.split('_')[2]  # 获取文件名中的cust_id,可能需要根据实际文件的命名进行修改
    real_out_dir = reading_out_dir + 'sub_' + str(cust_id) + '/'  # 按照cust_id分类每个参与者的特征创建对应文件夹
    os.system('{0}FeatureExtraction -f "{1}" -out_dir "{2}"'.format(openface_executive_root_dir, g, real_out_dir))

'''
1.计算凝视相关、头部姿态、面部动作单元、点分布模型、面部标志点与眼部标志点这六种帧级别原始特征的均值、中值、最大值、标准差、方差
'''
root_dir = 'D:\\Datasets\\original_features\\sheyang\\'
reading_export_dir = 'D:\\Datasets\\original_features\\sheyang\\reading_videos\\statistics_features\\'
reading_dir = 'reading_videos\\'

reading_gaze_features_list = []
reading_pose_features_list = []
reading_aus_features_list = []
reading_pdm_features_list = []
reading_el_features_list = []
reading_fl_features_list = []

sub_id_list = []
'''
计算Gaze相关的统计特征
'''
for g in tqdm(glob.glob(root_dir + reading_dir + 'sub_*\\*_gaze.xlsx')):
    df_gaze = pd.read_csv(g)
    df_gaze = df_gaze.loc[(df_gaze["confidence"] > 0.75) & (df_gaze["success"] == 1)]  # 筛选掉一些置信度较低的帧特征
    df_gaze_diff = df_gaze.diff()
    df_gaze = df_gaze.loc[:, 'gaze_0_x':'gaze_angle_y']

    sub_id = g.split('_')[-2]
    sub_id_list.append(sub_id)

    gaze_mean = df_gaze.mean()
    gaze_median = df_gaze.median()
    gaze_max = df_gaze.max()
    gaze_std = df_gaze.std()
    gaze_var = df_gaze.var()
    gaze_features = pd.concat([gaze_mean,
                               gaze_median,
                               gaze_max,
                               gaze_std,
                               gaze_var], ignore_index=True)
    gaze_features = gaze_features.tolist()
    reading_gaze_features_list.append(gaze_features)

'''
计算Pose相关特征
'''
for p in tqdm(glob.glob(root_dir + reading_dir + 'sub_*\\*_pose.xlsx')):
   df_pose = pd.read_csv(p)
   df_pose = df_pose.loc[(df_pose["confidence"] > 0.75) & (df_pose["success"] == 1)]
   df_pose = df_pose.loc[:, 'pose_Tx':'pose_Rz']
   pose_mean = df_pose.mean()
   pose_median = df_pose.median()
   pose_max = df_pose.max()
   pose_std = df_pose.std()
   pose_var = df_pose.var()
   pose_features = pd.concat([pose_mean,
                              pose_median,
                              pose_max,
                              pose_std,
                              pose_var], ignore_index=True)
   pose_features = pose_features.tolist()
   reading_pose_features_list.append(pose_features)

'''
计算AUs特征
'''
for a in tqdm(glob.glob(root_dir + reading_dir + 'sub_*\\*_aus.xlsx')):
    df_aus = pd.read_csv(a)
    df_aus = df_aus.loc[(df_aus["confidence"] > 0.75) & (df_aus["success"] == 1)]
    # df_aus = df_aus.loc[:, ['AU04_r','AU05_r','AU06_r','AU07_r','AU10_r','AU12_r']]
    df_aus = df_aus.loc[:, 'AU01_r':'AU45_r']
    aus_mean = df_aus.mean()
    aus_median = df_aus.median()
    aus_max = df_aus.max()
    aus_std = df_aus.std()
    aus_var = df_aus.var()
    aus_features = pd.concat([aus_mean,
                              aus_median,
                              aus_max,
                              aus_std,
                              aus_var], ignore_index=True)
    aus_features = aus_features.tolist()
    reading_aus_features_list.append(aus_features)

'''
计算PDM特征
'''
for p in tqdm(glob.glob(root_dir + reading_dir +'sub_*\\*_pdm.xlsx')):
    df_pdm = pd.read_csv(p)
    df_pdm = df_pdm.loc[(df_pdm["confidence"] > 0.75) & (df_pdm["success"] == 1)]
    df_pdm = df_pdm.loc[:, 'p_scale':'p_33']
    pdm_mean = df_pdm.mean()
    pdm_median = df_pdm.median()
    pdm_max = df_pdm.max()
    pdm_std = df_pdm.std()
    pdm_var = df_pdm.var()
    pdm_features = pd.concat([pdm_mean,
                              pdm_median,
                              pdm_max,
                              pdm_std,
                              pdm_var], ignore_index=True)
    pdm_features = pdm_features.tolist()
    reading_pdm_features_list.append(pdm_features)
'''
计算眼部标志点特征
'''
for e in tqdm(glob.glob(root_dir + reading_dir +'sub_*\\*_all.csv')):  # 这里直接使用特征未分割的全部特征文件，若要修改，需要在原始特征提取处进行同步修改
    df_el = pd.read_csv(e)
    df_el = df_el.loc[(df_el[" confidence"] > 0.75) & (df_el[" success"] == 1)]
    df_el = df_el.loc[:, ' eye_lmk_x_0':' eye_lmk_Z_55']
    el_mean = df_el.mean()
    el_median = df_el.median()
    el_max = df_el.max()
    el_std = df_el.std()
    el_var = df_el.var()
    el_features = pd.concat([el_mean,
                              el_median,
                              el_max,
                              el_std,
                              el_var], ignore_index=True)
    el_features = el_features.tolist()
    reading_el_features_list.append(el_features)

'''
计算面部标志点特征
'''
for ef in tqdm(glob.glob(root_dir + reading_dir +'sub_*\\*_all.csv')):  # 同眼部标志点
    df_fl = pd.read_csv(ef)
    df_fl = df_fl.loc[(df_fl[" confidence"] > 0.75) & (df_fl[" success"] == 1)]
    df_fl = df_fl.loc[:, ' x_0':' Z_67']
    fl_mean = df_fl.mean()
    fl_median = df_fl.median()
    fl_max = df_fl.max()
    fl_std = df_fl.std()
    fl_var = df_fl.var()
    fl_features = pd.concat([fl_mean,
                              fl_median,
                              fl_max,
                              fl_std,
                              fl_var], ignore_index=True)
    fl_features = fl_features.tolist()
    reading_fl_features_list.append(fl_features)


gaze_features_df = pd.DataFrame(reading_gaze_features_list, index=sub_id_list)
pose_features_df = pd.DataFrame(reading_pose_features_list, index=sub_id_list)
aus_features_df = pd.DataFrame(reading_aus_features_list, index=sub_id_list)
pdm_features_df = pd.DataFrame(reading_pdm_features_list, index=sub_id_list)
el_features_df = pd.DataFrame(reading_el_features_list, index=sub_id_list)
fl_features_df = pd.DataFrame(reading_fl_features_list, index=sub_id_list)


'''
统计特征按照不同类别的单一统计特征导出为单独文件，可以按需求进行组合使用
'''
gaze_features_df.to_excel(reading_export_dir + 'reading_gaze_features.xlsx')
pose_features_df.to_excel(reading_export_dir + 'reading_pose_features.xlsx')
aus_features_df.to_excel(reading_export_dir + 'reading_aus_features.xlsx')
pdm_features_df.to_excel(reading_export_dir + 'reading_pdm_features.xlsx')
el_features_df.to_excel(reading_export_dir + 'reading_el_features.xlsx')
fl_features_df.to_excel(reading_export_dir +'reading_fl_features.xlsx')
