import torch
import os

###########################
# 基本配置
###########################

#在确保所有gpu可用的前提下，可设置多个gpu，否则torch.cuda.is_available()显示为false
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
osenvs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
gpu_ids = [*range(osenvs)]
max_epoch = 5
loss_freq = 40
mode = 'Both'
# pretrained_path = 'models/xception-b5690688.pth'
device = torch.device("cuda")
resize = 380

###########################
#  路径
###########################
celeb_real_root = r"/hy-tmp/celeb-df/celeb-real"
celeb_syn_root = r"/hy-tmp/celeb-df/celeb-syn"
celeb_csv_root = r"/hy-tmp/celeb-df/csv"

dfdc_root = r"/hy-tmp/Dfdc"

model_path_name = r"G:\我的云端硬盘\models\F3\test_6(git_version)\model2.pth"
xception_pretrained_path =r"D:\DeepFakeProject_in_D\deepfake_project\our_code\f3net\models\xception-b5690688.pth"
efficient_pretrained_path=r"/hy-nas/f3-net/utils/deepware.pt"

dfdc_syn_root=r"/hy-tmp/Dfdc/Dfdc_syn"
dfdc_real_root=r"/hy-tmp/Dfdc/Dfdc_real"
dfdc_csv_root=r"/hy-tmp/Dfdc/csv"

ff_f2f_real_root = r"/hy-tmp/FF_f2f/FF_real"
ff_f2f_syn_root = r"/hy-tmp/FF_f2f/FF_f2f"
ff_f2f_csv_root = r"/hy-tmp/FF_f2f/csv"


ff_df_real_root = r"/hy-tmp/FF_df/FF_real"
ff_df_syn_root = r"/hy-tmp/FF_df/FF_df"
ff_df_csv_root = r"/hy-tmp/FF_df/csv"

ff_fs_real_root = r"/hy-tmp/FF_fs/FF_real"
ff_fs_syn_root = r"/hy-tmp/FF_fs/FF_fs"
ff_fs_csv_root = r"/hy-tmp/FF_fs/csv"

ff_nt_real_root = r"/hy-tmp/FF_nt/FF_real"
ff_nt_syn_root = r"/hy-tmp/FF_nt/FF_nt"
ff_nt_csv_root = r"/hy-tmp/FF_nt/csv"
