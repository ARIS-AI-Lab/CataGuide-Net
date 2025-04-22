
params = {
    "is_training": True,
    "batch_size": 16,
    "is_train_from_begin": False,
    "freeze_generate_model": False,
    "checkpoints_path": 'checkpoints_train',
    "clip_label": r'C:\Users\Charl\PycharmProjects\Video_Classification\data\merged.csv',
    "num_classes": 10, # included non-tool type
    "total_stage": 10,
    "num_epoch": 100,
    "surg_levels": 2,
    "mix_alpha": 0.2,
    "num_sample": 10,
    "max_checkpoints": 5,
    "class_mapping":{
        0: 0,
        1: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5,
        8: 6,
        9: 7,
        10: 8,
        11: 9
    },
    "model_param":{
        "pretrain_model_path": r'C:\Users\Charl\Video_Classification\trajectory_RL\checkpoints_pretrain\models_step_16000.pth',
        "max_learning_rate": 0.0008,
        "base_learning_rate": 0.0000004,
        "weight_decay": 1e-5,
        "save_checkpoint_epochs": 10,
        "warmup_epochs": 0,
        "hold_epochs": 0,
    },
    "pipline_params": {
        "npy_train_path": r'C:\Users\Charl\PycharmProjects\Video_Classification\npys_train',
        # "npy_train_path": r"D:\npys_test",
        "train_trajectory_path": r"D:\trajectory_folder",
        # "train_trajectory_path": r"D:\trajectory_folder_test",
    }
}