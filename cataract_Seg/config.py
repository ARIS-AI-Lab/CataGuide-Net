params = {
    "is_training": True,
    "is_train_from_begin": False,
    "detect_eyes": True,
    "num_epoch": 400,
    "batch_size": 6,
    "max_checkpoints": 6,
    "model_param":{
        "model_dir": './checkpoint_fine_tune',
        "pretrain_model_path": r'C:\Users\Charl\PycharmProjects\cataract_Seg\checkpoint_fine_tune\epoch_850.pth',
        "use_pretrained": False,
        "freeze_decoder": False,
        "max_learning_rate": 0.0001,
        "base_learning_rate": 0.0000004,
        "weight_decay": 1e-4,
        "save_checkpoint_epochs": 10,
        "warmup_epochs": 5,
        "hold_epochs": 100,
        "max_kpts": 2,
        "tool_and_eyes_class":{'Capsulorhexis Cystotome': 1, 'Capsulorhexis Forceps': 2, 'Slit Knife': 3, 'Gauge': 4,
                               'Incision Knife': 5, 'Irrigation-Aspiration': 6, 'Katena Forceps': 7, 'Spatula': 8,
                               'Lens Injector': 9, 'Phacoemulsification Tip': 10, 'Lens': 11, 'Cornea': 12,
                               'Pupil': 13, 'cornea1': 14, 'pupil1': 15},
    },
    "input_pipeline_params":{
        "image_size": [640, 640],
        "dataset_path": r'C:\Users\Charl\Downloads\cataract-1k\Annotations\Generated_Dataset\train',
    }
}
