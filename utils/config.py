import yaml


class Namespace(object):
    pass


config = Namespace()


def set_config(file):
    with open(file, 'r') as f:
        _config = yaml.load(f, Loader=yaml.FullLoader)

    # Env
    config.device = _config['ENV']['DEVICE']
    config.gpu_num = _config['ENV']['GPU_NUM']

    # Media
    config.media_source = str(_config['MEDIA']['SOURCE'])
    config.media_opt_auto = _config['MEDIA']['OPT_AUTO']
    config.media_fourcc = _config['MEDIA']['FOURCC']
    config.media_width = _config['MEDIA']['WIDTH']
    config.media_height = _config['MEDIA']['HEIGHT']
    config.media_fps = _config['MEDIA']['FPS']
    config.media_realtime = _config['MEDIA']['REALTIME']
    config.media_bgr = _config['MEDIA']['BGR']
    config.media_enable_param = _config['MEDIA']['ENABLE_PARAM']
    config.media_cv2_params = _config['MEDIA']['CV_PARAM']

    # Det
    config.det_model_type = _config['DET']['MODEL_TYPE']
    config.det_model_path = _config['DET']['DET_MODEL_PATH']
    config.det_half = _config['DET']['HALF']
    config.det_conf_thres = _config['DET']['CONF_THRES']
    config.det_obj_classes = _config['DET']['OBJ_CLASSES']
    # YOLO
    config.yolo_img_size = _config['DET']['YOLO']['IMG_SIZE']
    config.yolo_nms_iou = _config['DET']['YOLO']['NMS_IOU']
    config.yolo_agnostic_nms = _config['DET']['YOLO']['AGNOSTIC_NMS']
    config.yolo_max_det = _config['DET']['YOLO']['MAX_DET']

    # SPIN
    config.spin_device = _config['SPIN']['DEVICE']
    config.spin_smpl_mean_params = _config['SPIN']['SMPL_MEAN_PARAMS']
    config.spin_checkpoint = _config['SPIN']['CHECKPOINT']
    config.spin_smpl_model_dir = _config['SPIN']['SMPL_MODEL_DIR']
    config.spin_joint_regressor_extra = _config['SPIN']['JOINT_REGRESSOR_EXTRA']

    # VIBE
    config.vibe_device = _config['VIBE']['DEVICE']
    config.vibe_data_dir = _config['VIBE']['DATA_DIR']
    config.vibe_use_3dpw = _config['VIBE']['USE_3DPW']

    # HYBRIK
    config.hybrik_ckpt = _config['HYBRIK']['CKPT']
    config.hybrikx = _config['HYBRIK']['X']
    config.hybrik_save_img = _config['HYBRIK']['SAVE_IMG']
    config.hybrik_save_orig_img = _config['HYBRIK']['SAVE_ORIG_IMG']
    config.hybrik_save_mesh_img = _config['HYBRIK']['SAVE_MESH_IMG']
    config.hybrik_save_vid = _config['HYBRIK']['SAVE_VID']
    config.hybrik_save_mesh_vid = _config['HYBRIK']['SAVE_MESH_VID']
    config.hybrik_draw_heatmap = _config['HYBRIK']['DRAW_HEATMAP']

    # Logger
    if 'LOG' not in _config:
        raise ValueError("LOG_LEVEL is missing in config file")
    config.log_level = _config['LOG']['LOG_LEVEL']
    config.logger_name = _config['LOG']['LOGGER_NAME']
    config.console_log = _config['LOG']['CONSOLE_LOG']
    config.console_log_interval = _config['LOG']['CONSOLE_LOG_INTERVAL']
    config.file_log = _config['LOG']['FILE_LOG']
    config.file_log_dir = _config['LOG']['FILE_LOG_DIR']
    config.file_log_counter = _config['LOG']['FILE_LOG_COUNTER']
    config.file_log_rotate_time = _config['LOG']['FILE_LOG_ROTATE_TIME']
    config.file_log_rotate_interval = _config['LOG']['FILE_LOG_ROTATE_INTERVAL']


def get_config():
    return config
