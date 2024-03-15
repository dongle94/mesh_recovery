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
    config.media_source = _config['MEDIA']['SOURCE']
    config.media_opt_auto = _config['MEDIA']['OPT_AUTO']
    config.media_fourcc = _config['MEDIA']['FOURCC']
    config.media_width = _config['MEDIA']['WIDTH']
    config.media_height = _config['MEDIA']['HEIGHT']
    config.media_fps = _config['MEDIA']['FPS']
    config.media_realtime = _config['MEDIA']['REALTIME']
    config.media_bgr = _config['MEDIA']['BGR']

    # Det
    config.det_model_type = _config['DET']['MODEL_TYPE']
    config.det_model_path = _config['DET']['DET_MODEL_PATH']
    config.det_half = _config['DET']['HALF']
    config.det_conf_thres = _config['DET']['CONF_THRES']
    config.det_obj_classes = _config['DET']['OBJ_CLASSES']

    # YOLOV5
    config.yolov5_img_size = _config['YOLOV5']['IMG_SIZE']
    config.yolov5_nms_iou = _config['YOLOV5']['NMS_IOU']
    config.yolov5_agnostic_nms = _config['YOLOV5']['AGNOSTIC_NMS']
    config.yolov5_max_det = _config['YOLOV5']['MAX_DET']

    # TRACKER
    config.track_use_encoder = _config['TRACK']['TRACK_USE_ENCODER']
    config.track_model_type = _config['TRACK']['TRACK_MODEL_TYPE']
    config.track_model_path = _config['TRACK']['TRACK_MODEL_PATH']
    config.track_half = _config['TRACK']['TRACK_HALF']

    # HYBRIK
    config.hybrik_ckpt = _config['HYBRIK']['CKPT']
    config.hybrik_cfg = _config['HYBRIK']['CFG']
    config.hybrik_save_img = _config['HYBRIK']['SAVE_IMG']
    config.hybrik_save_orig_img = _config['HYBRIK']['SAVE_ORIG_IMG']
    config.hybrik_save_mesh_img = _config['HYBRIK']['SAVE_MESH_IMG']
    config.hybrik_save_vid = _config['HYBRIK']['SAVE_VID']
    config.hybrik_save_mesh_vid = _config['HYBRIK']['SAVE_MESH_VID']
    config.hybrik_draw_heatmap = _config['HYBRIK']['DRAW_HEATMAP']

    # Logger
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
