class CFG:
    area_threshold = 32 * 32
    iou_threshold = 0.5
    score_threshold = 0.0
    logit_threshold = 0.0
    similarity_threshold = 0.5

    down_sample_ratios = [16, 8, 4]
    top_k = 512
    chunk_k = 1024

    save_cache = True

    data_dir = "./data"
    cache_dir = "./cache"
    model_dir = "./model"
    output_dir = "./output"
