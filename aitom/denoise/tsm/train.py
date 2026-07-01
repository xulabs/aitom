import argparse
import json
import torch
import logging
import os
import csv
from easydict import EasyDict
from models.score_func import ScoreFuncModel
from dataset.rgb_dataset import RgbDataset
from dataset.mrc_dataset import MRCDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils.data_process import train_data_process_dsm,train_data_process_tsm,test_data_process,postprocess, compute_clean_stats
from utils.utils import count_parameters,loss_func,save_output,compute_metrics, save_image, collate_mrc, remap_to_raw_range
from utils.CleanTargetProjector import CleanTargetProjector


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def metrics_eval(dataset,model,cfg,device,visualization_path):
    model.eval()
    metrics = {
        "psnr": [], "ssim": [], "nmse": [], "mse": [],
        "n_psnr": [], "n_ssim": [], "n_nmse": [], "n_mse": []
    }
    with torch.no_grad():
        for batch in dataset:
            test_data_dict = test_data_process(batch, cfg, device)
            scores = model(test_data_dict)
            test_data_dict['scores'] = scores
            denoised_images = postprocess(test_data_dict,cfg, model)
            test_data_dict['denoised_images'] = denoised_images
            save_output(test_data_dict,visualization_path)
            sub_metrics = compute_metrics(test_data_dict)

            for k in metrics:
                metrics[k].extend(sub_metrics[k])
    for k in metrics:
        metrics[k] = np.asarray(metrics[k], dtype=np.float32)

    return metrics

def compute_val_loss(val_dataloader, clean_dataloader, projector, model, cfg, global_step, device, logger):
    model.eval()
    total_loss = 0.0
    total_dsm  = 0.0
    total_tsm  = 0.0
    total_steps = 0
    val_picked = 0
    val_total = 0
    val_sim_sum = 0.0

    with torch.no_grad():
        for batch in val_dataloader:
            val_dict_dsm = train_data_process_dsm(batch, cfg.model_config, device)
            val_dict_tsm = train_data_process_tsm(batch, cfg.model_config, device)

            raw_noisy = batch["noisy_image"].float().to(device)
            if raw_noisy.ndim == 4 and raw_noisy.shape[1] != 1:
                raw_noisy = raw_noisy.permute(0, 3, 1, 2)

            # Use weighted target for validation (single or dual-domain depending on --mix)
            y_tsm, best_sim = projector.get_weighted_target(raw_noisy, tau=0.5)
            val_dict_tsm["clean_images"] = y_tsm
            val_dict_tsm["similarity"] = best_sim.detach()

            outputs_dsm = model(val_dict_dsm, mode="val")
            outputs_tsm = model(val_dict_tsm, mode="val")

            loss_dict = loss_func(val_dict_dsm, val_dict_tsm, outputs_dsm, outputs_tsm, cfg, global_step, return_components=True)

            total_loss += loss_dict["loss"].item()
            total_dsm  += loss_dict["dsm"].item()
            total_tsm  += loss_dict["tsm"].item()
            val_picked += loss_dict["num_picked"]
            val_total  += loss_dict["batch_size"]
            val_sim_sum += loss_dict["mean_similarity"] * loss_dict["batch_size"]
            total_steps += 1
    pick_ratio = val_picked / max(val_total, 1)
    mean_similarity = val_sim_sum / max(val_total, 1)
    logger.info(
        f"[VAL-PICK] picked {val_picked}/{val_total} "
        f"({pick_ratio:.3f}), mean_sim={mean_similarity:.3f}"
    )
    model.train()
    return {"total": total_loss / total_steps, "dsm":   total_dsm  / total_steps, "tsm":   total_tsm  / total_steps}

def plot_loss_curves(train_total, val_total, train_dsm=None, val_dsm=None, train_tsm=None, val_tsm=None, save_path=None):
    fig, ax_left = plt.subplots(figsize=(10, 5))
    ax_right = ax_left.twinx()

    # -------- Left Y-axis: Total + DSM --------
    if train_total:
        steps, losses = zip(*train_total)
        ax_left.plot(steps, losses, label="Train Total", linewidth=2)

    if val_total:
        steps, losses = zip(*val_total)
        ax_left.plot(steps, losses, label="Val Total", linewidth=2, linestyle="--")

    if train_dsm:
        steps, losses = zip(*train_dsm)
        ax_left.plot(steps, losses, label="Train DSM", alpha=0.7)

    if val_dsm:
        steps, losses = zip(*val_dsm)
        ax_left.plot(steps, losses, label="Val DSM", linestyle="--", alpha=0.7)

    ax_left.set_xlabel("Training Step")
    ax_left.set_ylabel("Total / DSM Loss")
    ax_left.grid(True)

    # -------- Right Y-axis: TSM --------
    if train_tsm:
        steps, losses = zip(*train_tsm)
        ax_right.plot(steps, losses, label="Train TSM", alpha=0.7)

    if val_tsm:
        steps, losses = zip(*val_tsm)
        ax_right.plot(steps, losses, label="Val TSM", linestyle="--", alpha=0.7)

    ax_right.set_ylabel("TSM Loss")

    # -------- Legend handling (combine both axes) --------
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        loc="upper right"
    )

    plt.title("Total / DSM (Left) vs TSM (Right) Loss Curves")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.close()

def plot_similarity_curve(wt_hist, save_path=None):
    plt.figure(figsize=(10, 4))

    if wt_hist:
        steps, wts = zip(*wt_hist)
        wts = [float(w) for w in wts]
        plt.plot(steps, wts, label="w_t / Similarity", color="black", linestyle=":")

    plt.xlabel("Training Step")
    plt.ylabel("Similarity / w_t")
    plt.ylim(0.0, 1.05)
    plt.title("Similarity (w_t) Over Training")
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path)

    plt.close()


def sliding_window_inference(image, model, device, window_size=192, overlap=96, cfg=None, projector=None, csv_writer=None, file_name=None):
    total_picked = 0
    total_windows = 0
    _, C, H, W = image.shape
    stride = window_size - overlap

    pad_h = (stride - H % stride) % stride
    pad_w = (stride - W % stride) % stride
    image_padded = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, H_pad, W_pad = image_padded.shape

    output = torch.zeros_like(image_padded, device=device)
    count_map = torch.zeros_like(image_padded, device=device)

    for y in range(0, H_pad - window_size + 1, stride):
        for x in range(0, W_pad - window_size + 1, stride):

            patch = image_padded[:, :, y:y+window_size, x:x+window_size]
            patch_hwc = patch.permute(0, 2, 3, 1).contiguous().cpu().numpy() # CHW → HWC for the model input (same as test_data_process)
            #print(f"patch_hwc min={patch_hwc.min():.4f}, max={patch_hwc.max():.4f}")

            noiser_patch = patch.to(device) # CHW tensor for postprocess

            alpha = cfg.poisson_alpha
            sigma_g = cfg.gaussian_sigma

            poisson_var = torch.clamp(noiser_patch, min=0.0) / alpha
            gaussian_var = sigma_g ** 2
            total_std = torch.sqrt(poisson_var + gaussian_var + 1e-8)

            patch_dict = {
                "images": patch_hwc,              # unused but fine
                "noisy_images": patch_hwc,        # model expects this
                "noiser_images": noiser_patch,    # postprocess expects this
                "total_std": total_std,
                "file_names": ["patch"]
            }

            with torch.no_grad():
                scores = model(patch_dict)

            patch_dict["scores"] = scores
            #print(f"scores min={scores.min().item():.4f}, max={scores.max().item():.4f}")
            den_patch = postprocess(patch_dict, cfg, model)  # returns (b, h, w, c) numpy
            den_patch = torch.tensor(den_patch).permute(0, 3, 1, 2).float().to(device) # HWC numpy → CHW tensor
            
            if projector is not None:
                sims, best_sim = projector.similarity_all_clean(patch, return_all=False)
                picked = best_sim.item() >= 0.8

                total_windows += 1
                if picked:
                    total_picked += 1
            output[:, :, y:y+window_size, x:x+window_size] += den_patch # Accumulate patch
            count_map[:, :, y:y+window_size, x:x+window_size] += 1

    output = output / count_map # blend
    if csv_writer is not None:
        csv_writer.writerow([
            file_name,
            total_picked,
            total_windows,
            total_picked / max(total_windows, 1)
        ])
    return output[:, :, :H, :W]

def main(args: argparse.Namespace):
    
    with open(args.config_json, "r", encoding="utf-8") as f:
        cfg = EasyDict(json.load(f))

    dataset_tag = str(args.dataset)
    category_tag = "TSM"

    train_root = os.path.join("results", "train", dataset_tag, category_tag)
    train_ckpt_dir = os.path.join(train_root, "ckpt")
    train_loss_dir = os.path.join(train_root, "loss")
    train_sim_dir = os.path.join(train_root, "sim")
    train_vis_dir = os.path.join(train_root, "visualization")

    test_root = os.path.join("results", "test", dataset_tag)
    compare_root = os.path.join("results", "compare", dataset_tag, category_tag)

    log_dir = os.path.join("logs", dataset_tag, category_tag)

    os.makedirs(train_ckpt_dir, exist_ok=True)
    os.makedirs(train_loss_dir, exist_ok=True)
    os.makedirs(train_sim_dir, exist_ok=True)
    os.makedirs(train_vis_dir, exist_ok=True)
    os.makedirs(test_root, exist_ok=True)
    os.makedirs(compare_root, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir,
        f"{dataset_tag}_{cfg.running_config.model_type}_train_TSM_10A_5A_3A_2.log"
    )
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s",
        datefmt="%m/%d %I:%M:%S %p",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.info("Full configuration for running:\n%s", json.dumps(cfg, indent=2))

    if args.device == -1:
        device = torch.device("cpu")
        logger.info("Using CPU (device = -1).")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.device}")
            logger.info(f"Using GPU cuda:{args.device} - {torch.cuda.get_device_name(args.device)}")
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available, fallback to CPU.")
    

    train_dataset = MRCDataset(root_dir=args.data_dir, mode="train", dataset=args.dataset)
    val_dataset   = MRCDataset(root_dir=args.data_dir, mode="val", dataset=args.dataset)
    test_dataset  = MRCDataset(root_dir=args.data_dir, mode="test", dataset=args.dataset)
    clean_dataset = MRCDataset(root_dir=args.data_dir, mode="clean", dataset=args.dataset, abinitio=args.abinitio, mix=args.mix)
    
    train_dataloader = DataLoader(train_dataset,batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_mrc)
    val_dataloader = DataLoader(val_dataset,batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_mrc)
    test_dataloader = DataLoader(test_dataset,batch_size=1, shuffle=False, num_workers=0)
    clean_dataloader = DataLoader(clean_dataset,batch_size=1, shuffle=False, num_workers=0)

    logger.info("[INFO] Computing mu_y and sigma_y from clean dataset ...")
    mu_y, sigma_y = compute_clean_stats(clean_dataloader, device=device)
    cfg.model_config.clean_sigma = sigma_y
    cfg.model_config.clean_mu = mu_y

    model = ScoreFuncModel(cfg.model_config).to(device)
    
    logger.info("[INFO] Building CleanTargetProjector ...")
    projector = CleanTargetProjector(
        model=model,                    # UNet inside ScoreFuncModel]
        clean_dataloader=clean_dataloader,
        max_clean=cfg.model_config.get("max_clean", None),  # optional
        device=device,
        mix=args.mix,
    )
    
    if args.exp_name == "test":
        assert args.checkpoint is not None, "Must provide --checkpoint in test mode"
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[STAT] Model state loaded from {args.checkpoint}:", ckpt["model_state_dict"].keys())
        model.eval()
        print(f"[INFO] Loaded checkpoint from {args.checkpoint}")

        # Run sliding-window test
        save_dir = os.path.join(test_root, f"{dataset_tag}_test_outputs_TSM_10A_5A_3A_2")
        os.makedirs(save_dir, exist_ok=True)
        csv_path = f"{save_dir}/test_particle_stats.csv"
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["file_name","num_picked","total_windows","pick_ratio"])
        with torch.no_grad():
            for idx in range(len(test_dataset)):
                try:
                    batch = test_dataset[idx]
                except FileNotFoundError as e:
                    logger.warning(f"[SKIP] Missing file at test_dataset[{idx}]: {e}")
                    csv_file.flush()
                    continue
                except Exception as e:
                    logger.warning(f"[SKIP] Failed to load test_dataset[{idx}]: {e}")
                    csv_file.flush()
                    continue

                fname = batch["file_name"]

                out_path = os.path.join(save_dir, f"{fname}_denoised.mrc")
                if os.path.exists(out_path):
                    logger.info(f"[SKIP] Already exists: {out_path}")
                    continue

                micro = batch["noisy_image"]

                # make batch dimension
                if micro.ndim == 3:
                    micro = micro.unsqueeze(0)

                micro = micro.float().to(device)

                # HWC -> BCHW if needed
                if micro.ndim == 4 and micro.shape[-1] in (1, 3):
                    micro = micro.permute(0, 3, 1, 2).contiguous()

                try:
                    logger.info(f"[PROGRESS] Processing image {idx+1}/{len(test_dataset)}: {fname}")
                    denoised = sliding_window_inference(
                        micro,
                        model,
                        device,
                        window_size=args.img_size,
                        overlap=args.img_size // 2,
                        cfg=cfg.model_config,
                        projector=projector,
                        csv_writer=csv_writer,
                        file_name=fname
                    )

                    save_image(denoised.cpu(), out_path)
                    logger.info(f"[INFO] Saved {out_path}")
                    csv_file.flush()

                    # Clear GPU cache after each image
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    logger.error(f"[ERROR] RuntimeError processing {fname} at index {idx}: {e}")
                    csv_file.flush()
                    if "out of memory" in str(e).lower():
                        logger.error("[ERROR] GPU Out of Memory! Trying to clear cache...")
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    logger.error(f"[ERROR] Exception processing {fname} at index {idx}: {type(e).__name__}: {e}", exc_info=True)
                    csv_file.flush()
                    continue

        csv_file.close()
        logger.info(f"[COMPLETE] Test finished. Results saved to {save_dir}")
        return
    
    elif args.exp_name == "compare":
        assert args.checkpoint is not None, "Must provide --checkpoint in compare mode"
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"[INFO] Loaded checkpoint from {args.checkpoint}")

        # === Load TP / FP datasets ===
        tp_dataset = MRCDataset(root_dir=args.data_dir, mode="tp", dataset=args.dataset)
        fp_dataset = MRCDataset(root_dir=args.data_dir, mode="fp", dataset=args.dataset)

        tp_loader = DataLoader(tp_dataset, batch_size=8, shuffle=False)
        fp_loader = DataLoader(fp_dataset, batch_size=8, shuffle=False)

        save_dir = os.path.join(compare_root, f"feature_maps_{dataset_tag}_TSM_10A_5A_3A_2")
        os.makedirs(save_dir, exist_ok=True)

        def extract_features(loader, label):
            all_coeffs = []
            with torch.no_grad():
                for batch in loader:
                    if isinstance(batch, list):
                        batch = batch[0]  # or loop over list if multiple slices per file

                    imgs = batch["noisy_image"]
                    if imgs.ndim == 3:
                        imgs = imgs.unsqueeze(0)  # add batch dim
                    imgs = imgs.permute(0,3,1,2).float().to(device)
                    if "noiser_images" not in batch and "noisy_image" in batch:
                        batch["noiser_images"] = imgs
                    _, coeffs = model(batch, mode='test', return_coeffs=True)
                    all_coeffs.append(coeffs.cpu().numpy())
            coeffs = np.concatenate(all_coeffs, axis=0)
            np.save(os.path.join(save_dir, f"features_{label}.npy"), coeffs[:, :10])
            print(f"[INFO] Saved top-10 coefficients → {save_dir}/features_{label}.npy")

        # === Run on both TP and FP ===
        extract_features(tp_loader, "tp")
        extract_features(fp_loader, "fp")

        print(f"[INFO] Feature comparison extraction complete! Saved under {save_dir}")
        return

    max_epochs    = int(cfg.running_config.max_train_epoch)
    total_steps   = int(cfg.running_config.max_train_step)
    log_interval  = int(cfg.running_config.log_interval)
    save_interval = int(cfg.running_config.save_interval)

    num_parameters = count_parameters(model)
    logger.info("This model contains %.3fM parameters.", num_parameters)

    optim_name = cfg.optimizer_config.name
    optim_params = cfg.optimizer_config.params

    if hasattr(torch.optim, optim_name):
        OptimClass = getattr(torch.optim, optim_name)
        optimizer = OptimClass(model.parameters(), **optim_params)
        logger.info("Optimizer %s created with params: %s", optim_name, optim_params)
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")

    logger.info("Train Start...")
    train_loss_history = []
    train_dsm_history = []
    train_tsm_history = []
    val_loss_history = []
    val_dsm_history = []
    val_tsm_history = []
    similarity_history = []
    global_step = 0
    best_ssim = 0
    pbar = tqdm(total=total_steps, desc="train (steps)", dynamic_ncols=True)

    probe_batch = next(iter(train_dataloader))
    probe_noisy = probe_batch["noisy_image"].float().to(device)[0:1]

    model.train()
    clean_iter = iter(clean_dataloader)
    stop_training = False
    for epoch in range(1, max_epochs + 1):
        epoch_picked = 0
        epoch_total  = 0
        epoch_sim_sum = 0.0
        for batch in train_dataloader:

            raw_noisy = batch["noisy_image"].float().to(device)
            
            if global_step >= total_steps:
                break
        
            optimizer.zero_grad()
            
            try:
                clean_batch = next(clean_iter)
            except StopIteration:
                # restart clean loader when it runs out
                clean_iter = iter(clean_dataloader)
                clean_batch = next(clean_iter)
            
            train_data_dict_dsm = train_data_process_dsm(batch, cfg.model_config, device)
            train_data_dict_tsm = train_data_process_tsm(batch, cfg.model_config, device)

            if "clean_images" not in train_data_dict_tsm:
                clean_images = clean_batch["clean_images"].to(torch.float32).to(device)
                clean_images = clean_images.permute(0, 3, 1, 2).contiguous() / 127.5
                train_data_dict_tsm["clean_images"] = clean_images
            
            # probe similarity
            _, probe_sim = projector.similarity_all_clean(probe_noisy)
            similarity_history.append((global_step, probe_sim.item()))
            logger.info(f"[Probe] Step {global_step}: similarity for probe batch = {probe_sim.item():.4f}")

            # Get weighted clean target (single or dual-domain depending on --mix)
            tau = 0.5   # smaller = closer to hard NN
            y_tsm, best_sim = projector.get_weighted_target(
                raw_noisy,
                tau=tau,
                alpha_10A=0.5,
                alpha_3A=0.5,
            )
            train_data_dict_tsm["clean_images"] = y_tsm
            train_data_dict_tsm["similarity"] = best_sim.detach()
            logger.info(f"[Batch] Step {global_step}: similarity for current batch = {best_sim.item():.4f}")

            outputs_dsm = model(train_data_dict_dsm, mode='train')
            outputs_tsm = model(train_data_dict_tsm, mode='train')

            loss_dict = loss_func(
                train_data_dict_dsm,
                train_data_dict_tsm,
                outputs_dsm,
                outputs_tsm,
                cfg,
                global_step,
                return_components=True
            )

            loss = loss_dict["loss"]
            epoch_picked += loss_dict["num_picked"]
            epoch_total  += loss_dict["batch_size"]
            if loss_dict["mean_similarity"] is not None:
                epoch_sim_sum += loss_dict["mean_similarity"] * loss_dict["batch_size"]
            loss.backward()
                        
            optimizer.step()
            train_loss_history.append((global_step, loss.item()))
            train_dsm_history.append((global_step, loss_dict["dsm"].item()))
            train_tsm_history.append((global_step, loss_dict["tsm"].item()))

            global_step += 1
            if stop_training:
                logger.info(
                    f"Stopping training at epoch {epoch}, "
                    f"global_step={global_step}, total_steps={total_steps}"
                )
                break

            if (global_step % log_interval == 0) or (global_step == 1):
                pbar.set_postfix(
                    total=f"{loss.item():.4f}",
                    dsm=f"{loss_dict['dsm'].item():.4f}",
                    tsm=f"{loss_dict['tsm'].item():.4f}"
                )

            if global_step % save_interval == 0:
                logger.info(f"Evaluate the model at step {global_step}")

                visualization_path = os.path.join(
                    train_vis_dir,
                    f"{dataset_tag}_{category_tag}_10A_5A_3A_2",
                    f"step_{global_step}"
                )
                os.makedirs(visualization_path, exist_ok=True)
                val_loss_dict = compute_val_loss(val_dataloader, clean_dataloader, projector, model, cfg, global_step, device, logger)
                val_loss_history.append((global_step, val_loss_dict["total"]))
                val_dsm_history.append((global_step, val_loss_dict["dsm"]))
                val_tsm_history.append((global_step, val_loss_dict["tsm"]))
                logger.info(
                    f"[VAL] total={val_loss_dict['total']:.4f}, "
                    f"dsm={val_loss_dict['dsm']:.4f}, "
                    f"tsm={val_loss_dict['tsm']:.4f}"
                )
                metrics = metrics_eval(val_dataloader, model,cfg.model_config, device, visualization_path)
                logger.info("=== Test Metrics ===")
                logger.info(", ".join([f"{k}={metrics[k].mean():.4f}" for k in metrics]))
                
                if metrics['ssim'].mean()>best_ssim:
                    best_ssim = metrics['ssim'].mean()
                    checkpoint_path = os.path.join(train_ckpt_dir, f"{dataset_tag}_best_model_TSM_10A_5A_3A_2.pt")
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    torch.save({
                        "epoch": epoch,
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }, checkpoint_path)
                    logger.info(f"Checkpoint saved: {checkpoint_path}, best_step={global_step}, best_ssim={best_ssim:.4f}")
                
                projector.model = model.eval()   # ensure same weights
                projector.refresh_clean_features(clean_dataloader)
                logger.info(f"[Projector] Refreshed clean features at step {global_step}")

        # pick_ratio = epoch_picked / max(epoch_total, 1)
        # mean_similarity = epoch_sim_sum / max(epoch_total, 1)

        # logger.info(
            # f"[Train] Epoch {epoch} | "
            # f"Picked {epoch_picked}/{epoch_total} "
            # f"({pick_ratio:.3f}) | "
            # f"Mean sim {mean_similarity:.3f}"
        # )

    final_path = os.path.join(train_ckpt_dir, f"{dataset_tag}_TSM_10A_5A_3A_2.pt")
    plot_save_path = os.path.join(train_loss_dir, f"{dataset_tag}_loss_TSM_10A_5A_3A_2.png")
    sim_path = os.path.join(train_sim_dir, f"{dataset_tag}_sim_TSM_10A_5A_3A_2.png")
    plot_loss_curves(
        train_loss_history,
        val_loss_history,
        train_dsm_history,
        val_dsm_history,
        train_tsm_history,
        val_tsm_history,
        plot_save_path
    )
    # plot_similarity_curve(similarity_history,sim_path)
    # logger.info(f"Loss curves saved to {plot_save_path}")
    # logger.info(f"Similarity saved to {sim_path}")

    torch.save({
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, final_path)

    logger.info(f"Final model saved: {final_path}")

    
    logger.info("Training finished at step %d.", global_step)
    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tesing/ Training Script")

    parser.add_argument("--config_json", type=str, default="configs/gaussian_map.json",
                        help="Path to the JSON config file")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Path to the data directory")
    parser.add_argument("--exp_name", type=str, default="train",
                        help="Path to the experiment directory")
    parser.add_argument("--device", type=int, default=0,
                    help="-1 for CPU, >=0 for specific GPU index")
    parser.add_argument("--img_size", type=int, default=192,
                    help="Image size for Resize")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for test mode")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset suffix used in CSV and output file names")
    parser.add_argument("--abinitio", type=str2bool, nargs="?", const=True, default=False,
                        help="Use clean_{dataset}_abinitio.csv for the clean dataset")
    parser.add_argument("--mix", type=str2bool, nargs="?", const=True, default=False,
                        help="Mix 10A and 3A clean domains (folder names must contain '10A'/'3A')." \
                             " If False, use all clean images in a single softmax (default).")

    args = parser.parse_args()
    main(args)
