import ast
import logging
import os.path
import time

from tqdm import tqdm
import torch

from utils.parseCFG import parse_cfg
from utils.ini_utils import ini_cfg

def pad(samples, pad_val=0.0):
    lengths = [len(s) for s in samples]
    max_size = max(lengths)
    sample_shape = list(samples[0].shape[1:])
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)
    for i, sample in enumerate(samples):
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )
    if len(samples[0].shape) == 1:
        collated_batch = collated_batch.unsqueeze(1)  # targets
    elif len(samples[0].shape) == 2:
        pass  # collated_batch: [B, T, 1]
    elif len(samples[0].shape) == 4:
        pass  # collated_batch: [B, T, C, H, W]
    return collated_batch, lengths


def collate_pad(batch):
    batch_out = {}
    for data_type in batch[0].keys():
        pad_val = -1 if data_type == "target" else 0.0
        c_batch, sample_lengths = pad(
            [s[data_type] for s in batch if s[data_type] is not None], pad_val
        )
        batch_out[data_type + "s"] = c_batch
        batch_out[data_type + "_lengths"] = torch.tensor(sample_lengths)
    return batch_out


def eval_lipAuth(args):
    cfg = parse_cfg(data_cfg=args.data, model_cfg=args.model)
    cfg = ini_cfg(args, cfg)
    outfile = cfg.eval.outfile
    with open(outfile, 'w', encoding="utf-8") as f:
        f.write("Let's start to LipAuth:\n")

    from nets.nn import LipAuthOutput
    LipAuth = LipAuthOutput(
        cfg.default.root_dir,
        cfg.model.Output.output_param.support,
        cfg.model.Output.output_param.ids_user,
        cfg.model.Output.output_param.save_layerFeatures
    )
    is_vsr = 0
    for fea in cfg.model.Output.output_param.save_layerFeatures:
        if fea[1] in ['ce', 'ctc', 'att']:
            is_vsr = 1
        elif fea[1] in ['cer']:
            is_vsr = 2
    if is_vsr > 0:
        vsr_path = cfg.default.root_dir + "/" + cfg.model.Output.output_param.vsr_res
        lines = open(vsr_path, "r", encoding="utf-8").readlines()
        cers = {}
        for line in tqdm(lines):
            if "actual: " in line:
                video_path = line.split('   ')[0].split('_video_seg24s')[-1]
                if "icslr" in line.split('   ')[0]:
                    PID = str(int(video_path.split('/')[-2].replace("PID", "")))
                elif "grid" in line.split('   ')[0]:
                    PID = str(int(video_path.split('/')[-3].replace("s", "")))
                video_number = video_path.split('/')[-1].replace(".mp4", "")
                cer_per = float(line.split('   ')[3].split(": ")[-1])
                cers[PID+"_"+video_number] = cer_per
    if is_vsr == 2:
        vsr_path2 = cfg.default.root_dir + "/" + cfg.model.Output.output_param.vsr_res2
        lines2 = open(vsr_path2, "r", encoding="utf-8").readlines()
        cers2 = {}
        for line in tqdm(lines2):
            if "actual: " in line:
                video_path = line.split('   ')[0].split('_video_seg24s')[-1]
                if "icslr" in line.split('   ')[0]:
                    PID = str(int(video_path.split('/')[-2].replace("PID", "")))
                elif "grid" in line.split('   ')[0]:
                    PID = str(int(video_path.split('/')[-3].replace("s", "")))
                video_number = video_path.split('/')[-1].replace(".mp4", "")
                cer_per = float(line.split('   ')[3].split(": ")[-1])
                cers2[PID+"_"+video_number] = cer_per

    # obtain pairs
    pairfile_path = cfg.default.root_dir + '/' + args.lipauthEval_data
    with open(pairfile_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    all_sim_scores = []
    for line in tqdm(lines):
        # 'Pos 19_0302.mp4 19_0546.mp4
        label, support, query = line.split(' ')
        if is_vsr==1:
            if '_light' in query:
                query_cer_index = query.split(".")[0].replace('_light', '')
            elif '_DF' in query:
                query_cer_index = query.split('_')[0]+'_'+query.split('_')[4]
            else:
                query_cer_index = query.split(".")[0]
            cer = cers[query_cer_index]
            sim_scores = LipAuth.authentication(label, support, query, cer)
        elif is_vsr==2:
            if '_light' in query:
                query_cer_index = query.split(".")[0].replace('_light', '')
            elif '_DF' in query:
                query_cer_index = query.split('_')[0]+'_'+query.split('_')[4]
            else:
                query_cer_index = query.split(".")[0]
            cer = [cers[query_cer_index], cers2[query_cer_index]]
            sim_scores = LipAuth.authentication(label, support, query, cer)
        else:
            sim_scores = LipAuth.authentication(label, support, query)
        all_sim_scores.append(sim_scores)
        with open(outfile, 'a', encoding="utf-8") as f:
            f.write(line+"  scores: "+str(sim_scores))
            f.write("\n")

    from utils.eval_matrix import plot_lipauth_threshold_hist, plot_lipauth_err, cal_lipauth_matrics
    savefig_path = outfile.replace(outfile.split('/')[-1], "")
    plot_lipauth_threshold_hist(all_sim_scores, save_dir=savefig_path)
    keys_err = plot_lipauth_err(all_sim_scores, save_dir=savefig_path)
    key_thresholds = {}
    for key, _, threshold in cfg.model.Output.output_param.save_layerFeatures:
        key_thresholds[key] = float(threshold)
    opt_thresholds, confusion_metrics, key_metrics, Total_confusion_metrics, Total_metrics = cal_lipauth_matrics(all_sim_scores, key_thresholds)
    with open(outfile, 'a', encoding="utf-8") as f:
        f.write("\n\n\nopt_thresholds: " + str(opt_thresholds))
        f.write("\nconfusion_metrics: " + str(confusion_metrics))
        f.write("\nkey_metrics: " + str(key_metrics))
        f.write("\n\nkey_thresholds: " + str(key_thresholds))
        f.write("\nTotal_confusion_metrics: " + str(Total_confusion_metrics))
        f.write("\nTotal_metrics: " + str(Total_metrics))
        f.write("\n")


