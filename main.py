import sys
import json
import argparse
# print("DEBUG:  ->", sys.argv)


def preprocess_argv():
    new_argv = []

    for arg in sys.argv:
        if '=' in arg:
            key, value = arg.split('=', 1)
            new_argv.append(f'--{key}')
            new_argv.append(value)
        else:
            new_argv.append(arg)

    return new_argv


sys.argv = preprocess_argv()
parser = argparse.ArgumentParser(description="Your script description")
# 
# parser.add_argument('--task', type=str, default="vsr", required=True, help='predict mode')
# parser.add_argument('--mode', type=str, default="predict", required=True, help='predict mode')
parser.add_argument('task', type=str, default="vsr", help='predict mode')
parser.add_argument('mode', type=str, default="predict", help='predict mode')
# 
## 
parser.add_argument('--model', type=str, default="Lipv2_base.yaml", help='load model')
parser.add_argument('--data', type=str, default="cmlr.yaml", help='load dataset')
parser.add_argument('--root_dir', type=str, default="")  
parser.add_argument('--modality', type=str, default="video")  # audio, video, audio_video
parser.add_argument('--weights', type=str, default="") 
parser.add_argument('--aux_weights', type=str, default="")  
## train
parser.add_argument('--model_name', type=str, default="")
parser.add_argument('--run_exp_dir', type=str, default="run_exp")
parser.add_argument('--max_epochs', type=int, default=75)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--save_every_epoch', type=int, default=5)
parser.add_argument('--save_dir', type=str, default="run_exp")
parser.add_argument('--device', type=str, default=[0])  # (int | str | list, optional) device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
parser.add_argument('--pretrained', type=str, default="")  
parser.add_argument('--freeze', type=str, default="")  
parser.add_argument('--optimizer', type=str, default="Adam")
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--warmup_epochs', type=int, default=5)
parser.add_argument('--weight_decay', type=float, default=0.03)
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--workers', type=int, default=8)
## eval
parser.add_argument('--save_per_sample', type=bool, default=True)
parser.add_argument('--half', type=bool, default=False)
parser.add_argument('--ctc_weight', type=float, default=None)
parser.add_argument('--beam_size', type=int, default=None)
parser.add_argument('--lm_weight', type=float, default=None)
parser.add_argument('--test_data', type=str, default=None)
parser.add_argument('--lipauthEval_data', type=str, default=None)
parser.add_argument('--ids_user', type=str, help='JSON list of integers', default=None)
parser.add_argument('--save_featuresPath', type=str, default="/data/support_icslr/")
parser.add_argument('--attack_type', type=str, default="")
parser.add_argument('--attack_level', type=str, default="")
## predict
parser.add_argument('--source', type=str, default="/dataset/icslr/icslr_video_seg24s/main/PID0001/0001.mp4")
# - reWeights
parser.add_argument('--layer_depth', type=float, default=0)
parser.add_argument('--export_name', type=str, default='exportModel')
parser.add_argument('--export_layers', type=str, default='[10]')
parser.add_argument('--reWeights_name', type=str, default='frontend')
# - performance
parser.add_argument('--results', type=str, default='')


def main():
    args = parser.parse_args()
    task = args.task
    mode = args.mode

    if task in ["lipauth", "vsr", "asr", "avsr", "classifier"]:
        if mode in ["train", "resume"]:
            from train import run_train
            run_train(args)
        elif mode == "eval" or mode == "test":
            if task == "lipauth":
                from eval import eval_lipAuth
                eval_lipAuth(args)
            else:
                from eval import run_eval
                run_eval(args)
        elif mode == "register":
            if task == "lipauth":
                from ldw_register import run_register
                run_register(args)
            else:
                print("task should be lipauth")
        else:
            print("vsr task: print --weights --level")
    else:
        print("nothing!!")


# if __name__ == "__main__":
#     main()
if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True) 
    main()



