import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


__all__ = (
    "LipAuthOutput",
)


class LipAuthOutput(nn.Module):
    def __init__(self, root_dir: str, support_path: str, id_list: List[str], save_layerFeatures: List[float], auth_type="eval"):
        super().__init__()

        self.root_dir = root_dir

        if root_dir not in support_path:
            self.support_path = root_dir + "/" +support_path
        else:
            self.support_path = support_path
        self.id_list = id_list  
        self.save_layerFeatures = save_layerFeatures

        if auth_type == "demo":
            self.registered_features = self.load_registered_VSA()

    def authentication(self, label, support, query, cer=None):
        model_name = self.support_path.split("/")[-2]

        sim_dict = {}
        support_features = []
        for feature in self.save_layerFeatures:
            rep = self.support_path.split("/")[-1].split("_")[0]
            fea_dir = self.support_path.replace(model_name+"/"+rep, model_name+"/"+feature[0])

            query_path = fea_dir + "/" + query.replace(".mp4", ".pt").replace("\n", "")
            query_feature = torch.load(query_path)
            if isinstance(query_feature, dict):
                query_feature = query_feature["inputs"]

            if isinstance(support, list):
                for support_ in support:
                    support_path_ = fea_dir + "/" + support_.replace(".mp4", ".pt").replace("\n", "")
                    support_feature_ = torch.load(support_path_)
                    if isinstance(support_feature_, dict):
                        support_feature_ = support_feature_["inputs"]
                    support_features.append(support_feature_)
                support_feature = torch.cat(support_features, dim=0)
                query_feature = query_feature.expand(len(support_features), -1, -1)
            else:
                support_path = fea_dir + "/" + support.replace(".mp4", ".pt").replace("\n", "")
                support_feature = torch.load(support_path)
                if isinstance(support_feature, dict):
                    support_feature = support_feature["inputs"]

            if feature[1] == "cos":
                support_feature = support_feature.view(support_feature.shape[0], -1)
                query_feature = query_feature.view(query_feature.shape[0], -1)
                if query_feature.shape[0]==1:
                    cos_sim = F.cosine_similarity(support_feature, query_feature, dim=1).item()
                else:
                    cos_sim = F.cosine_similarity(support_feature, query_feature, dim=1).mean().item()
                cos_sim = (cos_sim + 1) / 2

                if label == 'Pos':
                    if cos_sim >= float(feature[2]):
                        sim_dict[feature[0]] = [cos_sim, 'TP']
                    else:
                        sim_dict[feature[0]] = [cos_sim, 'FN']
                else:
                    if cos_sim >= float(feature[2]):
                        sim_dict[feature[0]] = [cos_sim, 'FP']
                    else:
                        sim_dict[feature[0]] = [cos_sim, 'TN']

            elif feature[1] in ["ctc", "ce", "att"]:  
                cos_sim = -query_feature.item()
                if cer <= 0.2:  # 'Pos':
                    if cos_sim < float(feature[2]):
                        sim_dict[feature[0]] = [cos_sim, 'TP', cer]  
                    else:
                        sim_dict[feature[0]] = [cos_sim, 'FN', cer]  
                else:
                    if cos_sim < float(feature[2]):
                        sim_dict[feature[0]] = [cos_sim, 'FP', cer] 
                    else:
                        sim_dict[feature[0]] = [cos_sim, 'TN', cer]
            elif feature[1] in ["cer"]:
                cos_sim, label = cer
                if label <= 0.2: 
                    if cos_sim < 0.2:
                        sim_dict[feature[0]] = [-cos_sim, 'TP', label]  
                    else:
                        sim_dict[feature[0]] = [-cos_sim, 'FN', label]  
                else:
                    if cos_sim < 0.2:
                        sim_dict[feature[0]] = [-cos_sim, 'FP', label]  
                    else:
                        sim_dict[feature[0]] = [-cos_sim, 'TN', label]
            else:
                raise ValueError("Unrecognized matching methods！！！")

        return sim_dict

