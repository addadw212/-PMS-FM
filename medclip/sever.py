import copy
import os

import torch

from medclip import constants, PromptClassifier
import numpy as np
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))  # subtract the max for numerical stability
    return e_x / e_x.sum(axis=0)  # the sum is computed along the only axis (axis=0)
class Server:
    def __init__(self,
                 global_model=None,
                 select_model=None,
                 current_round=0,
                 client_ids=None,
                 soft_lambda=0.7,
                 log_file=None,
                 ):
        self.global_model = global_model
        self.select_model = select_model
        self.current_round = current_round
        self.weights = {}
        self.client_ids = client_ids
        self.soft_lambda = soft_lambda
        self.person_models = {}
        for client_id in client_ids:
            self.person_models[client_id] = copy.deepcopy(global_model)
        self.client_weights = constants.CLIENTS_WEIGHT
        self.log_file = log_file

    def log_metric(self, task, acc):
        log_file = self.log_file
        folder_path = os.path.dirname(log_file)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(log_file, 'a') as f:
            f.write(f'Round: { self.current_round},{task} :ACC: {acc:.4f}\n')

    def receive(self, client_id, model, model_type):
        print(f"server receives {client_id}'s model file")
        client_num = len(self.client_ids)
        if model_type not in self.weights.keys():
            if model_type == "person_model":
                if "person_weights" not in self.weights.keys():
                    self.weights["person_weights"] = {}
                self.weights["person_weights"][client_id] = copy.deepcopy(model)
            else:
                self.weights[model_type] = copy.deepcopy(model)
                model_dict = self.weights[model_type]
                for key in model_dict:
                    if model_dict[key].dtype == torch.float32:
                        model_dict[key] = model_dict[key] / client_num
        else:
            if model_type == "person_model":
                self.weights["person_weights"][client_id] = copy.deepcopy(model)
            else:
                model_dict = model
                for key in model_dict:
                    if model_dict[key].dtype == torch.float32:
                        self.weights[model_type][key] += model_dict[key] / client_num

    def aggregate(self):
        print("client starts aggregation")
        self.current_round += 1
        weights = self.weights
        dicts = {"global_model": self.global_model.state_dict(),
                 "select_model": self.select_model.state_dict(),
                 }
        for model_name, model_weight in weights.items():
            if model_name == "person_weights":
                continue
            model_dict = dicts[model_name]
            for key in model_weight.keys():
                if model_weight[key].dtype == torch.float32:
                    model_dict[key] += model_weight[key]
        for client_id in self.client_ids:
            if "person_weights" not in weights.keys():
                break
            person_weight = weights["person_weights"][client_id].copy()
            for key in weights["person_weights"][client_id]:
                if person_weight[key].dtype != torch.float32:
                    continue
                person_weight[key] = person_weight[key] * 0
                for client in self.client_ids:
                    if client == client_id:
                        person_weight[key] += weights["person_weights"][client][key] * self.soft_lambda
                    else:
                        person_weight[key] += weights["person_weights"][client][key] * (1 - self.soft_lambda) / (len(self.client_ids) - 1)
            self.person_models[client_id].load_state_dict(person_weight)
        self.weights = {}

    # def validate(self, val_global):
    #     thd = constants.THRESHOLD
    #     select_model = self.select_model.to("cuda:0")
    #     client_ids = self.client_ids
    #     person_models = self.person_models
    #     for client_id in client_ids:
    #         person_models[client_id].to("cuda:0")
    #     global_model = self.global_model.to("cuda:0")
    #     pred_list = []
    #     label_list = []
    #     tasks = [
    #         "Atelectasis",
    #         "Cardiomegaly",
    #         "Consolidation",
    #         "Edema",
    #         "Pleural Effusion",
    #     ]
    #     for i, batch_data in enumerate(val_global):
    #         for task in tasks:
    #             pixel = batch_data["pixel_values"].to("cuda:0")
    #             input_ids = batch_data["prompt_inputs"][task]["input_ids"].to("cuda:0")
    #             attention_mask = batch_data["prompt_inputs"][task]["attention_mask"].to("cuda:0")
    #             outputs = select_model(pixel=pixel,
    #                                    input_ids=input_ids,
    #                                    attention_mask=attention_mask).cpu().detach().numpy()
    #         outputs = (2 * outputs + outputs2) / 2
    #         max_index = np.argmax(outputs)
    #         person_model = person_models[client_ids[max_index]]
    #         if np.max(outputs) <= thd:
    #             person_model = global_model
    #         medclip_clf = PromptClassifier(person_model)
    #         medclip_clf.eval()
    #         output = medclip_clf(**batch_data)
    #         pred = output['logits']
    #         pred_list.append(pred)
    #         label_list.append(batch_data['labels'])
    #     pred_list = torch.cat(pred_list, 0)
    #     labels = torch.cat(label_list).cpu().detach().numpy()
    #     pred = pred_list.cpu().detach().numpy()
    #     pred_label = pred.argmax(1)
    #     acc = (pred_label == labels).mean()
    #     print(acc)
    #     self.log_metric( "person_model", acc)
    #     pred_list = []
    #     label_list = []
    #     for i, batch_data in enumerate(val_global):
    #         medclip_clf = PromptClassifier(global_model)
    #         medclip_clf.eval()
    #         outputs = medclip_clf(**batch_data)
    #         pred = outputs['logits']
    #         pred_list.append(pred)
    #         label_list.append(batch_data['labels'])
    #     pred_list = torch.cat(pred_list, 0)
    #     labels = torch.cat(label_list).cpu().detach().numpy()
    #     pred = pred_list.cpu().detach().numpy()
    #     pred_label = pred.argmax(1)
    #     acc = (pred_label == labels).mean()
    #     self.log_metric("global_model", acc)
    #     print(acc)
    #     global_model.to("cpu")
    #     for client_id in client_ids:
    #         person_models[client_id].to("cpu")



def save_model(self):
        save_dir = f'outputs/models/{self.current_round}'
        os.makedirs(save_dir, exist_ok=True)
        global_path = os.path.join(save_dir, "global_model.pth")
        torch.save(self.global_model.state_dict(), global_path)
        # select_path = os.path.join(save_dir, "select_model.pth")
        # torch.save(self.select_model.state_dict(), select_path)
        for client_id in self.client_ids:
            model_path = os.path.join(save_dir, f"person_model_{client_id}.pth")
            torch.save(self.person_models[client_id].state_dict(), model_path)
