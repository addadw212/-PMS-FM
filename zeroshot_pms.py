import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn.functional as F
from medclip import constants, MedCLIPModel, MedCLIPVisionModelViT, PromptClassifier
from medclip.multi_fusion import MLPFusion_Mdoel, CAFusion_Mdoel
from medclip.multi_fusion import PromptLearner
from medclip.prompt_net import PromptTranslator
from medclip.prompts import generate_chexpert_class_prompts
from medclip.dataset import ImageTextContrastiveDataset, ImageTextContrastiveCollator, ZeroShotImageDataset, \
    ZeroShotImageCollator
import os
import random

import numpy as np
import torch
from PIL import Image
from torchvision import  transforms
from transformers import AutoTokenizer

from medclip import constants, MedCLIPModel, MedCLIPVisionModelViT, PromptClassifier, MedCLIPProcessor, utils
import pandas as pd

from medclip.prompts import process_class_prompts, generate_rsna_class_prompts, generate_covid_class_prompts

seed = 43
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_batch_size(client_id, method):
    batch_size = constants.batch_size[method][client_id]
    return batch_size
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))  # subtract the max for numerical stability
    return e_x / e_x.sum(axis=0)  # the sum is computed along the only axis (axis=0)

global_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
select_model = MLPFusion_Mdoel(num_classes=constants.SELECT_NUM)
select_dict = torch.load('./outputs/models/best/select_model_mlp.pth', map_location=torch.device('cuda:0'))
select_model.load_state_dict(select_dict, False)
select_model.to("cuda:0")
global_dict = torch.load('./outputs/models/best_model/global_model.pth', map_location=torch.device('cuda:0'))
global_model.load_state_dict(global_dict, False)
global_model.to("cuda:0")
thd = constants.THRESHOLD
client_ids = ["client_1", "client_2", "client_3", "client_4"]
person_models = {}
prompt_models = {}

for client_id in client_ids:
    person_dict = torch.load(f'./outputs/models/best_model/person_model_{client_id}.pth',
                             map_location=torch.device('cuda:0'))
    person_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    person_model.load_state_dict(person_dict, False)
    person_models[client_id] = person_model
prompt_dict = torch.load(f'./outputs/models/best_model/global_promptNet.pth',
                         map_location=torch.device('cuda:0'))
prompt_global_model = PromptTranslator(prompt_len=1, prompt_depth=1).to("cuda:0")
prompt_global_model.load_state_dict(prompt_dict)
for client_id in client_ids:
    prompt_dict = torch.load(f'./outputs/models/best_model/{client_id}_promptNet.pth',
                             map_location=torch.device('cuda:0'))
    prompt_model = PromptTranslator(prompt_len=1, prompt_depth=1).to("cuda:0")
    prompt_model.load_state_dict(prompt_dict)
    prompt_models[client_id] = prompt_model
for client_id in client_ids:
    person_models[client_id].to("cuda:0")
    prompt_models[client_id].to("cuda:0")

transform = transforms.Compose([
                transforms.Resize((constants.IMG_SIZE, constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838], std=[0.27950088968644304])]
            )



def pad_img(img, min_size=224, fill_color=0):
    '''pad img to square.
    '''
    x, y = img.size
    size = max(min_size, x, y)
    utils.modify_img(img)
    new_im = Image.new('L', (size, size), fill_color)
    new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
    return new_im
def get_length(method, type):
    len = constants.total_size[method][type]
    return len
RSNA_path = 'data/data_list/rsna.csv'
COVID_path = 'data/data_list/covid.csv'
RSNA_data = 'data/data_set/RSNA'
COVID_data = 'data/data_set/COVID'
rsna = pd.read_csv(RSNA_path)
covid = pd.read_csv(COVID_path)



cnt = 0
for i, row in rsna.iterrows():
    print(i)
    path = RSNA_data + '/' + row['imgpath'] + '.jpg'
    image = Image.open(path).convert('RGB')
    pixel = pad_img(image)
    pixel = transform(pixel).unsqueeze(1).to('cuda:0')
    if pixel.shape[1] == 1: pixel = pixel.repeat((1, 3, 1, 1))
    cls_prompts = generate_rsna_class_prompts(n=10)
    cls_prompts = process_class_prompts(cls_prompts)
    logit = []
    pred = 0
    for task in ['Pneumonia', 'Normal']:
        input_ids = cls_prompts[task]["input_ids"].to("cuda:0")
        attention_mask =cls_prompts[task]["attention_mask"].to("cuda:0")
        text_features = select_model.text_model(input_ids=input_ids, attention_mask=attention_mask)[0].unsqueeze(0)
        image_features = select_model.image_model(pixel)
        combined_features = torch.cat((text_features, image_features), dim=1)
        x = torch.relu(select_model.fc1(combined_features))
        x = select_model.fc2(x)
        outputs = F.softmax(x, dim=1).cpu().detach().numpy()
        max_index = np.argmax(outputs)
        person_model = person_models[client_ids[max_index]]
        if np.max(outputs) <= 0.4:
            person_model = global_model
        text_embeds = person_model.encode_text(input_ids, attention_mask)[0, :]
        img_embeds = person_model.encode_image(pixel)
        _logit = person_model.compute_logits(img_embeds, text_embeds).cpu().detach().item()
        logit.append(_logit)
    if logit[0] > logit[1]:
        pred = 1
    if pred == row['label']:
        cnt += 1
length = get_length('pms', 'rsna')
acc = cnt / length
print(f"{acc:.3f}")

cnt = 0
for i, row in covid.iterrows():
    print(i)
    path = COVID_data + '/' + row['imgpath']
    image = Image.open(path).convert('RGB')
    pixel = pad_img(image)
    pixel = transform(pixel).unsqueeze(1).to('cuda:0')
    if pixel.shape[1] == 1: pixel = pixel.repeat((1, 3, 1, 1))
    cls_prompts = generate_rsna_class_prompts(n=10)
    cls_prompts = process_class_prompts(cls_prompts)
    logit = []
    pred = 0
    for task in ['Pneumonia', 'Normal']:
        input_ids = cls_prompts[task]["input_ids"].to("cuda:0")
        attention_mask =cls_prompts[task]["attention_mask"].to("cuda:0")
        text_features = select_model.text_model(input_ids=input_ids, attention_mask=attention_mask)[0].unsqueeze(0)
        image_features = select_model.image_model(pixel)
        combined_features = torch.cat((text_features, image_features), dim=1)
        x = torch.relu(select_model.fc1(combined_features))
        x = select_model.fc2(x)
        outputs = F.softmax(x, dim=1).cpu().detach().numpy()
        max_index = np.argmax(outputs)
        person_model = person_models[client_ids[max_index]]
        if np.max(outputs) <= 0.4:
            person_model = global_model
        img_embeds = person_model.encode_image(pixel)
        text_embeds = person_model.encode_text(input_ids, attention_mask)[0,:]
        _logit = person_model.compute_logits(img_embeds, text_embeds).cpu().detach().item()
        logit.append(_logit)
    if row['label'] == 'postive':
        label = 1
    else:
        label = 0
    if logit[0] > logit[1]:
        pred = 1
    if pred == label:
        cnt += 1
    length = get_length('pms', 'covid')
acc = cnt / length
print(f"{acc:.3f}")

