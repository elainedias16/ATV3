import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
from pathlib import Path
import pandas as pd



dataset_Path = "/srv12t/educampos/ATV3/dataset"

model = models.resnet18(pretrained=True)

layer = model._modules.get('avgpool')
model.eval()

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()



def gerar_embeddings(arquivo_imagem):

    img = Image.open(arquivo_imagem)
    img_transformada = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))

    embedding = torch.zeros(512) # armazenado espaço

    def capturar_embedding(m, i, o):
        embedding.copy_(o.data.reshape(o.data.size(1)))

    # capturando embeddings
    h = layer.register_forward_hook(capturar_embedding)
    model(img_transformada)

    h.remove()

    return np.array(embedding)



dataset_Path = Path(dataset_Path)

# final_df = []
# for dir in dataset_Path.iterdir():
#     all_imgs = []
#     imgs_ids = []
#     for sub_dir in dir.iterdir():
#         dir_name = sub_dir.stem
#         for sub2_dir in sub_dir.iterdir():
#             if sub2_dir.stem == "train":
#                 for img_dir in sub2_dir.iterdir():
#                     print(img_dir.stem)
#                     if img_dir.stem == "images":
#                         print(1)
#                         for image in img_dir.iterdir():
#                             imgs_ids.append(image.stem)
#                             all_imgs.append(gerar_embeddings(image))
#     print(dir_name)
#     names_df = pd.DataFrame(imgs_ids)       
#     imgs_df = pd.DataFrame(all_imgs)
#     mid_df = pd.concat([names_df,imgs_df],axis=1)
#     mid_df["dir"] = dir_name
#     print(mid_df.head())
#     mid_df.columns = ["img_name"] + [f"emb_dim{i}" for i in range(512)] + ["dir"]
#     final_df.append(mid_df)
    
# final_df = pd.concat(final_df,axis=0)
# final_df.to_csv("/srv12t/educampos/ATV3/dataset/final_df.csv")


test_df = []
for dir in dataset_Path.iterdir():
    all_imgs = []
    imgs_ids = []
    try:
        for sub_dir in dir.iterdir():
            dir_name = sub_dir.stem
            for sub2_dir in sub_dir.iterdir():
                if sub2_dir.stem == "test":
                    for img_dir in sub2_dir.iterdir():
                        print(img_dir.stem)
                        if img_dir.stem == "images":
                            print(1)
                            for image in img_dir.iterdir():
                                imgs_ids.append(image.stem)
                                all_imgs.append(gerar_embeddings(image))
        print(dir_name)
        names_df = pd.DataFrame(imgs_ids)       
        imgs_df = pd.DataFrame(all_imgs)
        mid_df = pd.concat([names_df,imgs_df],axis=1)
        mid_df["dir"] = dir_name
        print(mid_df.head())
        mid_df.columns = ["img_name"] + [f"emb_dim{i}" for i in range(512)] + ["dir"]
        test_df.append(mid_df)
    except:
        continue
    
    
test_df = pd.concat(test_df,axis=0)
test_df.to_csv("/srv12t/educampos/ATV3/dataset/test_df.csv")

