import torch
from torch import nn
from transformers import DistilBertModel , DistilBertConfig , DistilBertTokenizer
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
import timm
import numpy as np
import itertools

""" df=pd.read_csv("captions.txt")
df['idx']=[id for id in range(df.shape[0]//5) for _ in range(5)]
df.to_csv("captions.csv", index=False) """
image_path='Images'
captions_path=''


class CFG:
    debug = False
    image_path = image_path
    captions_path = captions_path
    batch_size = 32
    num_workers = 6
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'resnet50'
    image_embedding = 2048
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1


class AverageMetrics():
    def __init__(self,name='Metric') -> None:
        self.name= name
        self.reset()

    def reset(self):
        self.avg,self.sum,self.count=[0]*3

    def update(self,val,count=1):
        self.count +=count
        self.sum +=val* count
        self.avg=self.sum/self.count

    def __repr__(self) :
        text=f"{self.name}: {self.avg:.4f}"
        return text
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    

class ClipDataset(torch.utils.data.Dataset):
    def __init__(self,image_filenames, captions,tokenizer,):

        self.image_filenames=image_filenames
        self.captions=captions
        self.encoded_captions=tokenizer(
            list(captions),padding=True,truncation=True,max_length=CFG.max_length
        )
        """ self.transforms=transforms
     """
    def __getitem__(self,idx):
        item={
            key:torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image=cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image=cv2.resize(image,(CFG.size,CFG.size))
        """ image=self.transforms(image=image)["image"] """
        item['image']=torch.tensor(image).permute(2,0,1).float()
        item['caption']=self.captions[idx]

        return item
    
    def __len__(self):
        return len(self.captions)
    
""" def get_transforms(mode="train"):
    if mode=="train":
        pass
     """

class ImageEncoder(nn.Module):
    def __init__(self, model_name=CFG.model_name,pretrained=CFG.pretrained,trainable=CFG.trainable):
        super().__init__()
        self.model= timm.create_model(model_name, pretrained,num_classes=0,global_pool="avg")
        for p in self.model.parameters():
            p.requires_grad=trainable
        
    def forward(self,x):
        return self.model(x)
    

class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained,trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model=DistilBertModel.from_pretrained(model_name)
        else:
            self.model=DistilBertModel(config=DistilBertConfig())

        for p in self.model.parameters():
            p.requires_grad=trainable

        self.target_token_idx=0

    def forward(self,input_ids,attention_mask):
        output=self.model(input_ids=input_ids,attention_mask=attention_mask)
        last_hidden_state=output.last_hidden_state
        return last_hidden_state[:,self.target_token_idx,:]
    

class Project(nn.Module):
    def __init__(self,embedding_dim,projection_dim=CFG.projection_dim,dropout=CFG.dropout):
               
        super().__init__()
        self.projection= nn.Linear(embedding_dim,projection_dim)
        self.gelu=nn.GELU()
        self.fc=nn.Linear(projection_dim,projection_dim)
        self.dropoutlayer=nn.Dropout(dropout)
        self.layer_norm=nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected= self.projection(x)
        x=self.gelu(projected)
        x=self.fc(x)
        x=self.dropoutlayer(x)
        x= x+projected
        x= self.layer_norm(x)

        return x
    

class ClipModel(nn.Module):
    def __init__(self,temprature=CFG.temperature,image_embedding=CFG.image_embedding,text_embedding=CFG.text_embedding):
        super().__init__()
        self.image_encoder=ImageEncoder()
        self.text_encoder=TextEncoder()

        self.image_projection=Project(embedding_dim=image_embedding)
        self.text_projection=Project(embedding_dim=text_embedding)

        self.temprature=temprature

    def forward(self,x):
        image_features=self.image_encoder(x['image'])
        text_features=self.text_encoder(x["input_ids"],x["attention_mask"])
        image_emb=self.image_projection(image_features)
        text_emb=self.text_projection(text_features)

        logits=(text_emb @ image_emb.T)/self.temprature
        image_similarity= image_emb @ image_emb.T
        text_similarity= text_emb @ text_emb.T
        targets= F.softmax((image_similarity+text_similarity)/2*self.temprature, dim=-1)

        text_loss=cross_entropy(logits, targets, reduction= "none")
        image_loss= cross_entropy(logits.T,targets.T,reduction="none")

        loss= (image_loss+text_loss)/2.0

        return loss.mean()


def cross_entropy(preds,targets,reduction="none"):
    log_softmax= nn.Softmax(dim=-1)
    loss= (-targets* log_softmax(preds)).sum(1)
    if reduction=="none":
        return loss
    elif reduction=="mean":
        return loss.mean()
    


def traindfs():
    dataframe=pd.read_csv("captions.csv")
    max_id=dataframe["idx"].max()+1 if not CFG.debug else 100
    image_ids= np.arange(0,max_id)
    valid_ids= np.random.choice(image_ids,size=int(0.2*len(image_ids)), replace=False)
    train_ids= [id for id in image_ids if id not in valid_ids]
    train_df=dataframe[dataframe["idx"].isin(train_ids)].reset_index(drop=True)
    valid_df=dataframe[dataframe["idx"].isin(valid_ids)].reset_index(drop=True)
    return train_df, valid_df

def build_loader(df,tokenizer,mode):
    dataset= ClipDataset(df["image"].values,df["caption"].values,tokenizer=tokenizer)

    dataloader=torch.utils.data.DataLoader(dataset,batch_size=CFG.batch_size,num_workers=CFG.num_workers,shuffle=True if mode=='train' else False)
    return dataloader

def train_epoch(model,train_loader,optimizer,lr_scheduler,step):
    loss_meter=AverageMetrics()
    for batch in tqdm(train_loader):
        batch={k:v.to(CFG.device) for k,v in batch.items() if k!="caption"}
        loss= model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step=="batch":
            lr_scheduler.step()

        count=batch["image"].size(0)
        loss_meter.update(loss.item(),count)

    return loss_meter

def valid_epoch(model,valid_loader):
    loss_meter=AverageMetrics()
    for batch in tqdm(valid_loader):
        batch={k:v.to(CFG.device) for k,v in batch.items() if k !="caption"}
        loss= model(batch)

        count= batch["image"].size(0)
        loss_meter.update(loss.item(),count)

    return loss_meter

def training():
    train_df,valid_df= traindfs()
    tokenizer=DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    train_loader=build_loader(train_df,tokenizer,"train")
    valid_loader=build_loader(valid_df,tokenizer,"valid")


    model= ClipModel().to(CFG.device)
    param=[
        {"params": model.image_encoder.parameters(),"lr":CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(),"lr":CFG.text_encoder_lr},
        {"params":itertools.chain(
        model.image_projection.parameters(),model.text_projection.parameters()
        ),"lr":CFG.head_lr,"weight_decay":CFG.weight_decay}        
    ]
    optimizer= torch.optim.Adam(param,weight_decay=0.)
    lr_scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode="min",patience=CFG.patience,factor=CFG.factor)

    step="epoch"

    best_loss=float("inf")
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch+1}")
        model.train()
        train_loss=train_epoch(model,train_loader,optimizer,lr_scheduler,step)
        model.eval()

        with torch.inference_mode():
            valid_loss= valid_epoch(model,valid_loader)
        if valid_loss.avg < best_loss:
            best_loss= valid_loss.avg
            torch.save(model.state_dict(),"best.pt")
            print("saved model")

        lr_scheduler.step(valid_loss.avg)

def get_img_emb(valid_df,model_path):
    tokenizer=DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    valid_loader= build_loader(valid_df,tokenizer,mode="valid")
    model= ClipModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path,map_location=CFG.device))
    model.eval()

    valid_img_emb= []
    with torch.inference_mode():
        for batch in tqdm(valid_loader):
            img_features=model.image_encoder(batch['image'].to(CFG.device))
            img_emb=model.image_projection(img_features)
            valid_img_emb.append(img_emb)
        
    return model,torch.cat(valid_img_emb)

def matches(model,img_emb,querry,img_filenames,n=9):
    tokenizer=DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_querry= tokenizer([querry])
    batch= {
        key:torch.tensor(values).to(CFG.device)
        for key, values in encoded_querry.items()
    }

    with torch.inference_mode():
        text_features = model.text_encoder(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"])
        text_emb=model.text_projection(text_features)

    image_emb_n=F.normalize(img_emb,dim=-1)
    text_emb_n=F.normalize(text_emb,dim=-1)
    dot_sim= text_emb_n @ image_emb_n.T

    values, indices = torch.topk(dot_sim.squeeze(0),n*5)
    matches= [img_filenames[idx] for idx in indices[::5]]

    _,axes= plt.subplots(3,3,figsize=(10,10))
    for match,ax in zip(matches,axes.flatten()):
        img= cv2.imread(f"{CFG.image_path}/{match}")
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis('off')

    plt.show()

if __name__ =='__main__':
    #training()
    _,valid_df= traindfs()
    model,img_emb= get_img_emb(valid_df,'best.pt')
    matches(model,img_emb,querry="fishing on a boat",img_filenames=valid_df['image'].values,n=9)




    