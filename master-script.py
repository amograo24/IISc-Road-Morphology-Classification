# %%
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

from torch import nn
import torchvision
from torchvision.models import resnet50, efficientnet_b5, swin_v2_t, densenet161, resnet101, efficientnet_b7
from torchvision import transforms
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
# sensitivity, specificity, precision, recall, f1_score, roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix, classification_report, hamming_loss, zero_one_loss
from tqdm import tqdm
import wandb

import albumentations as A

# %%
DEVICE = "cuda" if torch.cuda.is_available() else "mps"
print(DEVICE)
torch.cuda.empty_cache()
RESUME = 0
NUM_EPOCHS = 30
ENCODER = "densenet161"
IMAGENET = True

BATCH_16 = 16
BATCH_8 = 8

DEST_DIR = f"./IISc-{ENCODER}-New-2-{'ImageNet' if IMAGENET else 'GRAY'}"
if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)

if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)

if not os.path.exists(f"{DEST_DIR}/models"):
    os.makedirs(f"{DEST_DIR}/models")

if not os.path.exists(f"{DEST_DIR}/logs"):
    os.makedirs(f"{DEST_DIR}/logs")

ROOT = "./data-iisc"
print(os.getcwd())
print(os.listdir(ROOT))


# %%
wandb.init(project=f"IISc-Urban-Morphology-Classification-{ENCODER}-New-2-{'ImageNet' if IMAGENET else 'GRAY'}",
           name = "Hope-2",
               config={
        "learning_rate": 0.0001,
        "architecture": ENCODER,
        "dataset": "IISc",
        "epochs": NUM_EPOCHS,
    },)

# %%
images_dir = "train_data_dir"
train_csv = "train_data.csv"
val_csv = "val_data.csv"
test_csv = "test_data.csv"

# %%
class Dataset(BaseDataset):

    def __init__(
            self,
            root,
            images_dir,
            csv,
            aug_fn=None,
            preprocessing=None,
            column_list = ["Image_ID", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6"]
    ):
        images_dir = os.path.join(root,images_dir)
        df = pd.read_csv(os.path.join(root,csv))

        self.ids = [
            (r[column_list[0]], r[column_list[1]], r[column_list[2]], r[column_list[3]], r[column_list[4]], r[column_list[5]], r[column_list[6]]) for i, r in df.iterrows()
        ]

        self.images = [os.path.join(images_dir, item[0]) for item in self.ids]
        self.labels = [item[1:] for item in self.ids]

        self.aug_fn = aug_fn
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        if not IMAGENET:
            image = cv2.imread(self.images[i],0)
            image = np.expand_dims(image, axis=-1)
        else:
            image = cv2.imread(self.images[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[i]

        if self.aug_fn:
            sample = self.aug_fn(image.shape)(image=image)
            image = sample['image']

        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']
            label = torch.tensor(label, dtype=torch.float32)

        return image, label
    
    def __len__(self):
        return len(self.images)

# %%
def resize_image(image_shp, target_size=512, train = False):
    """
    Resize the image to the target size
    :param image: The image to resize
    :param target_size: The target size
    :return: The resized image
    """
    h, w, _ = image_shp

    max_size = max(h, w)

    transform = A.Compose([
    A.PadIfNeeded(min_height=max_size, min_width=max_size, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)),
    A.Resize(512, 512, interpolation=cv2.INTER_AREA)] + [A.OneOf([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.8),
        A.Rotate(limit=15, p=0.7),
    ],p=0.8)] if train else [])

    return transform

# %%
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        # Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)

# %%
start_epoch = -1
max_score = 0
val_loss = 0
train_loss = 0
train_hamming_score = 0
val_hamming_score = 0
batch_size = 0
best_zero_one = 1
loss_name = "BCEWithLogitsLoss"

if ENCODER == "resnet50":
    if not IMAGENET:
        model = resnet50(weights=None, num_classes=6)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        print("in here")
    else:
        model = resnet50(weights='DEFAULT')
        model.fc = torch.nn.Linear(2048, 6)
elif ENCODER == "densenet161":
    if not IMAGENET:
        model = densenet161(weights=None, num_classes=6)
        model.features.conv0 = torch.nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    else:
        model = densenet161(weights='DEFAULT')
        model.classifier = torch.nn.Linear(2208, 6)
elif ENCODER == "resnet101":
    if not IMAGENET:
        model = resnet101(weights=None, num_classes=6)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    else:
        model = resnet101(weights='DEFAULT')
        model.fc = torch.nn.Linear(2048, 6)
elif ENCODER == "efficientnet_b5":
    if not IMAGENET:
        model = efficientnet_b5(weights=None, num_classes=6)
        model.features[0][0] = torch.nn.Conv2d(1, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    else:
        model = efficientnet_b5(weights='DEFAULT')
        model.classifier[1] = torch.nn.Linear(in_features=2048, out_features=6, bias=True)
elif ENCODER == "efficientnet_b7":
    if not IMAGENET:
        model = efficientnet_b7(weights=None, num_classes=6)
        model.features[0][0] = torch.nn.Conv2d(1, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    else:
        model = efficientnet_b7(weights='DEFAULT')
        model.classifier[1] = torch.nn.Linear(in_features=2048, out_features=6, bias=True)

model.to(DEVICE)
# print(model)
loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001),])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode="min",
    factor=0.4,
    patience=10,
    threshold=0.001,
    threshold_mode="abs",
)


# %%
if RESUME:
    checkpoint = torch.load(f"{DEST_DIR}/models/latest_model_{ENCODER}.pth",map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"]
    loss_name = checkpoint["loss_name"]
    max_score = checkpoint["max_score"]
    best_zero_one = checkpoint["best_zero_one"]

    train_loss = checkpoint["train_loss"]
    train_hamming_score = checkpoint["train_hamming_score"]

    val_loss = checkpoint["val_loss"]
    val_hamming_score = checkpoint["val_hamming_score"]

    batch_size = checkpoint["batch_size"]
    started_lr = checkpoint["started_lr"]


print("Scheduler State Dict Outside: ", scheduler.state_dict())
print("Epoch:", start_epoch)
print("Loss Function:", loss_name)
print("Max Val Hamming Score:", max_score)

print("Batch Size:", batch_size or BATCH_16)

print(f"Train Loss: {train_loss} || Val Loss: {val_loss}")
print(f"Train Hammming Score: {train_hamming_score} || Val Hamming Score: {val_hamming_score}")
print("Optimizer LR:", optimizer.param_groups[0]["lr"])

# %%
train_dataset = Dataset(
    root=ROOT,
    images_dir=images_dir,
    csv=train_csv,
    aug_fn=resize_image,
    # preprocessing=get_preprocessing()
    preprocessing=None
)

plt.imshow(train_dataset[0][0])
plt.show()

val_dataset = Dataset(
    root=ROOT,
    images_dir=images_dir,
    csv=val_csv,
    aug_fn=resize_image,
    preprocessing=get_preprocessing()
)

test_dataset = Dataset(
    root=ROOT,
    images_dir=images_dir,
    csv=test_csv,
    aug_fn=resize_image,
    preprocessing=get_preprocessing()
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,num_workers=2)


# %%
def epoch_runner(description:str, loader, model, loss, optimizer=None, device="cuda"):
    label_names = [ "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6"]
    epoch_loss = []
    original_labels = []
    predicted_labels = []
    # outputs_arr = []

    running_loss = 0.0
    count = 0

    # train_mode = (description.lower() == "train")

    run_modes = {"train":True,"val":False}
    mode = run_modes[description.lower()]

    # eps = 1e-10
    # print(description.title())
    # print(mode)

    if mode:
        model.train()
    else:
        model.eval()
    
    with torch.set_grad_enabled(mode):
        with tqdm(loader, desc=description.title()) as iterator:
            for images, labels in iterator:
                images = images.to(device)
                labels = labels.to(device)

                if mode:
                    optimizer.zero_grad()

                outputs = model.forward(images)
                # print(outputs.shape, labels.shape, labels.view(-1, 1).shape)   
                loss_value = loss(outputs, labels)

                if mode:
                    loss_value.backward()
                    optimizer.step()

                predicted = (torch.sigmoid(outputs) >= 0.5).int()

                running_loss += loss_value.item()
                count += 1
                epoch_loss.append(loss_value.item())
                original_labels.extend(labels.cpu().numpy().astype("int8"))
                predicted_labels.extend(predicted.cpu().numpy().astype("int8"))
                # outputs_arr.extend(outputs.cpu().detach().numpy().astype("float32"))
                # print(predicted)
                # print(predicted.cpu().numpy().astype("int8"))

                iterator.set_postfix({"loss":running_loss/count,"Accuracy":1-hamming_loss(original_labels, predicted_labels)})

        epoch_loss_value = np.mean(epoch_loss)
# labels=label_names
        epoch_classification_report = classification_report(original_labels, predicted_labels)

        print("Classification Report:\n", epoch_classification_report)
        epoch_cr_dictionary = classification_report(original_labels, predicted_labels, output_dict=True)

        epoch_mcm = multilabel_confusion_matrix(original_labels, predicted_labels)
        # epoch_auc = roc_auc_score(original_labels, outputs_arr,average=None)
        epoch_auc = [1.0,1.0,1.0,1.0,1.0,1.0]

        epoch_hamming_loss = hamming_loss(original_labels, predicted_labels)
        epoch_zero_one_loss = zero_one_loss(original_labels, predicted_labels)

    return epoch_loss_value, epoch_cr_dictionary, epoch_mcm, epoch_auc, epoch_hamming_loss, epoch_zero_one_loss

# %%
# model.to(DEVICE)
header = ["Epoch", "Loss", "Hamming Score", "AUC",
          "AUC_GI","AUC_IB","AUC_NP",
          "AUC_TB","AUC_RR","AUC_CDS",
          "Hamming Loss", "Zero One Loss",
          "Micro_Avg_P", "Micro_Avg_R", "Micro_Avg_F1", 
          "Macro_Avg_P", "Macro_Avg_R", "Macro_Avg_F1",
          "W_Avg_P", "W_Avg_R", "W_Avg_F1",
          "TN","FP","FN","TP"]

train_df = pd.DataFrame(columns=header)
val_df = pd.DataFrame(columns=header)
test_df = pd.DataFrame(columns=header)

# best_zero_one = 1

for epoch in range(start_epoch+1,start_epoch+NUM_EPOCHS+1):
    print("Epoch:", epoch)
    
    ####TRAIN####
    train_loss, train_crd, train_mcm, train_auc, train_hamming_loss, train_zero_one_loss = epoch_runner("train", 
                                                                                                        train_loader, model, 
                                                                                                        loss, optimizer, device=DEVICE)
    
    train_epoch_array = [epoch, train_loss, 1-train_hamming_loss, np.mean(train_auc)]
    train_epoch_array.extend(train_auc)
    train_epoch_array.extend([train_hamming_loss, train_zero_one_loss])
    train_epoch_array.extend([train_crd[a][b] for a in ['micro avg', 'macro avg', 'weighted avg'] for b in ['precision', 'recall', 'f1-score']])

    train_mcm_ravel = np.array([0,0,0,0])
    for matrix in train_mcm:
        train_mcm_ravel += matrix.ravel()
        
    train_epoch_array.extend(train_mcm_ravel)
    print("TN, FN, FP, TP: ",train_mcm_ravel)

    train_df.loc[len(train_df.index)] = train_epoch_array
    train_df.to_csv(f"{DEST_DIR}/logs/train_metrics_{start_epoch+1}_{ENCODER}.csv",index=False)
    ############

    ####VAL#####
    val_loss, val_crd, val_mcm, val_auc, val_hamming_loss, val_zero_one_loss = epoch_runner("val", val_loader, model, 
                                                                                                        loss, optimizer, device=DEVICE)
    
    val_epoch_array = [epoch, val_loss, 1-val_hamming_loss, np.mean(val_auc)]
    val_epoch_array.extend(val_auc)
    val_epoch_array.extend([val_hamming_loss, val_zero_one_loss])
    val_epoch_array.extend([val_crd[a][b] for a in ['micro avg', 'macro avg', 'weighted avg'] for b in ['precision', 'recall', 'f1-score']])

    val_mcm_ravel = np.array([0,0,0,0])
    for matrix in val_mcm:
        val_mcm_ravel += matrix.ravel()
        
    val_epoch_array.extend(val_mcm_ravel)
    print("TN, FN, FP, TP: ",val_mcm_ravel)

    val_df.loc[len(val_df.index)] = val_epoch_array
    val_df.to_csv(f"{DEST_DIR}/logs/val_metrics_{start_epoch+1}_{ENCODER}.csv",index=False)
    ############

    ####TEST####
    test_loss, test_crd, test_mcm, test_auc, test_hamming_loss, test_zero_one_loss = epoch_runner("val", test_loader, model, 
                                                                                                        loss, optimizer, device=DEVICE)
    
    test_epoch_array = [epoch, test_loss, 1-test_hamming_loss, np.mean(test_auc)]
    test_epoch_array.extend(test_auc)
    test_epoch_array.extend([test_hamming_loss, test_zero_one_loss])
    test_epoch_array.extend([test_crd[a][b] for a in ['micro avg', 'macro avg', 'weighted avg'] for b in ['precision', 'recall', 'f1-score']])

    test_mcm_ravel = np.array([0,0,0,0])
    for matrix in test_mcm:
        test_mcm_ravel += matrix.ravel()

    test_epoch_array.extend(test_mcm_ravel)
    print("TN, FN, FP, TP: ",test_mcm_ravel)

    test_df.loc[len(test_df.index)] = test_epoch_array
    test_df.to_csv(f"{DEST_DIR}/logs/test_metrics_{start_epoch+1}_{ENCODER}.csv",index=False)
    ############
    best=False
    if 1-val_hamming_loss >= max_score:
        max_score = 1-val_hamming_loss
        print("Highest Val Hamming Score!")
        best = True

    if val_zero_one_loss < best_zero_one:
        best_zero_one = val_zero_one_loss
        print("Lowest Zero One Loss!")
        best = True

    started_lr = optimizer.param_groups[0]["lr"]
    print("Started with LR:", started_lr)
    scheduler.step(val_loss)
    print("Changed LR to:", optimizer.param_groups[0]["lr"])
    

    checkpoint = {
        "epoch":epoch,

        "train_loss":train_loss,
        "train_hamming_loss":train_hamming_loss,
        "train_hamming_score":1-train_hamming_loss,
        "train_auc":np.mean(train_auc),
        "train_w_avg_p":train_crd["weighted avg"]['precision'],
        "train_w_avg_r":train_crd["weighted avg"]['recall'],
        "train_w_avg_f1":train_crd["weighted avg"]['f1-score'],
        "train_zero_one_loss":train_zero_one_loss,

        "val_loss": val_loss,
        "val_hamming_loss": val_hamming_loss,
        "val_hamming_score": 1 - val_hamming_loss,
        "val_auc":np.mean(val_auc),
        "val_w_avg_p": val_crd["weighted avg"]['precision'],
        "val_w_avg_r": val_crd["weighted avg"]['recall'],
        "val_w_avg_f1": val_crd["weighted avg"]['f1-score'],
        "val_zero_one_loss": val_zero_one_loss,

        "test_loss": test_loss,
        "test_hamming_loss": test_hamming_loss,
        "test_hamming_score": 1 - test_hamming_loss,
        "test_auc":np.mean(test_auc),
        "test_w_avg_p": test_crd["weighted avg"]['precision'],
        "test_w_avg_r": test_crd["weighted avg"]['recall'],
        "test_w_avg_f1": test_crd["weighted avg"]['f1-score'],
        "test_zero_one_loss": test_zero_one_loss,

        "loss_name":"BCEWithLogitsLoss",
        "batch_size":train_loader.batch_size,
        "max_score": max_score,
        "best_zero_one": best_zero_one,
        "started_lr": started_lr,
    }

    wandb.log(checkpoint)

    checkpoint['model_state_dict'] = model.state_dict()
    checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    checkpoint['scheduler_state_dict'] = scheduler.state_dict()


    torch.save(checkpoint, f"{DEST_DIR}/models/latest_model_{ENCODER}.pth")
    if best:
        torch.save(checkpoint, f"{DEST_DIR}/models/model_epoch_{epoch}_{ENCODER}.pth")

wandb.finish()


