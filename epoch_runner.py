import torch
import numpy as np
from sklearn.metrics import classification_report, multilabel_confusion_matrix, hamming_loss, zero_one_loss, roc_auc_score
from tqdm import tqdm


def epoch_runner(description:str, loader, model, loss, optimizer=None, device="cuda"):
    label_names = [ "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6"]
    epoch_loss = []
    original_labels = []
    predicted_labels = []
    outputs_arr = []

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

                iterator.set_postfix({"loss":running_loss/count,"Accuracy":1-hamming_loss(original_labels, predicted_labels)})

        epoch_loss_value = np.mean(epoch_loss)
# labels=label_names
        epoch_classification_report = classification_report(original_labels, predicted_labels)

        print("Classification Report:\n", epoch_classification_report)
        epoch_cr_dictionary = classification_report(original_labels, predicted_labels, output_dict=True)

        epoch_mcm = multilabel_confusion_matrix(original_labels, predicted_labels)
        epoch_auc = roc_auc_score(original_labels, outputs_arr,average=None)
        # epoch_auc = [1.0,1.0,1.0,1.0,1.0,1.0]

        epoch_hamming_loss = hamming_loss(original_labels, predicted_labels)
        epoch_zero_one_loss = zero_one_loss(original_labels, predicted_labels)

    return epoch_loss_value, epoch_cr_dictionary, epoch_mcm, epoch_auc, epoch_hamming_loss, epoch_zero_one_loss
