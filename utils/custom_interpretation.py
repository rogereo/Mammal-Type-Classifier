import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import confusion_matrix

class CustomInterpretation:
    # minimal class for interpreting model predictions
    
    def __init__(self, inputs, labels, preds, outputs, losses, class_names):
        # store necessary info
        self.inputs = inputs
        self.labels = labels
        self.preds = preds
        self.outputs = outputs
        self.losses = losses
        self.class_names = class_names

    @classmethod
    def from_model(cls, model, dataloader, device, class_names):
        # run model once on dataloader and gather data
        model.eval()
        inputs, labels, preds, outputs, losses = [], [], [], [], []
        crit = torch.nn.CrossEntropyLoss(reduction='none')
        
        with torch.no_grad():
            for imgs, lbls in dataloader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outs = model(imgs)
                batch_losses = crit(outs, lbls)
                _, batch_preds = torch.max(outs, 1)

                for i in range(len(imgs)):
                    inputs.append(imgs[i].cpu().numpy())
                    labels.append(lbls[i].item())
                    preds.append(batch_preds[i].item())
                    outputs.append(outs[i].cpu())
                    losses.append(batch_losses[i].item())

        return cls(inputs, labels, preds, outputs, losses, class_names)

    def top_losses(self, k=None, largest=True):
        # return top k losses and indices
        losses_t = torch.tensor(self.losses)
        if k is None:
            k = len(losses_t)
        return torch.topk(losses_t, k, largest=largest)

    def plot_top_losses(self, k=6, largest=True, cols=3):
        # show samples with highest or lowest losses
        vals, idxs = self.top_losses(k, largest=largest)
        rows = (k + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        axes = axes.flatten() if rows > 1 else [axes]
        
        # add figure-level title
        fig.suptitle("Prediction/Actual/Loss/Probability", fontsize=12)

        for i, idx in enumerate(idxs):
            idx = idx.item()
            img = self.inputs[idx].transpose(1, 2, 0)
            loss_val = vals[i].item()
            pred_lbl = self.class_names[self.preds[idx]]
            act_lbl = self.class_names[self.labels[idx]]
            prob = F.softmax(self.outputs[idx].unsqueeze(0), dim=1)[0, self.preds[idx]].item()
            
            axes[i].imshow(img)
            axes[i].set_title(f"{pred_lbl}/{act_lbl}/{loss_val:.2f}/{prob:.2f}", fontsize=8)
            axes[i].axis("off")
        
        for j in range(i+1, len(axes)):
            axes[j].axis("off")
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)  # adjust so title doesn't overlap plots
        plt.show()


    def plot_random_predictions(self, n=6, cols=3):
        # randomly select samples to display
        indices = random.sample(range(len(self.inputs)), min(n, len(self.inputs)))
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        axes = axes.flatten() if rows > 1 else [axes]

        # add figure-level title
        fig.suptitle("Prediction/Actual/Loss/Probability", fontsize=12)

        for i, idx in enumerate(indices):
            img = self.inputs[idx].transpose(1, 2, 0)
            pred_lbl = self.class_names[self.preds[idx]]
            act_lbl = self.class_names[self.labels[idx]]
            loss_val = self.losses[idx]
            prob = F.softmax(self.outputs[idx].unsqueeze(0), dim=1)[0, self.preds[idx]].item()

            # color red if incorrect
            color = 'red' if pred_lbl != act_lbl else 'black'
            axes[i].imshow(img)
            axes[i].set_title(f"{pred_lbl}/{act_lbl}/{loss_val:.2f}/{prob:.2f}", fontsize=8, color=color)
            axes[i].axis("off")

        for j in range(i+1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        # adjust layout so the suptitle doesn't overlap the subplots
        plt.subplots_adjust(top=0.90)
        plt.show()

    def confusion_matrix(self):
        # compute confusion matrix
        return confusion_matrix(self.labels, self.preds)

    def plot_confusion_matrix(self):
        # show confusion matrix
        cm = self.confusion_matrix()
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel("predicted")
        plt.ylabel("actual")
        plt.title("confusion matrix")
        plt.show()
