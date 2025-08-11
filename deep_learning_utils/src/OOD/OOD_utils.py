import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
import random
from sklearn.calibration import CalibrationDisplay
from torch.utils.data.dataloader import DataLoader
from torch import nn



def compute_scores(logits, score_function = 'max_softmax', T = 1):
    
    if score_function == 'max_softmax':
        score = F.softmax(logits/T, 1)
        score = score.max(dim=1)[0]
    
    else:
        score = score_function(logits,T) 

    return score


def compute_T_scaling(logits, y_gt,y_pred, T_vec,plots = False):

    best_ece = 1
    T_best = T_vec[0]
    best_scores, best_accuracies, e_best = [],[],[]
    
    for temp in T_vec:
        scores_test = compute_scores(logits, score_function = 'max_softmax', T=temp)
     
        b, e = np.histogram(scores_test.cpu(), bins=10)
      
        scores, accuracies = [],[]
        ece = 0
        
        for i in range(0,len(e)-1):
            inside_id = np.where((scores_test.cpu() >= e[i]) & (scores_test.cpu() < e[i+1]))[0]

            if len(inside_id):
                conf = scores_test[inside_id].mean().item()
            else:
                conf = 0.0
            scores.append(conf)


            if len(inside_id):
                a = torch.sum(y_pred[inside_id] == y_gt[inside_id])/len(inside_id)
                a = a.item()
            else:
                a = 0
            accuracies.append(a)
        
            ece += ((len(inside_id))/len(scores_test))*abs(a-scores[i])
          
        if ece < best_ece:
            best_ece = ece
            T_best = temp
            best_scores, best_accuracies = scores, accuracies
            e_best = e


    
    if plots:

        if max(best_accuracies)<max(best_scores):
            plt.bar(e[:-1], best_scores, width=np.diff(e_best), edgecolor="black", align="edge",label='scores', facecolor='red')
            plt.title(f'ECE:{100*best_ece:2.2f}%, T: {T_best}')
            plt.legend()
               #plt.show()
               
            plt.bar(e[:-1], best_accuracies, width=np.diff(e_best), edgecolor="black", align="edge",label='accuracies', facecolor='blue')
            plt.legend()
            plt.show()
        
        else:
            plt.bar(e[:-1], best_accuracies, width=np.diff(e_best), edgecolor="black", align="edge",label='accuracies',facecolor='blue')
            plt.legend()
   
            plt.bar(e[:-1], best_scores, width=np.diff(e_best), edgecolor="black", align="edge",label='scores',facecolor='red')
            plt.title(f'ECE:{100*best_ece:2.2f}%, T: {T_best}')
            plt.legend()
               #plt.show()
               
            plt.show()
            
        
    return best_ece,T_best



def OOD_pipeline(model, device, ID_ds, OOD_ds, batch_size, num_workers,seed=42, plot=True, T_scaling=None, plot_T_scaling=False,score_function='max_softmax', plot_performances=True):
    

    # For reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Loaders
    test_loader = DataLoader(ID_ds, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    fake_loader = DataLoader(OOD_ds, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    model.to(device)
    model.eval()

    N = len(test_loader.dataset)
    M = len(fake_loader.dataset)

    y_gt = torch.zeros(N, dtype=torch.long, device=device)
    y_pred = torch.zeros(N, dtype=torch.long, device=device)
    test_logits = torch.zeros((N, len(ID_ds.classes)), dtype=torch.float, device=device)
    fake_logits = torch.zeros((M, len(ID_ds.classes)), dtype=torch.float, device=device)

    start = 0
    with torch.no_grad():
        for data in test_loader:
            x, y = data
            x, y = x.to(device), y.to(device)
            logits = model(x)
            bsz = y.size(0)
            end = start + bsz

            y_gt[start:end] = y
            y_pred[start:end] = logits.argmax(1)
            test_logits[start:end, :] = logits

            start = end

    start = 0
    with torch.no_grad():
        for data in fake_loader:
            x, y = data
            x = x.to(device)
            logits = model(x)
            bsz = x.size(0)
            end = start + bsz
            fake_logits[start:end, :] = logits
            start = end

    if T_scaling is not None:
        best_ece, T_best = compute_T_scaling(test_logits, y_gt, y_pred, T_scaling, plot_T_scaling)

        if plot_T_scaling:
            fig, ax = plt.subplots(figsize=(6, 6))
            for temp in T_scaling:
                scores_temp = compute_scores(test_logits, score_function='max_softmax', T=temp)
                CalibrationDisplay.from_predictions(
                    (y_gt == y_pred).cpu(),
                    scores_temp.cpu(),
                    ax=ax,
                    label=f'T: {temp:.2f}',
                    n_bins=10
                )
            ax.set_title("Calibration curve for different temperatures")
            plt.show()

        scores_test = compute_scores(test_logits, score_function='max_softmax', T=T_best)
        scores_fake = compute_scores(fake_logits, score_function='max_softmax', T=T_best)
        scores_test_preT = compute_scores(test_logits, score_function='max_softmax')
        scores_fake_preT = compute_scores(fake_logits, score_function='max_softmax')

    else:
        scores_test = compute_scores(test_logits, score_function=score_function)
        scores_fake = compute_scores(fake_logits, score_function=score_function)

    if plot:
        if plot_performances:
            accuracy = (y_pred == y_gt).sum().item() / len(y_gt)
            print(f'Accuracy: {accuracy:.4f}')
            
            cm = metrics.confusion_matrix(y_gt.cpu(), y_pred.cpu())
            cmn = cm.astype(np.float32)
            cmn /= cmn.sum(1, keepdims=True)
            cmn = (100 * cmn).astype(np.int32)
            
            fig, ax = plt.subplots(figsize=(8, 8))
            disp = metrics.ConfusionMatrixDisplay(cmn, display_labels=ID_ds.classes)
            disp.plot(ax=ax, cmap='viridis')
            
            # Ruota le etichette sull'asse X
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            
            ax.set_title("Confusion Matrix")
            plt.tight_layout()
            plt.show()


        if plot_T_scaling:
            
            plt.hist(scores_test_preT.cpu(), density=True, alpha=0.5, bins=25, label='ID (before T)')
            plt.hist(scores_fake_preT.cpu(), density=True, alpha=0.5, bins=25, label='OOD (before T)')
            plt.title("Distribution before T-scaling")
            plt.legend()
            plt.show()

            plt.hist(scores_test.cpu(), density=True, alpha=0.5, bins=25, label='ID (after T)')
            plt.hist(scores_fake.cpu(), density=True, alpha=0.5, bins=25, label='OOD (after T)')
            plt.title("Distribution after T-scaling")
            plt.legend()
            plt.show()

        else:
            # No T-scaling: single plots
            plt.hist(scores_test.cpu(), density=True, alpha=0.5, bins=25, label='ID')
            plt.hist(scores_fake.cpu(), density=True, alpha=0.5, bins=25, label='OOD')
            plt.title("Distribution of ID/OOD scores")
            plt.legend()
            plt.show()

        y_labels_ID = (y_gt == y_pred).cpu()
        metrics.RocCurveDisplay.from_predictions(
            y_labels_ID, scores_test.cpu(), color='green')
        plt.title("ROC curve: ID-only (calibration)")
        plt.show()

        y_labels_OOD = torch.cat([
            torch.ones_like(scores_test),
            torch.zeros_like(scores_fake)
        ]).cpu()
        y_scores_OOD = torch.cat([scores_test, scores_fake]).cpu()
        metrics.PrecisionRecallDisplay.from_predictions(
            y_labels_OOD, y_scores_OOD, color='green')
        plt.title("Precision-Recall curve: OOD detection")
        plt.show()

    if T_scaling is not None:
        return T_best, best_ece


def FGSM(x, model, budget=0.1,y_true = None, y_target=None, loss_fun=nn.CrossEntropyLoss()):
    model.eval()
    x = x.clone().detach().requires_grad_(True)

    output = model(x)

    classes = list(range(output.size(1)))

    if y_target is not None:
        if y_target not in classes:
            raise ValueError(f"Target label out of range! Valid: {classes}")
        elif y_target == y_true:
            raise ValueError(f"Target label and true label are the same!")
        y = torch.tensor([y_target], device=x.device)
    else:
        y = y_true if torch.is_tensor(y_true) else torch.tensor([y_true], device=x.device)

    loss = loss_fun(output, y)

    model.zero_grad()
    loss.backward()

    perturbation = budget * torch.sign(x.grad)

    if y_target is not None:
        xadv = x - perturbation 
    else:
        xadv = x + perturbation  

    xadv = torch.clamp(xadv, -1, 1).detach()
    return xadv
