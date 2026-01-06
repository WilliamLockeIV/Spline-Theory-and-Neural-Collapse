import torch
import torch.nn as nn
import torch.linalg as LA
import torch.nn.functional as F
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def fit(self, train_dl, optimizer, scheduler, epochs=None, nc_steps=1, val_dl=None):
    '''
    Params:
        train_dl:       Train dataloader
        optimizer:      Optimizer
        scheduler:      Learning rate scheduler

        epochs (int):   Number of epochs to train. If None, will train for twice the
                        number of epochs necessary to reach 0 classification error.

        nc_steps (int): Calculate neural collapse statistics every n epochs. Default
                        of 1 calculates metrics after every epoch. Metrics will always
                        be calculated before first and after last epochs.

        val_dl:         Optional second dataloader to calculate neural collapse statistics.
                        This should normally be the same as train_dl (as it is when
                        val_dl=None), but if train_dl is over-sampled either for class
                        balance or to increase the number of samples per epoch, it is
                        recommended to use the non-over-sampled train_dl as val_dl to
                        calculate neural collapse statistics.

    Returns:
        Resets and updates self.log, a dictionary storing neural collapse statistics at every n epochs.
        Returns a Pandas DataFrame version of self.log, indexed by epoch.
    '''
    # Reset model log
    self.log = {'Epochs':[],'Class Error':[], 'CE Loss':[], 'In-Class':[], 'Out-Class':[], 'Covariance':[], 'Class Norms':[], 'Class Angles':[], 
                'Class Max Angles':[], 'Linear Norms':[], 'Linear Angles':[], 'Linear Max Angles':[], 'Duality':[], 'NCC':[]}
    if val_dl is None:
        val_dl = train_dl
    ce_loss = nn.CrossEntropyLoss(reduction='mean')
    start_epoch = 0
    end_epoch = epochs
    _ = self.measure_neural_collapse(val_dl, epoch=start_epoch, log=True)

    # If epochs is None, train for twice the number of epochs necessary to reach classification error
    # of 0. If epochs is not None, train for the specified number of epochs.
    if epochs is None:
        error = 0.5
        epoch = start_epoch
        while error > 0:
            epoch += 1
            error = 0
            loss_sum = 0
            for batch in train_dl:
                optimizer.zero_grad()
                image, class_labels = batch
                image, class_labels = image.to(device), class_labels.to(device)
                pred = self.forward(image)
                pred_class = torch.argmax(pred, dim=1)
                error += (pred_class!=class_labels).sum()
                loss_mean = ce_loss(pred, class_labels)
                loss_sum += loss_mean * len(batch)
                loss_mean.backward()
                optimizer.step()
            if epoch % nc_steps == 0:
                _ = self.measure_neural_collapse(val_dl, epoch=epoch, log=True)
            scheduler.step()
            print(f'{epoch:<10} {error:<10} {loss_sum:.4f}')
        start_epoch = epoch
        end_epoch = epoch * 2

    for epoch in range(start_epoch+1, end_epoch+1):
        error = 0
        loss_sum = 0
        for batch in train_dl:
            optimizer.zero_grad()
            image, class_labels = batch
            image, class_labels = image.to(device), class_labels.to(device)
            pred = self.forward(image)
            pred_class = torch.argmax(pred, dim=1)
            error += (pred_class!=class_labels).sum()
            loss_mean = ce_loss(pred, class_labels)
            loss_sum += loss_mean * len(batch)
            loss_mean.backward()
            optimizer.step()
        if epoch % nc_steps == 0:
            _ = self.measure_neural_collapse(val_dl, epoch=epoch, log=True)
        scheduler.step()
        print(f'{epoch:<10} {error:<10} {loss_sum:.4f}')

    _ = self.measure_neural_collapse(val_dl, epoch=epoch, log=True)
    log = pd.DataFrame(self.log).set_index('Epochs')
    return log

def measure_neural_collapse(self, dl, epoch=-1, log=False):
    all_features = []
    all_labels = []
    all_preds = []
    # Store last-layer feature activations
    def feature_activations(module, input):
        all_features.append(input[0].clone().detach())
        return None
    # Run model on all inputs while storing last-layer feature activations
    hook = self.fc.register_forward_pre_hook(feature_activations)
    with torch.no_grad():
        error = 0
        loss_sum = 0
        ce_loss_sum = nn.CrossEntropyLoss(reduction='sum')
        for batch in dl:
            image, class_labels = batch
            image, class_labels = image.to(device), class_labels.to(device)
            pred = self.forward(image)
            pred_class = torch.argmax(pred, dim=1)
            error += (pred_class!=class_labels).sum()
            loss_sum += ce_loss_sum(pred, class_labels)
            all_preds.append(pred_class)
            all_labels.append(class_labels)
        error = error.item()
        loss_sum = loss_sum.item()
    hook.remove()
    # Calculate global average and class average feature activations
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_features = torch.cat(all_features)
    global_mean = all_features.mean(dim=0)
    class_features = {idx.item():all_features[all_labels==idx] for idx in all_labels.unique(sorted=True)}
    class_means = {idx:features.mean(dim=0) for idx, features in class_features.items()}
    # Calculate centered class means
    centered_means = {idx:mean - global_mean for idx, mean in class_means.items()}
    # Extract linear layer weights
    linear_weights = self.fc.weight.clone().detach()

    # NC1
    '''
    Calculate covariance within classes and between classes. We can calculate this over
    raw vectors rather than centered vectors because covariance is invariate to shifts.
    We weight the classes in order to account for unbalanced datasets.
    '''
    class_weights = torch.tensor([features.shape[0] for features in class_features.values()]).to(device)
    class_weights = class_weights / class_weights.sum()
    within_class_cov = torch.stack([features.T.cov(correction=0) for features in class_features.values()])
    within_class_cov = (within_class_cov * class_weights[:,None,None]).sum(dim=0)
    between_class_cov = {idx:class_means[idx].repeat((features.shape[0],1)) for idx, features in class_features.items()}
    between_class_cov = torch.cat(list(between_class_cov.values())).T.cov(correction=0)
    normalized_cov = torch.matmul(within_class_cov, torch.linalg.pinv(between_class_cov))

    # NC2
    '''
    Calculate the norms and angles of the centered class means and linear layer weights.
    '''
    mean_norms = {idx:LA.vector_norm(mean) for idx, mean in centered_means.items()}
    mean_angles = torch.stack([F.normalize(mean, dim=0) for mean in centered_means.values()])
    mean_angles = torch.matmul(mean_angles, mean_angles.T)

    weight_norms = {idx:LA.vector_norm(linear_weights[idx]) for idx in range(linear_weights.shape[0])}
    weight_angles = F.normalize(linear_weights, dim=1)
    weight_angles = torch.matmul(weight_angles, weight_angles.T)    

    # NC3
    '''
    Calculate the difference between the (normalized) last-layer activations matrix and
    the linear classifier weight matrix.
    '''
    normed_classes = torch.stack([mean for mean in centered_means.values()])
    normed_classes = normed_classes / LA.matrix_norm(normed_classes, ord='fro')
    normed_weights = linear_weights / LA.matrix_norm(linear_weights, ord='fro')
    duality = normed_classes - normed_weights

    # NC4
    '''
    Calculate the proportion of predictions that agree with (or inversely disagree with)
    Nearest Class Center (NCC) prediction, i.e. selecting the class whose class mean
    is closest to the last-layer activations.
    '''
    all_means = torch.stack(list(class_means.values()))
    nearest_class = all_features[None,:,:] - all_means[:,None,:]
    nearest_class = torch.argmin(LA.vector_norm(nearest_class, dim=2),dim=0)
    ncc_error = (all_preds != nearest_class).cpu()

    # If log == True, update the model's self.log dictionary
    if log is True:
        in_class = within_class_cov.trace().item()
        out_class = between_class_cov.trace().item()
        covariance = normalized_cov.trace().item() / self.fc.out_features
        class_norms = torch.stack(list(mean_norms.values()))
        class_norms = (torch.std(class_norms) / torch.mean(class_norms)).item()
        rows, cols = torch.triu_indices(mean_angles.shape[0], mean_angles.shape[1], offset=1)
        class_angles = torch.std(mean_angles[rows, cols]).item()
        class_shift_angles = torch.mean(torch.abs(mean_angles[rows,cols]+1/(self.fc.out_features-1))).item()
        linear_norms = torch.stack(list(weight_norms.values()))
        linear_norms = (torch.std(linear_norms) / torch.mean(linear_norms)).item()
        rows, cols = torch.triu_indices(weight_angles.shape[0], weight_angles.shape[1], offset=1)
        linear_angles = torch.std(weight_angles[rows, cols]).item()
        linear_shift_angles = torch.mean(torch.abs(weight_angles[rows,cols]+1/(self.fc.out_features-1))).item()
        self_duality = LA.matrix_norm(duality, ord='fro').item()
        ncc = (ncc_error.sum()/len(ncc_error)).item()
        self.log['Epochs'].append(epoch)
        self.log['Class Error'].append(error)
        self.log['CE Loss'].append(loss_sum)
        self.log['In-Class'].append(in_class)
        self.log['Out-Class'].append(out_class)
        self.log['Covariance'].append(covariance)
        self.log['Class Norms'].append(class_norms)
        self.log['Class Angles'].append(class_angles)
        self.log['Class Shifted Angles'].append(class_shift_angles)
        self.log['Linear Norms'].append(linear_norms)
        self.log['Linear Angles'].append(linear_angles)
        self.log['Linear Shifted Angles'].append(linear_shift_angles)
        self.log['Duality'].append(self_duality)
        self.log['NCC'].append(ncc)

    return (within_class_cov, between_class_cov, normalized_cov, mean_norms, 
            mean_angles, weight_norms, weight_angles, duality, ncc_error)

def evaluate(self, dl):
    with torch.no_grad():
        ce_loss = nn.CrossEntropyLoss(reduction='mean')
        error = 0
        loss_sum = 0
        for batch in dl:
            image, class_labels = batch
            image, class_labels = image.to(device), class_labels.to(device)
            pred = self.forward(image)
            pred_class = torch.argmax(pred, dim=1)
            loss_mean = ce_loss(pred, class_labels)
            loss_sum += loss_mean * len(batch)
            error += (pred_class != class_labels).sum()
        return error.item(), loss_sum.item()