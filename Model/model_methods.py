import torch
import torch.nn as nn
import torch.linalg as LA
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def fit(self, train_dl, optimizer, scheduler, epochs=None, neural_collapse=False, 
        local_complexity=False, lc_radius=0.015, lc_dim=None, lc_seed=None, 
        eval_steps=1, eval_dl=None):
    '''
    Params:
        train_dl:                   Train dataloader
        optimizer:                  Optimizer
        scheduler:                  Learning rate scheduler
        epochs (int):               Number of epochs to train. If None, will train for twice the
                                    number of epochs necessary to reach 0 classification error.

        neural_collapse (bool):     If True, calculate neural collapse metrics every n epochs.
        local_complexity (bool):    If True, calculate local complexity metrics every n epochs.
        lc_radius (float):          Radius of convex hulls for local complexity calculation.
        lc_dim (int):               Dimension of convex hulls for local complexity calculation.
                                    If None, will default to number of input dimensions.
        lc_seed (int):              Seed for local complexity reproducibility.

        eval_steps (int):           Update self.log with training loss, classification error, and 
                                    (optionally) neural collapse and local complexity metrics every 
                                    n epochs. Metrics will always be calculated before first and after 
                                    last epochs.

        eval_dl:                    Optional second dataloader to calculate neural collapse and local
                                    complexity. If train_dl is over-sampled either for class balance 
                                    or to increase the number of samples per epoch, it is recommended 
                                    to use the original, non-over-sampled dataset as eval_dl to
                                    calculate neural collapse and local complexity metrics.

    Returns:
        Returns a Pandas DataFrame of self.log, a dictionary storing training loss and classification
        error across epochs, optionally with neural collapse and local complexity metrics.
    '''
    # Reset model log and optional metric logs
    self.log = {'Epochs':[],'Class Error':[], 'CE Loss':[]} 
    nc_log = {'In-Class':[], 'Out-Class':[], 'Covariance':[], 'Class Norms':[], 'Class Angles':[], 'Class Shifted Angles':[], 
              'Linear Norms':[], 'Linear Angles':[], 'Linear Shifted Angles':[], 'Duality':[], 'NCC':[]}
    lc_log = {'Local Complexity':[]}

    if neural_collapse is True:
        self.log.update(nc_log)
    if local_complexity is True:
        self.log.update(lc_log)

    # Set eval_dl to train_dl if not otherwise specified
    if eval_dl is None:
        eval_dl = train_dl

    ce_loss = nn.CrossEntropyLoss(reduction='mean')
    ce_loss_sum = nn.CrossEntropyLoss(reduction='sum')
    start_epoch = 0
    end_epoch = epochs

    # Record metrics at initialization
    self.log['Epochs'].append(start_epoch)
    self.eval()
    with torch.no_grad():
        error = 0
        loss_sum = 0
        for batch in eval_dl:
            image, class_labels = batch
            image, class_labels = image.to(device), class_labels.to(device)
            pred = self.forward(image)
            pred_class = torch.argmax(pred, dim=1)
            error += (pred_class!=class_labels).sum()
            loss_sum += ce_loss_sum(pred, class_labels)
    self.log['Class Error'].append(error)
    self.log['CE Loss'].append(loss_sum)
    if neural_collapse is True:
        _ = self.get_neural_collapse(eval_dl, log=True)
    if local_complexity is True:
        _ = self.get_local_complexity(eval_dl, lc_radius, lc_dim, lc_seed, log=True)
    self.train()

    # If epochs is None, train for twice the number of epochs necessary to reach classification error
    # of 0. If epochs is not None, train for the specified number of epochs.
    if epochs is None:
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
            scheduler.step()
            print(f'{epoch:<10} {error:<10} {loss_sum:.4f}')

            # Every n epochs, record metrics
            if epoch % eval_steps == 0:
                self.log['Epochs'].append(epoch)
                self.eval()
                with torch.no_grad():
                    error = 0
                    loss_sum = 0
                    for batch in eval_dl:
                        image, class_labels = batch
                        image, class_labels = image.to(device), class_labels.to(device)
                        pred = self.forward(image)
                        pred_class = torch.argmax(pred, dim=1)
                        error += (pred_class!=class_labels).sum()
                        loss_sum += ce_loss_sum(pred, class_labels)
                self.log['Class Error'].append(error)
                self.log['CE Loss'].append(loss_sum)
                if neural_collapse is True:
                    _ = self.get_neural_collapse(eval_dl, log=True)
                if local_complexity is True:
                    _ = self.get_local_complexity(eval_dl, lc_radius, lc_dim, lc_seed, log=True)
                self.train()

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
        scheduler.step()
        print(f'{epoch:<10} {error:<10} {loss_sum:.4f}')

        # Every n epochs, record metrics
        if epoch % eval_steps == 0:
            self.log['Epochs'].append(epoch)
            self.eval()
            with torch.no_grad():
                error = 0
                loss_sum = 0
                for batch in eval_dl:
                    image, class_labels = batch
                    image, class_labels = image.to(device), class_labels.to(device)
                    pred = self.forward(image)
                    pred_class = torch.argmax(pred, dim=1)
                    error += (pred_class!=class_labels).sum()
                    loss_sum += ce_loss_sum(pred, class_labels)
            self.log['Class Error'].append(error)
            self.log['CE Loss'].append(loss_sum)
            if neural_collapse is True:
                _ = self.get_neural_collapse(eval_dl, log=True)
            if local_complexity is True:
                _ = self.get_local_complexity(eval_dl, lc_radius, lc_dim, lc_seed, log=True)
            self.train()

    if epoch != self.log['Epochs'][-1]:
        self.log['Epochs'].append(epoch)
        self.eval()
        with torch.no_grad():
            error = 0
            loss_sum = 0
            for batch in eval_dl:
                image, class_labels = batch
                image, class_labels = image.to(device), class_labels.to(device)
                pred = self.forward(image)
                pred_class = torch.argmax(pred, dim=1)
                error += (pred_class!=class_labels).sum()
                loss_sum += ce_loss_sum(pred, class_labels)
        self.log['Class Error'].append(error)
        self.log['CE Loss'].append(loss_sum)
        if neural_collapse is True:
            _ = self.get_neural_collapse(eval_dl, log=True)
        if local_complexity is True:
            _ = self.get_local_complexity(eval_dl, lc_radius, lc_dim, lc_seed, log=True)
        self.train()    
    log = pd.DataFrame(self.log).set_index('Epochs')
    return log


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


# Define functions to store model activations and partitions
def get_activations(self, name):
    '''
    params:
        name (string): Name of layer to store activations

    returns:
        None, writes activations to model dictionary "activations"
    '''
    def hook(module, input, output):
        self.activations[name] = output.detach().clone()
    return hook


def get_neural_collapse(self, dl, log=False):
    self.activations = dict()
    name, module = list(self.named_modules())[-2]
    hook = module.register_forward_hook(self.get_activations(name))

    # Run model on all inputs while storing last-layer feature activations
    with torch.no_grad():
        all_features = []
        all_labels = []
        all_preds = []
        for batch in dl:
            image, class_labels = batch
            image, class_labels = image.to(device), class_labels.to(device)
            pred = self.forward(image)
            pred_class = torch.argmax(pred, dim=1)
            all_features.append(self.activations[name])
            all_preds.append(pred_class)
            all_labels.append(class_labels)
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

    return (all_preds, all_labels, all_features)


def get_local_complexity(self, dl, radius=0.015, dim=25, seed=None, log=False):
    '''
    params:
        dl (DataLoader):  Dataloader over which to calculate local complexity
        radius (float):   Radius of convex neighborhood around inputs in which to calculate local
                          complexity. Smaller values tend to be deformed less by deep networks.
        dim (int):        Number of dimensions of convex neighborhood. If None, will default to
                          number of input dimensions.
        seed (int):       The orthogonal hulls around datapoints are oriented randomly. Set an integer
                          seed for reproducibility between function calls. If dim=1, orientation is
                          always the same regardless of seed or lack thereof.
        log (bool):       If True, update self.log with total local complexity.

    returns:
        local_complexity (dict): Nested dictionary of {layer: {Complexity:float, Eccentricity:float}},
                                along with a final key of {Total: {Complexity:float, Eccentricity:float}}
                                for the entire model.
    '''
    self.activations = dict()
    hooks = []
    samples = 0
    local_complexity = dict()
    for name, module in self.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(self.get_activations(name)))
            local_complexity[name] = {'Complexity':0, 'Eccentricity':0}

    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        for batch in dl:
            image, class_labels = batch
            image, class_labels = image.to(device), class_labels.to(device)
            samples += len(class_labels)
            image_dim = np.prod(image.shape[1:])
            if dim is None:
                dim = image_dim
            rand = torch.rand((image_dim, dim), device=device)
            ortho, _ = torch.linalg.qr(rand)
            ortho = ortho.reshape((dim, *image.shape[1:]))
            ortho = ortho * radius
            ortho_hulls = torch.cat([image[:,None], image[:,None] + ortho, image[:,None] - ortho], dim=1)
            input_hulls = ortho_hulls.reshape(ortho_hulls.shape[0] * ortho_hulls.shape[1], *ortho_hulls.shape[2:])
            _ = self.forward(input_hulls)

            for layer, activation in self.activations.items():
                activation = activation.reshape(ortho_hulls.shape[0], ortho_hulls.shape[1], -1)
                eccentricity = torch.cdist(activation, activation).amax(dim=(1,2)).sum()
                signs = torch.sign(activation)
                complexity = (signs[:,1:] != signs[:,:1]).any(dim=1).sum()
                local_complexity[layer]['Complexity'] += complexity.item()
                local_complexity[layer]['Eccentricity'] += eccentricity.item()

    for hook in hooks:
        hook.remove()

    for layer, dictionary in local_complexity.items():
        for key in dictionary.keys():
            local_complexity[layer][key] /= samples

    total_complexity = sum([local_complexity[layer]['Complexity'] for layer in local_complexity.keys()])
    max_eccentricity = max([local_complexity[layer]['Eccentricity'] for layer in local_complexity.keys()])
    local_complexity['Total'] = {'Complexity':total_complexity, 'Eccentricity':max_eccentricity}

    if log is True:
        self.log['Local Complexity'].append(local_complexity['Total']['Complexity'])
        self.log['Eccentricity'].append(local_complexity['Total']['Eccentricity'])

    return local_complexity


def get_partitions(self, x_span, y_span):
    '''
    params:
        x_span (tuple): Tuple of the form (x_min, x_max, x_samples) indicating the minimum and maximum x-values
                        over which to calculate partitions and the number of samples within that range.
        y_span (tuple): Tuple of the form (y_min, y_max, y_samples), same purpose as for x_span but for y-dimension.

    returns:
        partitions (dict): Dictionary of {layer:vertices} storing vertices that can be used to graph the partitions
                           drawn by each layer's neurons.
    '''
    partitions = dict()
    x_span = np.linspace(*x_span)
    y_span = np.linspace(*y_span)
    meshgrid = np.meshgrid(x_span, y_span)
    uniform_input = torch.tensor(np.stack([meshgrid[0].reshape(-1), meshgrid[1].reshape(-1)], 1), dtype=torch.float32)
    hooks = []
    for name, module in self.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(self.get_activations(name)))
    with torch.no_grad():
        self.forward(uniform_input)
    for hook in hooks:
        hook.remove()
    for layer, activation in self.activations.items():
        paths = [plt.contour(
            meshgrid[0], meshgrid[1], activation[:,i].reshape(meshgrid[0].shape), [0]
        ) for i in range(activation.shape[1])]
        paths = [path.get_paths()[0] for path in paths]
        plt.close()
        paths = [path.vertices[:-1] for path in paths]
        partitions[layer] = paths

    return partitions