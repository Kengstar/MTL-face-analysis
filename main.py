import os
import torch
from torch import optim


from sacred import Experiment
from sacred.observers import FileStorageObserver
from model.Hypernet_CelebA import HyperNet_CelebA
from model.loss import mtl_loss_liebel

##sacred stuff
ex = Experiment(" MTL Modell")
ex.observers.append(FileStorageObserver.create('MTLModelExperiments'))


@ex.config
def config():
    batch_size= 128
    epochs = 15
    lr = 1e-3
    path = ""   ##output path for saving images for example
    notes = "Experiment descripton"
    cfg_dict= {} ## if there are to many hyper_params, collect here and dependency inject the dict

    attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
                  'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
                  'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                  'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
                  'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
                  'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

    input_size = (224, 224)
    batch_size = 300#128
    lr = 0.001
    tasks = ['landmarks', 'detection', 'segmentation']
    nr_of_epochs = 0
    init_variances = [1.0, 1.0, 1.0, 1.0]                 ### lm , attr, segm


###pytorch stuff and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HyperNet_CelebA(tasks=config()['tasks'], attributes=config()['attributes'],
                        init_variances=config()["init_variances"],
                        nr_of_landmarks=5).to(device)
params = list(model.parameters()) + model.uncertainties ### add uncertainty parameters to optimizer
optimizer = optim.Adam(params, lr=config()["lr"])
total_loss_fn = mtl_loss_liebel



@ex.capture
def train(epoch, train_loader):
    model.train()
    loss_total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        loss_total = 0.0
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        ##for task in tasks
            # loss_i = loss_fn_task_i(target_i,output_i)
            #loss_total += total_loss_fn(model.uncertainties[i], loss_i)

        loss_total.backward()
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, loss_total / len(train_loader.dataset)))
    ex.log_scalar('mean_train_loss', loss_total / len(train_loader.dataset))

@ex.capture
def test(data_loader, val=True):
    model.eval()
    loss_total = 0
    ###define all test metrics
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            ##for task in tasks
                # loss_i = loss_fn_task_i(target_i,output_i)
                # loss_total += total_loss_fn(model.uncertainties[i], loss_i)
                ###calc test metrics
    loss_total /= len(data_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(loss_total))
    if val:
        ex.log_scalar('mean_val_loss', loss_total)
    else:
        ex.log_scalar('mean_test_loss', loss_total)
    return loss_total




@ex.automain
def run_experiment(batch_size, path, epochs, train_split_ratio):
    print "Run Experiment"

    ex.info["model"] = repr(model)

    ## prepare datasets and loaders
    ## for individual use
    train_loader, val_loader, test_loader = None

    # training, validation, testing


    best_val_loss = 1000000.0
    for epoch in range(epochs):
        train(epoch, train_loader=train_loader)
        cur_val_loss = test(data_loader=val_loader)
        if cur_val_loss < best_val_loss:
            torch.save(model.state_dict(), os.path.join(ex.observers[0].dir, "weights.pt"))

    test(data_loader=test_loader, val=False)





