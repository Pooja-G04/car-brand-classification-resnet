import time
import torch
import os
from tempfile import TemporaryDirectory

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        # Initialize lists to store losses and accuracies
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []


        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']: 
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:  # dataloaders content print
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients every iteration (before processing every batch)
                    optimizer.zero_grad()  

                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):  # gradients computed in training phase
                        outputs = model(inputs) # output contains raw predictions for each batch (logits)
                                                # outputs shape is (batch_size, num_classes)
                        _, preds = torch.max(outputs, 1)  # outputs is the 1D array along the class dimension corresponding to the class with the highest probability
                        loss = criterion(outputs, labels)

                        # backprop + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train': # update learning rate every iteration
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Store losses and accuracies in respective lists
                if phase == 'train':
                    train_losses.append(epoch_loss)
                    train_accs.append(epoch_acc)
                else:
                    val_losses.append(epoch_loss)
                    val_accs.append(epoch_acc)

                # saving model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model, train_losses, train_accs, val_losses, val_accs
