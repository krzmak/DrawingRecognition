import torch
import csv
from sklearn.metrics import confusion_matrix

class NetworkOperations:

    def __init__(self, model, check_point_name):

        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.check_point_name = check_point_name

    def train_network(self, number_of_epoch, train_loader, val_loader, optimizer, loss_fn, csv_save_path):

        max_acc = 0
        
        with open(csv_save_path , mode='w', newline='') as file:
            writer = csv.writer(file)
            
            writer.writerow(['Epoch', 'Train Total', 'Train Correct', 'Train Accuracy', 'Loss', 
                            'Validation Total', 'Validation Correct', 'Validation Accuracy'])

            for epoch in range(number_of_epoch):

                true_labels = []
                predicted_labels = []
                self.model.train()

                total_loss = 0.0
                total_acc = 0.0
                total = 0

                for batch in train_loader:

                    drawings, labels = batch
                    drawings = drawings.to(self.device)
                    labels = labels.type(torch.LongTensor)
                    labels = labels.to(self.device)
                    total += labels.size(0)

                    optimizer.zero_grad()

                    outputs = self.model(drawings)

                    loss = loss_fn(outputs, labels)
                    loss.backward()

                    optimizer.step()

                    total_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)

                    total_acc += (labels == predicted).sum().item()

                    true_labels.extend(labels.tolist())
                    predicted_labels.extend(predicted.tolist())

                epoch_loss = total_loss / len(train_loader)
                epoch_acc = 100 * total_acc / total

                val_total, val_total_acc, val_epoch_acc, val_confiusion_matrix = self.validate_model(val_loader)

                if val_epoch_acc > max_acc:
                    max_acc = val_epoch_acc
                    self.save_checkpoint(epoch, optimizer, max_acc)

                writer.writerow([epoch + 1, total, total_acc, epoch_acc, epoch_loss,
                                val_total, val_total_acc, val_epoch_acc])

                print(" __Train dataset__   Number of drawings in epoch: %d, correctly assigned classes: %d, (%.2f%%). Epoch loss: %.4f" % (total, total_acc, epoch_acc, epoch_loss))
                print("__Validation dataset__   Number of drawings in epoch: %d, correctly assigned classes: %d, (%.2f%%)." % (val_total, val_total_acc, val_epoch_acc))

            cm = confusion_matrix(true_labels, predicted_labels)

            print(cm)

            print(val_confiusion_matrix)


    @torch.no_grad()
    def validate_model(self, val_loader):

        true_labels = []
        predicted_labels = []

        total_acc = 0.0
        total = 0

        for batch in val_loader:

            drawings, labels = batch
            drawings = drawings.to(self.device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(self.device)
            total += labels.size(0)

            outputs = self.model(drawings)

            _, predicted = torch.max(outputs.data, 1)

            total_acc += (labels == predicted).sum().item()

            true_labels.extend(labels.tolist())
            predicted_labels.extend(predicted.tolist())

        epoch_acc = 100 * total_acc / total

        cm = confusion_matrix(true_labels, predicted_labels)

        return total, total_acc, epoch_acc, cm


    def save_checkpoint(self, epoch, optimizer, best_acc):
        state = {
            'epoch': epoch + 1,
            'model': self.model.state_dict(),
            'best_accuracy': best_acc,
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, self.check_point_name)
