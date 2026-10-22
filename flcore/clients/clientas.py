import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client
from torch.autograd import grad




class clientAS(Client):


    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.fim_trace_history = []

    def train(self, is_selected):
        if is_selected:
            trainloader = self.load_train_data()
            self.model.train()

        
            start_time = time.time()

            max_local_epochs = self.local_epochs
            if self.train_slow:
                max_local_epochs = np.random.randint(1, max_local_epochs // 2)

            for step in range(max_local_epochs):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))
                    output = self.model(x)
                    loss = self.loss(output, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()


            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time


            self.model.eval()
            fim_trace_sum = 0
            for i, (x, y) in enumerate(self.load_train_data()):
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                nll = -torch.nn.functional.log_softmax(outputs, dim=1)[range(len(y)), y].mean()

                grads = grad(nll, self.model.parameters())

                for g in grads:
                    fim_trace_sum += torch.sum(g ** 2).detach()

            self.fim_trace_history.append(fim_trace_sum.item())


        else:
            trainloader = self.load_train_data()
            self.model.eval()
            fim_trace_sum = 0
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                nll = -torch.nn.functional.log_softmax(outputs, dim=1)[range(len(y)), y].mean()

                grads = grad(nll, self.model.parameters())

                for g in grads:
                    fim_trace_sum += torch.sum(g ** 2).detach()

            self.fim_trace_history.append(fim_trace_sum.item())


    def evaluate(self):
        testloader = self.load_test_data()
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in testloader:
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        accuracy = 100. * correct / total
        return accuracy
    

    def set_parameters(self, model, progress):

        local_prototypes = [[] for _ in range(self.num_classes)]
        batch_size = 16
        trainloader = self.load_train_data(batch_size=batch_size)

        for x_batch, y_batch in trainloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            with torch.no_grad():
                proto_batch = self.model.base(x_batch)

            for proto, y in zip(proto_batch, y_batch):
                local_prototypes[y.item()].append(proto)

        mean_prototypes = []

        for class_prototypes in local_prototypes:

            if not class_prototypes == []:
                stacked_protos = torch.stack(class_prototypes)

                mean_proto = torch.mean(stacked_protos, dim=0)
                mean_prototypes.append(mean_proto)
            else:
                mean_prototypes.append(None)

        alignment_optimizer = torch.optim.SGD(model.base.parameters(), lr=0.01)
        alignment_loss_fn = torch.nn.MSELoss()

        for _ in range(1):
            for x_batch, y_batch in trainloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                global_proto_batch = model.base(x_batch)
                loss = 0
                for label in y_batch.unique():
                    if mean_prototypes[label.item()] is not None:
                        loss += alignment_loss_fn(global_proto_batch[y_batch == label], mean_prototypes[label.item()])
                alignment_optimizer.zero_grad()
                loss.backward()
                alignment_optimizer.step()

        for new_param, old_param in zip(model.base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()



