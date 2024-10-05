import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np
from model_building import build_LM_model, build_GNN_model
from dataloader import build_LM_dataloader, build_GNN_dataloader 
import os
import json
from pathlib import Path
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR


class LM_Trainer:
    def __init__(
            self, 
            model_name, 
            classifier_n_layers,
            classifier_hidden_dim,
            device, 
            train_epochs,
            optimizer_name,
            lr,
            weight_decay,
            dropout,
            att_dropout,
            lm_dropout,
            activation,
            warmup,
            max_length,
            batch_size,
            grad_accumulation,
            pl_ratio,
            eval_patience,
            LM_data_filepath,
            train_filepath,
            train_idx,
            valid_idx,
            test_idx,
            labels,
            user_seq,
            run
            ):
        
        self.model_name = model_name
        self.device = device
        self.train_epochs = train_epochs
        self.optimizer_name = optimizer_name.lower()
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.att_dropout = att_dropout
        self.lm_dropout = lm_dropout
        self.warmup = warmup
        self.max_length = max_length
        self.batch_size = batch_size
        self.grad_accumulation = grad_accumulation
        self.pl_ratio = pl_ratio
        self.eval_patience = eval_patience
        self.LM_data_filepath = LM_data_filepath    
        self.train_filepath = train_filepath
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        self.labels = labels
        self.user_seq = user_seq
        self.run = run
        self.do_mlm_task = False
        
        self.iter = 0
        self.best_iter = 0
        self.best_valid_acc = 0
        self.best_epoch = 0
        self.criterion = CrossEntropyLoss()
        self.results = {}

        
        self.train_steps_per_epoch = self.train_idx.shape[0] // self.batch_size + 1
        self.train_steps = int(self.train_steps_per_epoch * self.train_epochs)
        self.optimizer_args = dict(lr=lr, weight_decay=weight_decay)

        self.model_config = {
            'lm_model': model_name,
            'dropout': dropout,
            'att_dropout': att_dropout,
            'lm_dropout': self.lm_dropout,
            'classifier_n_layers': classifier_n_layers,
            'classifier_hidden_dim': classifier_hidden_dim,
            'activation': activation,
            'device': device,
            'return_mlm_loss': True if self.do_mlm_task else False
            }
        
        self.dataloader_config = {
            'batch_size': batch_size,
            }
        
    

        
    def build_model(self):
        self.model, self.tokenizer = build_LM_model(self.model_config)
        self.METADATA_id = self.tokenizer.convert_tokens_to_ids('METADATA:')
        self.POST_id = self.tokenizer.convert_tokens_to_ids('POST:')
        self.THREAD_id = self.tokenizer.convert_tokens_to_ids('THREAD:')
       
    def get_optimizer(self, parameters):
        
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(parameters, **self.optimizer_args)
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(parameters, **self.optimizer_args)
        elif self.optimizer_name == "adadelta":
            optimizer = torch.optim.Adadelta(parameters, **self.optimizer_args)
        elif self.optimizer_name == "radam":
            optimizer = torch.optim.RAdam(parameters, **self.optimizer_args)
        else:
            return NotImplementedError
        
        return optimizer
    
    def get_scheduler(self, optimizer):
        return get_cosine_schedule_with_warmup(optimizer, self.train_steps_per_epoch * self.warmup, self.train_steps) 
    
    def train(self):
        print('LM training start!')
        optimizer = self.get_optimizer(self.model.parameters())
        scheduler = self.get_scheduler(optimizer)
        if os.listdir(self.train_filepath) and os.path.exists(self.LM_data_filepath / 'embeddings.pt'):
            print('Train checkpoint exists, loading from checkpoint...')
            print('Please make sure you use the same parameter setting as the one of the train checkpoint!')
            ckpt = torch.load(self.train_filepath / os.listdir(self.train_filepath)[0])
            self.model.load_state_dict(ckpt['model'])
            test_acc, test_f1 = self.eval('test')
            self.results['train accuracy'] = test_acc
            self.results['train f1'] = test_f1
        
        else:
            step = 0
            valid_acc_best = 0
            valid_step_best = 0
            
            torch.save({'model': self.model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, self.train_filepath / 'best.pkl')
            
            train_loader = build_LM_dataloader(self.dataloader_config, self.train_idx, self.user_seq, self.labels, 'train')

            for epoch in range(int(self.train_epochs)+1):
                self.model.train()
                print(f'------LM Training Epoch: {epoch}/{int(self.train_epochs)}------')
                for batch in tqdm(train_loader):
                    step += 1
                    if step >= self.train_steps:
                        break
                    tokenized_tensors, labels, _ = self.batch_to_tensor(batch)

                    _, output = self.model(tokenized_tensors)
                    loss = self.criterion(output, labels)
                    loss /= self.grad_accumulation
                    loss.backward()
                    self.run.log({'LM Train Loss': loss.item()})
                    
                    if step % self.grad_accumulation == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    scheduler.step()

                    if step % self.eval_patience == 0:
                        valid_acc, valid_f1 = self.eval()

                        print(f'LM Train Valid Accuracy = {valid_acc}')
                        print(f'LM Train Valid F1 = {valid_f1}')
                        self.run.log({'LM Train Valid Accuracy': valid_acc})
                        self.run.log({'LM Train Valid F1': valid_f1})

                        if valid_acc > valid_acc_best:
                            valid_acc_best = valid_acc
                            valid_step_best = step
                            self.best_iter = self.iter
                            self.best_epoch = epoch
                            
                            torch.save({'model': self.model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, self.train_filepath / 'best.pkl')
                    
            
            print(f'The highest Train valid accuracy is {valid_acc_best}!')
            print(f'Load model from step {valid_step_best}')
            self.model.eval()
            all_outputs = []
            all_labels = []
            embeddings = []
            infer_loader = build_LM_dataloader(self.dataloader_config, None, self.user_seq, self.labels, mode='infer')
            with torch.no_grad():
                ckpt = torch.load(self.train_filepath / 'best.pkl')
                self.model.load_state_dict(ckpt['model'])
                optimizer.load_state_dict(ckpt['optimizer'])
                scheduler.load_state_dict(ckpt['scheduler'])
                for batch in tqdm(infer_loader):
                    tokenized_tensors, labels, _ = self.batch_to_tensor(batch)
                    embedding, output = self.model(tokenized_tensors)
                    embeddings.append(embedding.cpu())
                    all_outputs.append(output.cpu())
                    all_labels.append(labels.cpu())
                
                all_outputs = torch.cat(all_outputs, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                embeddings = torch.cat(embeddings, dim=0)
        

                test_predictions = torch.argmax(all_outputs[self.test_idx], dim=1).numpy()
                test_labels = torch.argmax(all_labels[self.test_idx], dim=1).numpy()
                torch.save(embeddings, self.LM_data_filepath / 'embeddings.pt')

                test_acc = accuracy_score(test_predictions, test_labels)
                test_f1 = f1_score(test_predictions, test_labels)
                self.results['LM train accuracy'] = test_acc
                self.results['LM train f1'] = test_f1
        

        print(f'LM Train Accuracy = {test_acc}')
        print(f'LM Train F1 = {test_f1}')
        self.run.log({'LM Train Accuracy': test_acc})
        self.run.log({'LM Train F1': test_f1})


        

    

    def eval(self, mode='valid'):
        if mode == 'valid':            
            eval_loader =  build_LM_dataloader(self.dataloader_config, self.valid_idx, self.user_seq, self.labels, mode='eval')
        elif mode == 'test':
            eval_loader =  build_LM_dataloader(self.dataloader_config, self.test_idx, self.user_seq, self.labels, mode='eval')
        self.model.eval()

        valid_predictions = []
        valid_labels = []

        with torch.no_grad():
            for batch in tqdm(eval_loader):
                tokenized_tensors, labels, _ = self.batch_to_tensor(batch)

                _, output = self.model(tokenized_tensors)

                valid_predictions.append(torch.argmax(output, dim=1).cpu().numpy())
                valid_labels.append(torch.argmax(labels, dim=1).cpu().numpy())

            valid_predictions = np.concatenate(valid_predictions)
            valid_labels = np.concatenate(valid_labels)
            valid_acc = accuracy_score(valid_labels, valid_predictions)
            valid_f1 = f1_score(valid_labels, valid_predictions)
            

            return valid_acc, valid_f1


    def test(self):
        print('Computing test accuracy and f1 for LM...')
        ckpt = torch.load(self.train_filepath / 'best.pkl')
        self.model.load_state_dict(ckpt['model'])
        test_acc, test_f1 = self.eval('test')
        print(f'LM Test Accuracy = {test_acc}')
        print(f'LM Test F1 = {test_f1}')
        self.run.log({'LM Test Accuracy': test_acc})
        self.run.log({'LM Test F1': test_f1})
        self.results['accuracy'] = test_acc
        self.results['f1'] = test_f1

    def batch_to_tensor(self, batch):
                    
        tokenized_tensors = self.tokenizer(text=batch[0], return_tensors='pt', max_length=self.max_length, truncation=True, padding='longest', add_special_tokens=False)
        for key in tokenized_tensors.keys():
            tokenized_tensors[key] = tokenized_tensors[key].to(self.device)
        labels = batch[1].to(self.device)
    
        if len(batch) == 3:
            is_pl = batch[2].to(self.device)
            return tokenized_tensors, labels, is_pl
        else:
            return tokenized_tensors, labels, None
        
    def load_embedding(self):
        embeddings = torch.load(self.LM_data_filepath / f'embeddings.pt')
        return embeddings
    
    def save_results(self, path):
        json.dump(self.results, open(path, 'w'), indent=4)
 


class GNN_Trainer:
    def __init__(
        self, 
        model_name, 
        device, 
        optimizer_name,
        lr,
        weight_decay,
        dropout,
        batch_size,
        gnn_n_layers,
        n_relations,
        activation,
        gnn_epochs_per_iter,
        pl_ratio,
        GNN_filepath,
        train_idx,
        valid_idx,
        test_idx,
        labels,
        edge_index, 
        edge_type,
        run,
        att_heads,
        gnn_hidden_dim,
        lm_name
        ):
    
        self.model_name = model_name
        self.device = device
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout        
        self.batch_size = batch_size
        self.gnn_n_layers = gnn_n_layers
        self.n_relations = n_relations
        self.activation = activation
        self.gnn_epochs_per_iter = gnn_epochs_per_iter
        self.pl_ratio = pl_ratio
        self.GNN_filepath = GNN_filepath
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        self.labels = labels
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.run = run
        self.att_heads = att_heads
        self.gnn_hidden_dim = gnn_hidden_dim
        self.lm_input_dim = 1024 if lm_name.lower() in ['roberta-large'] else 768
        self.iter = 0
        self.best_iter = 0
        self.best_valid_acc = 0
        self.best_valid_epoch = 0
        self.criterion = CrossEntropyLoss()

        
        self.results = {}
        self.get_train_idx_all()
        self.optimizer_args = dict(lr=lr, weight_decay=weight_decay)
        
        self.model_config = {
            'GNN_model': model_name,
            'optimizer': optimizer_name,
            'gnn_n_layers': gnn_n_layers,
            'n_relations': n_relations,
            'activation': activation,
            'dropout': dropout,
            'gnn_hidden_dim': gnn_hidden_dim,
            'lm_input_dim': self.lm_input_dim,
            'att_heads': att_heads,
            'device': device
            }
        
        self.dataloader_config = {
            'batch_size': batch_size,
            'n_layers': gnn_n_layers
            }
        
    

    def build_model(self):
        self.model = build_GNN_model(self.model_config)

    def get_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=self.gnn_epochs_per_iter, eta_min=0)
    

    def get_optimizer(self):
        
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), **self.optimizer_args)
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), **self.optimizer_args)
        elif self.optimizer_name == "adadelta":
            optimizer = torch.optim.Adadelta(self.model.parameters(), **self.optimizer_args)
        elif self.optimizer_name == "radam":
            optimizer = torch.optim.RAdam(self.model.parameters(), **self.optimizer_args)
        else:
            return NotImplementedError
        
        return optimizer
   
   
    def train(self, embeddings_LM):
        
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
        print('GNN training start!')
        train_loader = build_GNN_dataloader(self.dataloader_config, self.train_idx_all, embeddings_LM, self.labels , self.edge_index, self.edge_type, mode='train') #, is_pl=self.is_pl)
        print(self.gnn_epochs_per_iter)
        for epoch in tqdm(range(self.gnn_epochs_per_iter)):
            self.model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                batch_size = batch.batch_size
                x_batch = batch.x.to(self.device)

                edge_index_batch = batch.edge_index.to(self.device)
                edge_type_batch = batch.edge_type.to(self.device)
                labels = batch.labels[0: batch_size].to(self.device)

                output = self.model(x_batch, edge_index_batch, edge_type_batch) 
                output = output[0: batch_size]


                loss = self.criterion(output, labels)

                loss.backward()
                optimizer.step()
                scheduler.step()
                self.run.log({'GNN Train Loss': loss.item()})
        

            valid_acc, valid_f1 = self.eval(embeddings_LM)
     
            self.run.log({'GNN Valid Accuracy': valid_acc})
            self.run.log({'GNN Valid F1': valid_f1})

            if valid_acc > self.best_valid_acc:
                self.best_valid_acc = valid_acc
                self.best_epoch = epoch
                self.best_iter = self.iter
                torch.save({'model': self.model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, self.GNN_filepath / 'best.pkl')
        print(f'The highest valid accuracy is {self.best_valid_acc}!')
    
    def infer(self, embeddings_LM):
        self.model.eval()
        infer_loader = build_GNN_dataloader(self.dataloader_config, None, embeddings_LM, self.labels, self.edge_index, self.edge_type, mode='infer')

        all_outputs = []
        all_labels = []
        with torch.no_grad():
            ckpt = torch.load('/home/abdoul.amadou/gnn/test/LMBot/TwiBot-20_seed_1/GNN/best.pkl')
            self.model.load_state_dict(ckpt['model'])
            for batch in infer_loader:
                batch_size = batch.batch_size
                x_batch = batch.x.to(self.device)

                edge_index_batch = batch.edge_index.to(self.device)
                edge_type_batch = batch.edge_type.to(self.device)
                labels = batch.labels[0: batch_size].to(self.device)
                
                output = self.model(x_batch, edge_index_batch, edge_type_batch)
                output = output[0: batch_size]

                all_outputs.append(output.cpu())
                all_labels.append(labels.cpu())
            
            all_outputs = torch.cat(all_outputs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            test_predictions = torch.argmax(all_outputs, dim=0).numpy()
            test_labels = torch.argmax(all_labels[self.test_idx], dim=0).numpy()
            test_acc = accuracy_score(test_predictions, test_labels)
            test_f1 = f1_score(test_predictions, test_labels, average='macro')
            print(f'GNN Test Accuracy = {test_acc}')
            print(f'GNN Test F1 = {test_f1}')



    def eval(self, embeddings_LM,  mode='valid'):
        if mode == 'valid':            
            eval_loader =  build_GNN_dataloader(self.dataloader_config, self.valid_idx, embeddings_LM,  self.labels, self.edge_index, self.edge_type, mode='eval')
        elif mode == 'test':
            eval_loader =  build_GNN_dataloader(self.dataloader_config, self.test_idx, embeddings_LM, self.labels, self.edge_index, self.edge_type, mode='eval')
        self.model.eval()

        valid_predictions = []
        valid_labels = []

        with torch.no_grad():
            for batch in eval_loader:
                batch_size = batch.batch_size
                x_batch = batch.x.to(self.device)
                edge_index_batch = batch.edge_index.to(self.device)
                edge_type_batch = batch.edge_type.to(self.device)
                labels = batch.labels[0: batch_size].to(self.device)
                
                output = self.model(x_batch, edge_index_batch, edge_type_batch)
                output = output[0: batch_size]

                valid_predictions.append(torch.argmax(output, dim=1).cpu().numpy())
                valid_labels.append(torch.argmax(labels, dim=1).cpu().numpy())

            valid_predictions = np.concatenate(valid_predictions)
            valid_labels = np.concatenate(valid_labels)
            valid_acc = accuracy_score(valid_labels, valid_predictions)
            valid_f1 = f1_score(valid_labels, valid_predictions)
            conf_matrix = confusion_matrix(valid_labels, valid_predictions)
            return valid_acc, valid_f1

        

    def test(self, embeddings_LM):
        print('Computing test accuracy and f1 for GNN...')
        ckpt = torch.load(self.GNN_filepath / 'best.pkl')
        self.model.load_state_dict(ckpt['model'])
        test_acc, test_f1 = self.eval(embeddings_LM, 'test')
        print(f'GNN Test Accuracy = {test_acc}')
        print(f'GNN Test F1 = {test_f1}')
        self.run.log({'GNN Test Accuracy': test_acc})
        self.run.log({'GNN Test F1': test_f1})
        self.results['accuracy'] = test_acc
        self.results['f1'] = test_f1


    def save_results(self, path):
        json.dump(self.results, open(path, 'w'), indent=4)
        
    def get_train_idx_all(self):
        n_total = self.labels.shape[0]
        all = set(np.arange(n_total))
        exclude = set(self.train_idx.numpy())
        n = self.train_idx.shape[0]
        pl_ratio_GNN = min(self.pl_ratio, (n_total - n) / n)
        n_pl_GNN = int(n * pl_ratio_GNN)
        self.pl_idx = torch.LongTensor(np.random.choice(np.array(list(all - exclude)), n_pl_GNN, replace=False))
        print(self.pl_idx)
        self.train_idx_all = torch.cat((self.train_idx, self.pl_idx))
