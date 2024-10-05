import argparse


def parser_args():
    parser = argparse.ArgumentParser()
    # wandb argument
    parser.add_argument('--project_name', type=str)
    parser.add_argument('--experiment_name', type=str)
    # dataset argument
    
    parser.add_argument('--batch_size_LM', type=int, default=16)
    parser.add_argument('--batch_size_GNN', type=int, default=300000)
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--is_processed', type=bool, default=True)
    parser.add_argument('--reset_split', type=str, default='6,2,2')

    # model argument
    
    parser.add_argument('--LM_model', type=str, default='bert')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--optimizer_LM', type=str, default='adamw')
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--LM_classifier_n_layers', type=int, default=2)
    parser.add_argument('--LM_classifier_hidden_dim', type=int, default=128)
    parser.add_argument('--LM_dropout', type=float, default=0.1)
    parser.add_argument('--LM_att_dropout', type=float, default=0.1)
    parser.add_argument('--warmup', type=float, default=0.6)

    parser.add_argument('--GNN_model', type=str, default='rgcn')
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_relations', type=int, default=3)
    parser.add_argument('--activation', type=str, default='leakyrelu')
    parser.add_argument('--optimizer_GNN', type=str, default='adamw')
    parser.add_argument('--GNN_dropout', type=float, default=0.4)
    parser.add_argument('--att_heads', type=int, default=8)



    # train evaluation test argument
    parser.add_argument('--seeds', type=str, default='1,2,3,4,5')
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--LM_train_epochs', type=float, default=3)
    parser.add_argument('--LM_eval_patience', type=int, default=20)
    parser.add_argument('--LM_accumulation', type=int, default=1)
    parser.add_argument('--GNN_epochs_per_iter', type=int, default=200)
    parser.add_argument('--pl_ratio_LM', type=float, default=0.5)
    parser.add_argument('--pl_ratio_GNN', type=float, default=0)


    parser.add_argument('--lr_LM', type=float, default=3e-5)
    parser.add_argument('--weight_decay_LM', type=float, default=0.1)

    parser.add_argument('--lr_GNN', type=float, default=5e-4)
    parser.add_argument('--weight_decay_GNN', type=float, default=1e-5)


    args = parser.parse_args()
    return args