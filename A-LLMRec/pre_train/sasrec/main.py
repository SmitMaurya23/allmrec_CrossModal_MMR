import os
import time
import torch
import argparse

from model import SASRec
from data_preprocess import *
from utils import *
from tqdm import tqdm

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, action='store_true')
parser.add_argument('--state_dict_path', default=None, type=str)

args = parser.parse_args()

if __name__ == '__main__':
    # Debugging device usage
    print(f"Using device: {args.device}")

    # Dataset processing
    preprocess(args.dataset)
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    print(f'user num: {usernum}, item num: {itemnum}')

    num_batch = len(user_train) // args.batch_size
    total_interactions = sum(len(user_train[u]) for u in user_train)
    print(f'average sequence length: {total_interactions / len(user_train):.2f}')
    
    # Dataloader
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

    # Model initialization
    model = SASRec(usernum, itemnum, args).to(args.device)
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception as e:
            print(f"Error initializing parameter {name}: {e}")
    
    model.train()

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            kwargs, checkpoint = torch.load(args.state_dict_path, map_location=torch.device(args.device))
            kwargs['args'].device = args.device
            model = SASRec(**kwargs).to(args.device)
            model.load_state_dict(checkpoint)
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
            print(f"Resuming from epoch {epoch_start_idx}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {args.state_dict_path}. Please check the path.")
        except Exception as e:
            print(f"Failed to load model checkpoint: {e}")

    # Inference only
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print(f'Test results (NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f})')

    # Training loop
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    T = 0.0
    t0 = time.time()
    
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):
        if args.inference_only: break
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)

            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            if step % 100 == 0:
                print(f"Loss in epoch {epoch} iteration {step}: {loss.item()}")

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating...', end='')
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print(f'\nEpoch: {epoch}, Time: {T}s, Validation (NDCG@10: {t_valid[0]:.4f}, HR@10: {t_valid[1]:.4f}), Test (NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f})')

            t0 = time.time()
            model.train()

        # Save the model at the end of training
        if epoch == args.num_epochs:
            folder = args.dataset
            fname = f'SASRec.epoch={epoch}.lr={args.lr}.layer={args.num_blocks}.head={args.num_heads}.hidden={args.hidden_units}.maxlen={args.maxlen}.pth'
            os.makedirs(folder, exist_ok=True)
            torch.save([model.kwargs, model.state_dict()], os.path.join(folder, fname))

    sampler.close()
    print("Done")
