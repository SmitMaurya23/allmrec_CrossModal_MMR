import random
import pickle

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import numpy as np

from models.recsys_model import *
from models.llm4rec import *
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    def __init__(self, rec_dim, text_dim, device):
        super().__init__()
        self.device = device
        self.temperature = nn.Parameter(torch.ones(1))
        self.rec_proj = nn.Linear(rec_dim, 256).to(device)
        self.text_proj = nn.Linear(text_dim, 256).to(device)
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        ).to(device)

    def forward(self, rec_emb, text_emb):
        rec_feat = self.rec_proj(rec_emb).to(self.device)  # [batch, 256]
        text_feat = self.text_proj(text_emb).to(self.device)  # [batch, 256]

        # Cross attention 
        attn = torch.matmul(rec_feat, text_feat.transpose(-2, -1)).to(self.device) / self.temperature.to(self.device)
        attn = F.softmax(attn, dim=-1)

        # Feature fusion
        rec_attended = torch.matmul(attn, text_feat)  
        fused = self.fusion(torch.cat([rec_feat, rec_attended], dim=-1)).to(self.device)  # [batch, 256]
    
        return fused  # Returns [batch, 256] tensor

class two_layer_mlp(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.fc1 = nn.Linear(dims, 128)
        self.fc2 = nn.Linear(128, dims)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x1 = self.fc2(x)
        return x, x1

class A_llmrec_model(nn.Module):
    def __init__(self, args):
        super(A_llmrec_model, self).__init__()
        self.device = args.device
        self.rec_model = ...  # Your existing recommender model initialization
        self.text_model = ...  # Your existing text model initialization
        self.attention = CrossModalAttention(args.rec_dim, args.text_dim, args.device)

    def forward(self, data, mode):
        rec_emb, text_emb = self.rec_model(data[0]), self.text_model(data[1])
        fused_emb = self.attention(rec_emb, text_emb)
        if mode == 'generate':
            # Implement your inference logic here
            pass
        else:
            # Implement your training logic here
            pass
        return fused_emb

    def save_model(self, args, epoch1=None, epoch2=None):
        # Implement model saving logic
        pass

    def load_model(self, args, phase1_epoch=None, phase2_epoch=None):
        # Implement model loading logic
        pass

class A_llmrec_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        rec_pre_trained_data = args.rec_pre_trained_data
        self.args = args
        self.device = args.device
        
        with open(f'./data/amazon/{args.rec_pre_trained_data}_text_name_dict.json.gz', 'rb') as ft:
            self.text_name_dict = pickle.load(ft)
        
        self.recsys = RecSys(args.recsys, rec_pre_trained_data, self.device)
        self.item_num = self.recsys.item_num
        self.rec_sys_dim = self.recsys.hidden_units
        self.sbert_dim = 768
        self.rec_proj = nn.Linear(self.rec_sys_dim, 256).to(self.device)
        
        self.mlp = two_layer_mlp(self.rec_sys_dim)
        if args.pretrain_stage1:
            self.sbert = SentenceTransformer('nq-distilbert-base-v1')
            self.mlp2 = two_layer_mlp(self.sbert_dim)
        
        self.mse = nn.MSELoss()
        
        self.maxlen = args.maxlen
        self.NDCG = 0
        self.HIT = 0
        self.rec_NDCG = 0
        self.rec_HIT = 0
        self.lan_NDCG = 0
        self.lan_HIT = 0
        self.num_user = 0
        self.yes = 0
        
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        
        if args.pretrain_stage2 or args.inference:
            self.llm = llm4rec(device=self.device, llm_model=args.llm)
            
            self.log_emb_proj = nn.Sequential(
                nn.Linear(self.rec_sys_dim, self.llm.llm_model.config.hidden_size),
                nn.LayerNorm(self.llm.llm_model.config.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.llm.llm_model.config.hidden_size, self.llm.llm_model.config.hidden_size)
            )
            nn.init.xavier_normal_(self.log_emb_proj[0].weight)
            nn.init.xavier_normal_(self.log_emb_proj[3].weight)

            self.item_emb_proj = nn.Sequential(
                nn.Linear(128, self.llm.llm_model.config.hidden_size),
                nn.LayerNorm(self.llm.llm_model.config.hidden_size),
                nn.GELU(),
                nn.Linear(self.llm.llm_model.config.hidden_size, self.llm.llm_model.config.hidden_size)
            )
            nn.init.xavier_normal_(self.item_emb_proj[0].weight)
            nn.init.xavier_normal_(self.item_emb_proj[3].weight)
            
    def save_model(self, args, epoch1=None, epoch2=None):
        out_dir = f'./models/saved_models/'
        create_dir(out_dir)
        out_dir += f'{args.rec_pre_trained_data}{args.recsys}{epoch1}_'
        if args.pretrain_stage1:
            torch.save(self.sbert.state_dict(), out_dir + 'sbert.pt')
            torch.save(self.mlp.state_dict(), out_dir + 'mlp.pt')
            torch.save(self.mlp2.state_dict(), out_dir + 'mlp2.pt') 
        
        out_dir += f'{args.llm}{epoch2}'
        if args.pretrain_stage2:
            torch.save(self.log_emb_proj.state_dict(), out_dir + 'log_proj.pt')
            torch.save(self.item_emb_proj.state_dict(), out_dir + 'item_proj.pt')
            
    def load_model(self, args, phase1_epoch=None, phase2_epoch=None):
        out_dir = f'./models/saved_models/{args.rec_pre_trained_data}{args.recsys}{phase1_epoch}_'
        
        mlp = torch.load(out_dir + 'mlp.pt', map_location=args.device)
        self.mlp.load_state_dict(mlp)
        del mlp
        for name, param in self.mlp.named_parameters():
            param.requires_grad = False

        if args.inference:
            out_dir += f'{args.llm}{phase2_epoch}'
            
            log_emb_proj_dict = torch.load(out_dir + 'log_proj.pt', map_location=args.device)
            self.log_emb_proj.load_state_dict(log_emb_proj_dict)
            del log_emb_proj_dict
            
            item_emb_proj_dict = torch.load(out_dir + 'item_proj.pt', map_location=args.device)
            self.item_emb_proj.load_state_dict(item_emb_proj_dict)
            del item_emb_proj_dict

    def find_item_text(self, item, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'
        if title_flag and description_flag:
            return [f'"{self.text_name_dict[t].get(i, t_)}, {self.text_name_dict[d].get(i, d_)}"' for i in item]
        elif title_flag and not description_flag:
            return [f'"{self.text_name_dict[t].get(i, t_)}"' for i in item]
        elif not title_flag and description_flag:
            return [f'"{self.text_name_dict[d].get(i, d_)}"' for i in item]
    
    def find_item_text_single(self, item, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'
        if title_flag and description_flag:
            return f'"{self.text_name_dict[t].get(item, t_)}, {self.text_name_dict[d].get(item, d_)}"'
        elif title_flag and not description_flag:
            return f'"{self.text_name_dict[t].get(item, t_)}"'
        elif not title_flag and description_flag:
            return f'"{self.text_name_dict[d].get(item, d_)}"'
        
    def get_item_emb(self, item_ids):
        with torch.no_grad():
            item_embs = self.recsys.model.item_emb(torch.LongTensor(item_ids).to(self.device))
            item_embs, _ = self.mlp(item_embs)
        
        return item_embs
    
    def forward(self, data, optimizer=None, batch_iter=None, mode='phase1'):
        if mode == 'phase1':
            self.pre_train_phase1(data, optimizer, batch_iter)
        if mode == 'phase2':
            self.pre_train_phase2(data, optimizer, batch_iter)
        if mode == 'generate':
            self.generate(data)

    def pre_train_phase1(self, data, optimizer, batch_iter):
        epoch, total_epoch, step, total_step = batch_iter
            
        self.sbert.train()
        optimizer.zero_grad()

        u, seq, pos, neg = data
        indices = [self.maxlen * (i + 1) - 1 for i in range(u.shape[0])]
        
        with torch.no_grad():
            log_emb, pos_emb, neg_emb = self.recsys.model(u, seq, pos, neg, mode='item')
            
        log_emb_ = log_emb[indices]
        pos_emb_ = pos_emb[indices]
        neg_emb_ = neg_emb[indices]
        pos_ = pos.reshape(pos.size)[indices]
        neg_ = neg.reshape(neg.size)[indices]
        
        start_inx = 0
        end_inx = 60
        iterss = 0
        mean_loss = 0
        bpr_loss = 0
        gt_loss = 0
        rc_loss = 0
        text_rc_loss = 0
        original_loss = 0
        while start_inx < len(log_emb_):
            log_emb = log_emb_[start_inx:end_inx]
            pos_emb = pos_emb_[start_inx:end_inx]
            neg_emb = neg_emb_[start_inx:end_inx]
            
            pos__ = pos_[start_inx:end_inx]
            neg__ = neg_[start_inx:end_inx]
            
            start_inx = end_inx
            end_inx += 60
            iterss += 1
            
            pos_text = self.find_item_text(pos__)
            neg_text = self.find_item_text(neg__)
            
            pos_token = self.sbert.tokenize(pos_text)
            pos_text_embedding = self.sbert({'input_ids': pos_token['input_ids'].to(self.device), 'attention_mask': pos_token['attention_mask'].to(self.device)})['sentence_embedding']
            neg_token = self.sbert.tokenize(neg_text)
            neg_text_embedding = self.sbert({'input_ids': neg_token['input_ids'].to(self.device), 'attention_mask': neg_token['attention_mask'].to(self.device)})['sentence_embedding']
            
            pos_text_matching, pos_proj = self.mlp(pos_emb)
            neg_text_matching, neg_proj = self.mlp(neg_emb)

            self.cross_modal = CrossModalAttention(self.rec_sys_dim, self.sbert_dim, self.device)
            pos_proj = self.rec_proj(pos_emb)  # Project to same 256 dim
            
            pos_fused = self.cross_modal(log_emb, pos_text_embedding)
            neg_fused = self.cross_modal(log_emb, neg_text_embedding)

            # Project pos_emb to match dimensions before multiplication
            pos_emb_proj = nn.Linear(pos_emb.size(-1), pos_fused.size(-1)).to(self.device)
            projected_pos_emb = pos_emb_proj(pos_emb)

            # Now both tensors will have matching dimensions
            pos_logits = (pos_fused * projected_pos_emb).mean(axis=1)

            neg_emb_proj_layer = nn.Linear(neg_emb.size(-1), neg_fused.size(-1)).to(self.device)
            projected_neg_emb = neg_emb_proj_layer(neg_emb)
            neg_logits = (neg_fused * projected_neg_emb).mean(axis=1)

            pos_labels = torch.ones(pos_logits.shape, device=pos_logits.device)
            neg_labels = torch.zeros(neg_logits.shape, device=neg_logits.device)

            loss = self.bce_criterion(pos_logits, pos_labels)
            loss += self.bce_criterion(neg_logits, neg_labels)

            total_loss = loss
            total_loss.backward()
            optimizer.step()
            
        print("Loss in epoch {}/{} iteration {}/{}: {} / BPR loss: {} / Matching loss: {} / Item reconstruction: {} / Text reconstruction: {}".format(epoch, total_epoch, step, total_step, mean_loss/iterss, bpr_loss/iterss, gt_loss/iterss, rc_loss/iterss, text_rc_loss/iterss))

    def pre_train_phase2(self, data, optimizer, batch_iter):
        epoch, total_epoch, step, total_step = batch_iter
            
        optimizer.zero_grad()
        u, seq, pos, neg = data
        mean_loss = 0
        
        text_input = []
        text_output = []
        interact_embs = []
        candidate_embs = []
        self.llm.eval()
        
        with torch.no_grad():
            log_emb = self.recsys.model(u, seq, pos, neg, mode='log_only')
            
        for i in range(len(u)):
            target_item_id = pos[i][-1]
            target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
            
            interact_text, interact_ids = self.make_interact_text(seq[i][seq[i] > 0], 10)
            candidate_num = 20
            candidate_text, candidate_ids = self.make_candidate_text(seq[i][seq[i] > 0], candidate_num, target_item_id, target_item_title)
            
            input_text = ''
            input_text += ' is a user representation.'
                
            if self.args.rec_pre_trained_data == 'Movies_and_TV':
                input_text += 'This user has watched '
            elif self.args.rec_pre_trained_data == 'Video_Games':
                input_text += 'This user has played '
            elif self.args.rec_pre_trained_data == 'Luxury_Beauty' or self.args.rec_pre_trained_data == 'Toys_and_Games':
                input_text += 'This user has bought '
                
            input_text += interact_text
            
            if self.args.rec_pre_trained_data == 'Movies_and_TV':
                input_text += ' in the previous. Recommend one next movie for this user to watch next from the following movie title set, '
            elif self.args.rec_pre_trained_data == 'Video_Games':
                input_text += ' in the previous. Recommend one next game for this user to play next from the following game title set, '            
            elif self.args.rec_pre_trained_data == 'Luxury_Beauty' or self.args.rec_pre_trained_data == 'Toys_and_Games':
                input_text += ' in the previous. Recommend one next item for this user to buy next from the following item title set, '
                    
            input_text += candidate_text
            input_text += '. The recommendation is '

            text_input.append(input_text)
            text_output.append(target_item_title)

            interact_embs.append(self.item_emb_proj(self.get_item_emb(interact_ids)))
            candidate_embs.append(self.item_emb_proj(self.get_item_emb(candidate_ids)))
            
        samples = {'text_input': text_input, 'text_output': text_output, 'interact': interact_embs, 'candidate': candidate_embs}
        log_emb = self.log_emb_proj(log_emb)
        loss_rm = self.llm(log_emb, samples)
        loss_rm.backward()
        optimizer.step()
        mean_loss += loss_rm.item()
        print("A-LLMRec model loss in epoch {}/{} iteration {}/{}: {}".format(epoch, total_epoch, step, total_step, mean_loss))

    def generate(self, data):
        u, seq, pos, neg, rank = data
        
        answer = []
        text_input = []
        interact_embs = []
        candidate_embs = []
        with torch.no_grad():
            log_emb = self.recsys.model(u, seq, pos, neg, mode='log_only')
            for i in range(len(u)):
                target_item_id = pos[i]
                target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
                
                interact_text, interact_ids = self.make_interact_text(seq[i][seq[i] > 0], 10)
                candidate_num = 20
                candidate_text, candidate_ids = self.make_candidate_text(seq[i][seq[i] > 0], candidate_num, target_item_id, target_item_title)
                
                input_text = ''
                input_text += ' is a user representation.'
                if self.args.rec_pre_trained_data == 'Movies_and_TV':
                    input_text += 'This user has watched '
                elif self.args.rec_pre_trained_data == 'Video_Games':
                    input_text += 'This user has played '
                elif self.args.rec_pre_trained_data == 'Luxury_Beauty' or self.args.rec_pre_trained_data == 'Toys_and_Games':
                    input_text += 'This user has bought '
                    
                input_text += interact_text
                
                if self.args.rec_pre_trained_data == 'Movies_and_TV':
                    input_text += ' in the previous. Recommend one next movie for this user to watch next from the following movie title set, '
                elif self.args.rec_pre_trained_data == 'Video_Games':
                    input_text += ' in the previous. Recommend one next game for this user to play next from the following game title set, '            
                elif self.args.rec_pre_trained_data == 'Luxury_Beauty' or self.args.rec_pre_trained_data == 'Toys_and_Games':
                    input_text += ' in the previous. Recommend one next item for this user to buy next from the following item title set, '
                
                input_text += candidate_text
                input_text += '. The recommendation is '
                
                answer.append(target_item_title)
                text_input.append(input_text)
                
                interact_embs.append(self.item_emb_proj(self.get_item_emb(interact_ids)))
                candidate_embs.append(self.item_emb_proj(self.get_item_emb(candidate_ids)))
        
        log_emb = self.log_emb_proj(log_emb)
        atts_llm = torch.ones(log_emb.size()[:-1], dtype=torch.long).to(self.device)
        atts_llm = atts_llm.unsqueeze(1)
        log_emb = log_emb.unsqueeze(1)
        
        with torch.no_grad():
            self.llm.llm_tokenizer.padding_side = "left"
            llm_tokens = self.llm.llm_tokenizer(
                text_input,
                padding="longest",
                return_tensors="pt"
            ).to(self.device)
            
            with torch.cuda.amp.autocast():
                inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens.input_ids)
                
                llm_tokens, inputs_embeds = self.llm.replace_hist_candi_token(llm_tokens, inputs_embeds, interact_embs, candidate_embs)
                    
                attention_mask = llm_tokens.attention_mask
                inputs_embeds = torch.cat([log_emb, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)
                    
                outputs = self.llm.llm_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=False,
                    top_p=0.9,
                    temperature=1,
                    num_beams=1,
                    max_length=512,
                    min_length=1,
                    pad_token_id=self.llm.llm_tokenizer.eos_token_id,
                    repetition_penalty=1.5,
                    length_penalty=1,
                    num_return_sequences=1,
                )

            outputs[outputs == 0] = 2  # convert output id 0 to 2 (eos_token_id)
            output_text = self.llm.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [text.strip() for text in output_text]

        for i in range(len(text_input)):
            with open(f'./recommendation_output.txt', 'a') as f:
                f.write(text_input[i])
                f.write('\n\n')
                
                f.write('Answer: ' + answer[i])
                f.write('\n\n')
                
                f.write('LLM: ' + str(output_text[i]))
                f.write('\n\n')

        return output_text

        


    