import warnings
warnings.filterwarnings("ignore")

from models.attention import C_MultiheadAttention
from models.vit import VisionTransformer, interpolate_pos_embed
from models.med import BertConfig, BertModel, BertMLMLMHeadModel
from models.blip import create_vit, init_tokenizer
from models.clip_models import mlm_model,GELU
from collections import OrderedDict
from transformers import BertTokenizer
import torch
from torch import nn
import torch.nn.functional as F
import os
import nltk
import math
from module import EfficientConceptRouter, EnhancedConceptMatcher


class MLP_En(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim=512, hidden_dim=512, output_dim=512, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        if num_layers > 0:
            h = [hidden_dim] * (num_layers - 1)
            self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip(
                [input_dim] + h, h + [output_dim]))
        else:
            self.layers = []

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ACR(nn.Module):
    def __init__(self, d_model=512, vision_width=1024, text_width=768, mlp=False):
        super().__init__()
        self.concept_router = EfficientConceptRouter(d_model)
        self.img2text_attn = C_MultiheadAttention(embed_dim=d_model, num_heads=8, dropout=0.1)
        self.mlp = mlp
        if self.mlp:
            self.text_proj = MLP_En(input_dim=d_model, hidden_dim=d_model, output_dim=d_model, num_layers=1)
            self.img_proj = MLP_En(input_dim=d_model, hidden_dim=d_model, output_dim=d_model, num_layers=1)

        self.tf_pow = 2.0
        self.tf_scale = nn.Parameter(torch.Tensor([1.0]))
        self.tf_sigma = nn.Parameter(torch.Tensor([0.5]))
        self.norm_img = nn.LayerNorm(d_model)

    def forward(self, img_feat, np_feat, word_feat, word_key_padding_mask=None, np_key_padding_mask=None,
                img_f=None, text_f=None):
        routed_feature = self.concept_router(img_feat, np_feat, word_feat,
                                             np_key_padding_mask=np_key_padding_mask,
                                             word_key_padding_mask=word_key_padding_mask)
        F_s_np = self.img2text_attn(img_feat, np_feat, word_feat, routed_feature=routed_feature,
                                    word_key_padding_mask=word_key_padding_mask,
                                    np_key_padding_mask=np_key_padding_mask)[0]

        if self.mlp:
            text_embed = self.text_proj(img_feat + F_s_np)
            img_embed = self.img_proj(img_feat)
        else:
            text_embed = F_s_np
            img_embed = img_feat

        dis_coef = (F.normalize(img_embed, p=2, dim=-1) * F.normalize(text_embed, p=2, dim=-1)).sum(dim=-1, keepdim=True)
        dis_coef = self.tf_scale * torch.exp(- (1 - dis_coef).pow(self.tf_pow) / (2 * self.tf_sigma ** 2))  # norm distribution
        dis_weight = dis_coef.squeeze(-1)
        fuse_img_feat = self.norm_img(img_feat) * dis_coef

        # normalized features
        image_global_norm = img_f / img_f.norm(dim=-1, keepdim=True)
        text_global_norm = text_f / text_f.norm(dim=-1, keepdim=True)
        token_feas_norm = fuse_img_feat / fuse_img_feat.norm(dim=-1, keepdim=True)

        dis_mask = dis_weight > 0.1

        i2w_score = torch.matmul(image_global_norm.unsqueeze(1), token_feas_norm.transpose(-1, -2)).squeeze(1) * dis_mask
        t2w_score = torch.matmul(text_global_norm.unsqueeze(1), token_feas_norm.transpose(-1, -2)).squeeze(1) * dis_mask
        loss_scl = (torch.norm((i2w_score - t2w_score), p=2, dim=-1) ** 2).mean() * 0.5

        loss_vis = 1 - F.cosine_similarity(routed_feature, image_global_norm)
        loss_txt = 1 - F.cosine_similarity(routed_feature, text_global_norm)
        loss_route = (loss_vis + loss_txt).mean()

        loss_scl_g = loss_scl + 0.5 * loss_route

        return loss_scl_g


class SCRNet(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=224,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 num_classes=11003,
                 queue_size = 57600,
                 momentum = 0.995,
                 scl=False,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()
        self.task = ['global','local','seg','attr']

        self.num_classes = num_classes
        self.logit_scale = torch.ones([]) * (1 / 0.02)
        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07*torch.ones([]))
        self.embed_dim = 256
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer,jigsaw=True)
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.config=med_config
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)
        if 'local' not in self.task:
            for k, v in self.text_encoder.named_parameters():
                if 'cross' in k:
                    v.requires_grad = False
        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, self.embed_dim)
        self.text_proj = nn.Linear(text_width, self.embed_dim)

        # create momentum encoders
        self.visual_encoder_m, vision_width = create_vit(vit, image_size)
        self.vision_proj_m = nn.Linear(vision_width, self.embed_dim)
        self.text_encoder_m = BertModel(config=med_config, add_pooling_layer=False)
        self.text_proj_m = nn.Linear(text_width, self.embed_dim)
        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                        ]
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
        self.register_buffer("ptr_queue", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)


        if 'local' in self.task:
            self.itm_head = nn.Linear(text_width, 2)
            self.itm_local_head = nn.Linear(text_width, 2)
        if 'attr' in self.task:
            self.text_decoder = BertMLMLMHeadModel(config=med_config)
        if scl:
            self.extra_encoder_layer = ACR(d_model=text_width, vision_width=vision_width, text_width=text_width, mlp=True)
            for p in self.extra_encoder_layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

            self.concept_matcher = EnhancedConceptMatcher(embed_dim, text_width, configs['max_concepts'])
            for p in self.concept_matcher.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, image, caption, alpha, idx, caps=None):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=56,
                              return_tensors="pt").to(image.device)
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                        return_dict=True, mode='text')
        text_embeds = text_output.last_hidden_state
        image_embeds = self.visual_encoder(image)

        text_feats = self.text_proj(text_embeds[:,0,:])

        image_feats = self.vision_proj(image_embeds[:,0,:])

        tokens = [self.tokenizer.tokenize(cap) for cap in caption]

        ###============== Image-text Contrastive Learning ===================###
        image_feats = F.normalize(image_feats, dim=-1)
        text_feats = F.normalize(text_feats, dim=-1)
        global_loss, idx, image_feat_m, text_feat_m = self.get_itc_loss(image_feats, text_feats, image, text, alpha, idx)
        idxs = concat_all_gather(idx)
        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idxs)

        if caps is not None:
            np_text_input = self.tokenizer(caps, padding='max_length', truncation=True, max_length=56, return_tensors="pt").to(image.device)
            np_text_ids, np_text_atts = np_text_input.input_ids, np_text_input.attention_mask
            np_text_output = self.text_encoder(np_text_ids, attention_mask=np_text_atts, return_dict=True, mode='text')
            y_mask1 = (text.attention_mask == 0)   # transfer pad token to ignore tokens 33
            y_mask1[:, 0] = True  # transfer cls token to ignore tokens
            id_concepts, id_concept_atts = self.get_identity_concept_feas(caps, np_text_input.input_ids, np_text_output.last_hidden_state, config['max_tokens'], self.tokenizer, image.device)
            y_mask = (id_concept_atts == 0)   # transfer pad token to ignore tokens
            loss_scl = self.extra_encoder_layer(image_embeds[:, 1:, :], id_concepts,
                                                text_output.last_hidden_state, word_key_padding_mask=y_mask1, np_key_padding_mask=y_mask, img_f=image_embeds[:, 0, :], text_f=text_output.last_hidden_state[:, 0, :])
            global_loss = global_loss + loss_scl

        if 'local' in self.task:
            local_loss = self.itm_loss(image_embeds, text, image_feats, text_feats, idx)

        if 'attr' in self.task:
            ara_loss = self.ara_loss(text, tokens, image_embeds)

        return global_loss, local_loss, ara_loss

    def get_itc_loss(self, image_feat, text_feat, image, text, alpha, idx):
        idx = idx.view(-1,1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()],dim=1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image)
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]), dim=-1)
            image_feat_m_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)

            text_output_m = self.text_encoder_m(text.input_ids, attention_mask=text.attention_mask,
                                                return_dict=True, mode='text')
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]), dim=-1)
            text_feat_m_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            sim_i2t_m = image_feat_m @ text_feat_m_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_m_all / self.temp

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_m_all / self.temp
        sim_t2i = text_feat @ image_feat_m_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2

        return loss_ita, idx, image_feat_m, text_feat_m, image_embeds_m


    def itm_loss(self, image_embeds, text, image_feat, text_feat, idx):

        idx = idx.view(-1,1)

        encoder_input_ids = text.input_ids.clone()

        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id  # change [CLS] to special token
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        # forward the positve image-text pair
        bs = image_embeds.shape[0]
        output_pos = self.text_encoder(encoder_input_ids,  # fusion of image and text from one pair
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       )
        # compute sample similarity
        with torch.no_grad():
            mask = torch.eq(idx, idx.t())

            sim_i2t = image_feat @ text_feat.t() / self.temp
            sim_t2i = text_feat @ image_feat.t() / self.temp

            weights_i2t = F.softmax(sim_i2t, dim=1) #+ 1e-4
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i = F.softmax(sim_t2i, dim=1) #+ 1e-4# for selecting hard negative
            weights_t2i.masked_fill_(mask, 0)  # exclude positive sample

        # select a negative image (from all ranks) for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)

        # select a negative text (from all ranks) for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.argmax(weights_i2t[b])
            text_ids_neg.append(encoder_input_ids[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)  # ||                ||
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        output_neg = self.text_encoder(text_ids_all,
                                       attention_mask=text_atts_all,
                                       encoder_hidden_states=image_embeds_all,
                                       encoder_attention_mask=image_atts_all,
                                       return_dict=True,
                                       )

        last_hid = output_pos.last_hidden_state[:, 1:, :]
        pos_segments = [last_hid[:, :36], last_hid[:, 36:72]]
        pos_segments = torch.cat(pos_segments, dim=0)
        pos_segments = torch.mean(pos_segments, dim=1)

        last_hid = output_neg.last_hidden_state[:, 1:, :]
        neg_segments = [last_hid[:, :36], last_hid[:, 36:72]]
        neg_segments = torch.cat(neg_segments, dim=0)
        neg_segments = torch.mean(neg_segments, dim=1)

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]],
                                  dim=0)  # multi-feat
        vl_output = self.itm_head(vl_embeddings)  # (bs*3) * 2

        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(image_embeds.device)  # (bs * 3)

        lc_embeddings = torch.cat([pos_segments, neg_segments], dim=0)
        lc_labels = torch.cat([torch.ones(2 * bs, dtype=torch.long), torch.zeros(4 * bs, dtype=torch.long)],
                              dim=0).to(image_embeds.device)
        lc_output = self.itm_local_head(lc_embeddings)
        loss_lc = F.cross_entropy(lc_output, lc_labels)
        loss = F.cross_entropy(vl_output, itm_labels) + loss_lc

        return loss

    def ara_loss(self,text,token,image_embeds):

        loss = 0.
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        mask_input_ids = text.input_ids.clone()
        mask_labels = mask_input_ids.clone()
        mask_probability = 0.4

        probability_matrix = torch.full(mask_labels.shape, mask_probability)
        mask_input_ids, mask_labels = self.attr_mask(mask_input_ids, token,
                                                   targets=mask_labels, probability_matrix=probability_matrix)

        decoder_output_mlm = self.text_decoder(mask_input_ids,
                                               attention_mask=text.attention_mask,
                                               encoder_hidden_states=image_embeds,
                                               encoder_attention_mask=image_atts,
                                               labels=mask_labels,
                                               return_dict=True,
                                               output_hidden_states=True,
                                               task='mlm'
                                               )
        loss += decoder_output_mlm.loss


        return loss

    def attr_mask(self, input_ids, tokens, targets=None, masked_indices=None, probability_matrix=None):
        phrase = False
        if phrase == True:
            mask_pos = torch.zeros(probability_matrix.shape)
            grammar = "ATTR: {(<JJ><CC>?)*<NN>?<NNS>?}"
            cp = nltk.RegexpParser(grammar)

            for i, tok in enumerate(tokens):
                tree = cp.parse(nltk.pos_tag(tok))
                attr_tag = 0
                in_attr = False
                for j, pos in enumerate(tree.pos()):
                    if pos[1] == 'ATTR':
                        if in_attr:
                            pass
                        else:
                            attr_tag = attr_tag + 1
                            in_attr = True
                        if j < min(mask_pos.shape[1] - 1, 70):
                            mask_pos[i, j + 1] = attr_tag
                    else:
                        in_attr = False
                probability = torch.cat([torch.zeros(1), torch.ones(attr_tag) * 0.8])
                masked_attr = torch.bernoulli(probability)
                for j in range(min(mask_pos.shape[1], 70)):
                    mask_pos[i, j] = masked_attr[int(mask_pos[i, j])]
            if masked_indices is None:
                masked_indices = mask_pos.bool()
        else:
            for i, tok in enumerate(tokens):
                for j, pos in enumerate(nltk.pos_tag(tok), start=1):
                    if pos[1] not in ['JJ', 'NN', 'NNS', 'NNP', 'NNPS'] and j < 72:
                        probability_matrix[i, j] = 0
            if masked_indices is None:
                masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        input_ids[masked_indices] = self.tokenizer.mask_token_id
        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

def build_model(pretrained='',mode='train',**kwargs):
    model = SCL(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained,mode)
        print('missing keys:',msg.missing_keys)
    return model

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output      # rank_num * tensor_size

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)


def load_checkpoint(model, url_or_filename,mode='train'):
    if os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')

    state_dict = checkpoint['model']

    if 'ptr_queue' in state_dict.keys():
        del state_dict['ptr_queue']
    if 'image_queue' in state_dict.keys():
        del state_dict['image_queue']
    if 'text_queue' in state_dict.keys():
        del state_dict['text_queue']
    if 'idx_queue' in state_dict.keys():
        del state_dict['idx_queue']
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],
                                                                   model.visual_encoder)
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)


    print('load checkpoint from %s' % url_or_filename)
    return model, msg


