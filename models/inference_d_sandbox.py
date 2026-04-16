import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import BioGptForCausalLM, BioGptTokenizer


VIT_DIM = 768
DENSENET_DIM = 1024
PROJ_DIM = 1024
LABEL_MAP = {0: 'Normal', 1: 'Abnormal', 2: 'Unclear'}
DEVICE = torch.device('cpu')

VIT_TF = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

DN_TF = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

EFFNET_TF = DN_TF

_sandbox_runtime = None


def _ensure_biogpt_local(base_dir: str) -> str:
    local_dir = os.path.join(base_dir, 'model_files', 'hf_cache', 'microsoft_biogpt')
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        return local_dir

    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id='microsoft/biogpt',
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    return local_dir


def clean_output(text: str) -> str:
    sents = [s.strip() for s in str(text).split('.') if len(s.strip()) > 5]
    seen = set()
    unique = []
    for s in sents:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return '. '.join(unique[:6]).strip() or 'No findings'


def classify_text(report: str) -> str:
    text = str(report).lower()
    abnormal = [
        'cardiomegaly', 'pneumonia', 'effusion', 'pneumothorax',
        'consolidation', 'atelectasis', 'opacity', 'infiltrate',
        'edema', 'fracture', 'nodule', 'mass', 'fibrosis',
        'hyperinflat', 'pleural', 'enlarged', 'tortuous',
        'degenerative', 'scoliosis', 'granuloma', 'calcif',
    ]
    normal = [
        'no acute', 'normal', 'unremarkable', 'clear',
        'no significant', 'no evidence', 'negative',
        'within normal', 'no pneumothorax', 'no effusion',
        'no consolidation', 'no infiltrate',
    ]
    ab_hits = sum(1 for k in abnormal if k in text)
    no_hits = sum(1 for k in normal if k in text)
    if ab_hits > no_hits:
        return 'Abnormal'
    if no_hits >= ab_hits:
        return 'Normal'
    return 'Unclear'


def build_rag_prompt(retrieved_cases: list, tokenizer, max_ctx_tokens: int = 60):
    parts = []
    for case in retrieved_cases:
        report = case['report'].strip()
        tok_ids = tokenizer.encode(report, add_special_tokens=False)
        if len(tok_ids) > max_ctx_tokens:
            report = tokenizer.decode(tok_ids[:max_ctx_tokens], skip_special_tokens=True)
        parts.append(report)
    prompt_text = 'Similar cases: ' + ' | '.join(parts) + ' Generate report:'
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    return prompt_text, prompt_ids


def _discover_model_b_ckpt(base_dir: str) -> str:
    candidates = [
        os.environ.get('MODEL_B_CKPT'),
        os.path.join(base_dir, 'model_b_best.pth'),
        os.path.join(base_dir, 'model_files', 'models', 'model_b_best.pth'),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path

    for root, _, files in os.walk(base_dir):
        if 'model_b_best.pth' in files:
            return os.path.join(root, 'model_b_best.pth')

    raise FileNotFoundError(
        'model_b_best.pth not found. Set MODEL_B_CKPT or place the checkpoint in the repo.'
    )


class DualEncoderD(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.densenet = timm.create_model('densenet121', pretrained=True, num_classes=0)
        self.fusion = nn.Sequential(
            nn.Linear(VIT_DIM + DENSENET_DIM, PROJ_DIM),
            nn.LayerNorm(PROJ_DIM),
            nn.GELU(),
            nn.Linear(PROJ_DIM, PROJ_DIM),
        )
        for p in self.vit.parameters():
            p.requires_grad = False
        for p in self.densenet.parameters():
            p.requires_grad = False

    def forward(self, img_vit: torch.Tensor, img_dn: torch.Tensor) -> torch.Tensor:
        vit_feat = self.vit(img_vit)
        dn_feat = self.densenet(img_dn)
        fused = torch.cat([vit_feat, dn_feat], dim=1)
        return F.normalize(self.fusion(fused), p=2, dim=1)


class EncoderEfficientNet(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
        base = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.pool = base.avgpool
        self.proj = nn.Linear(1536, out_dim)
        self.bn = nn.BatchNorm1d(out_dim, momentum=0.01)
        for name, p in self.features.named_parameters():
            p.requires_grad = name.startswith('8')

    def forward(self, x):
        return self.bn(self.proj(self.pool(self.features(x)).flatten(1)))


class ModelB(nn.Module):
    def __init__(self, biogpt_hidden: int, pad_id: int, decoder_source: str):
        super().__init__()
        self.encoder = EncoderEfficientNet(biogpt_hidden)
        self.decoder = BioGptForCausalLM.from_pretrained(decoder_source)
        self.pad_id = pad_id

    def forward(self, images, input_ids, sample_labels=None):
        img_token = self.encoder(images).unsqueeze(1)
        tok_emb = self.decoder.biogpt.embed_tokens(input_ids)
        inputs_embeds = torch.cat([img_token, tok_emb], dim=1)
        img_mask = torch.ones(images.size(0), 1, device=images.device, dtype=torch.long)
        text_mask = (input_ids != self.pad_id).long()
        full_mask = torch.cat([img_mask, text_mask], dim=1)
        labels_t = input_ids.clone()
        labels_t[labels_t == self.pad_id] = -100
        labels_t = torch.cat([
            torch.full((input_ids.size(0), 1), -100, device=input_ids.device),
            labels_t,
        ], dim=1)
        out = self.decoder(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=labels_t)
        loss = out.loss
        if sample_labels is not None:
            w = torch.where(
                sample_labels.to(loss.device) == 1,
                torch.tensor(2.0, device=loss.device),
                torch.tensor(1.0, device=loss.device),
            )
            loss = loss * w.mean()
        return loss, out.logits


@dataclass
class SandboxRuntime:
    tokenizer: BioGptTokenizer
    model_b: ModelB
    dual_encoder: DualEncoderD
    faiss_d: object
    reports_d: np.ndarray
    labels_d: np.ndarray
    model_b_ckpt: str


def load_model_d_sandbox(base_dir: str) -> SandboxRuntime:
    global _sandbox_runtime
    if _sandbox_runtime is not None:
        return _sandbox_runtime

    import faiss
    try:
        import timm  # noqa: F401
    except ImportError as exc:
        raise RuntimeError('timm is required for sandbox Model D inference.') from exc

    index_d_dir = os.path.join(base_dir, 'model_files', 'index_d')
    faiss_d_path = os.path.join(index_d_dir, 'faiss_d.index')
    reports_d_path = os.path.join(index_d_dir, 'reports_d.npy')
    labels_d_path = os.path.join(index_d_dir, 'labels_d.npy')

    for path in [faiss_d_path, reports_d_path, labels_d_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Required sandbox asset missing: {path}')

    biogpt_dir = _ensure_biogpt_local(base_dir)
    tokenizer = BioGptTokenizer.from_pretrained(biogpt_dir)
    tokenizer.pad_token = tokenizer.eos_token

    model_b = ModelB(
        biogpt_hidden=1024,
        pad_id=tokenizer.pad_token_id,
        decoder_source=biogpt_dir,
    ).to(DEVICE)
    ckpt_path = _discover_model_b_ckpt(base_dir)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model_b.load_state_dict(state)
    model_b.eval()

    torch.manual_seed(42)
    np.random.seed(42)
    dual_encoder = DualEncoderD().to(DEVICE).eval()

    runtime = SandboxRuntime(
        tokenizer=tokenizer,
        model_b=model_b,
        dual_encoder=dual_encoder,
        faiss_d=faiss.read_index(faiss_d_path),
        reports_d=np.load(reports_d_path, allow_pickle=True),
        labels_d=np.load(labels_d_path),
        model_b_ckpt=ckpt_path,
    )
    _sandbox_runtime = runtime
    return runtime


@torch.no_grad()
def generate_model_d_sandbox(image_path: str, base_dir: Optional[str] = None) -> dict:
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(__file__))

    runtime = load_model_d_sandbox(base_dir)
    pil = Image.open(image_path).convert('RGB')

    iv = VIT_TF(pil).unsqueeze(0).to(DEVICE)
    idn = DN_TF(pil).unsqueeze(0).to(DEVICE)
    q_emb = runtime.dual_encoder(iv, idn)
    q_np = q_emb.cpu().numpy().astype(np.float32)

    scores, ret_idxs = runtime.faiss_d.search(q_np, 3)
    retrieved = []
    for rank, (score, ridx) in enumerate(zip(scores[0], ret_idxs[0])):
        ridx = int(ridx)
        retrieved.append({
            'rank': rank + 1,
            'score': float(score),
            'report': str(runtime.reports_d[ridx]),
            'label': LABEL_MAP.get(int(runtime.labels_d[ridx]), 'Unclear'),
        })

    _, prompt_ids = build_rag_prompt(retrieved, runtime.tokenizer, max_ctx_tokens=60)
    prompt_tensor = torch.tensor([prompt_ids], device=DEVICE)

    gen_img = EFFNET_TF(pil).unsqueeze(0).to(DEVICE)
    img_feat = runtime.model_b.encoder(gen_img)
    img_token = img_feat.unsqueeze(1)
    prompt_embs = runtime.model_b.decoder.biogpt.embed_tokens(prompt_tensor)
    inputs_embeds = torch.cat([img_token, prompt_embs], dim=1)
    attn_mask = torch.ones(inputs_embeds.shape[:2], device=DEVICE, dtype=torch.long)

    gen_ids = runtime.model_b.decoder.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        max_new_tokens=120,
        min_new_tokens=15,
        no_repeat_ngram_size=4,
        num_beams=4,
        early_stopping=True,
        do_sample=False,
        eos_token_id=runtime.tokenizer.eos_token_id,
        pad_token_id=runtime.tokenizer.pad_token_id,
    )

    report = clean_output(runtime.tokenizer.decode(gen_ids[0], skip_special_tokens=True))
    if 'Generate report:' in report:
        report = report.split('Generate report:')[-1].strip()
    if not report:
        raise RuntimeError('Sandbox generation returned empty text.')

    return {
        'text': report,
        'clinical_label': classify_text(report),
        'retrieved_cases': retrieved,
        'model_b_ckpt': runtime.model_b_ckpt,
    }
