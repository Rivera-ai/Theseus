import html
import os
import re
import urllib.parse as ul
import ftfy
import torch
from bs4 import BeautifulSoup
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, T5EncoderModel
from Utils import readtexts_from_file
from colossalai.shardformer.modeling.jit import get_jit_fused_dropout_add_func
from colossalai.shardformer.modeling.t5 import get_jit_fused_T5_layer_ff_forward, get_T5_layer_self_attention_forward
from colossalai.shardformer.policies.base_policy import Policy, SubModuleReplacementDescription
import torch.nn as nn

# Algunas clases y funciones han sido tomadas del repo Open-Sora: https://github.com/hpcaitech/Open-Sora



def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states

    @staticmethod
    def from_native_module(module, *args, **kwargs):
        assert module.__class__.__name__ == "FusedRMSNorm", (
            "Recovering T5LayerNorm requires the original layer to be apex's Fused RMS Norm."
            "Apex's fused norm is automatically used by Hugging Face Transformers https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L265C5-L265C48"
        )

        layer_norm = T5LayerNorm(module.normalized_shape, eps=module.eps)
        layer_norm.weight.data.copy_(module.weight.data)
        layer_norm = layer_norm.to(module.weight.device)
        return layer_norm


class T5EncoderPolicy(Policy):
    def config_sanity_check(self):
        assert not self.shard_config.enable_tensor_parallelism
        assert not self.shard_config.enable_flash_attention

    def preprocess(self):
        return self.model

    def module_policy(self):
        from transformers.models.t5.modeling_t5 import T5LayerFF, T5LayerSelfAttention, T5Stack

        policy = {}

        # check whether apex is installed
        try:

            # recover hf from fused rms norm to T5 norm which is faster
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="layer_norm",
                    target_module=T5LayerNorm(),
                ),
                policy=policy,
                target_key=T5LayerFF(),
            )
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(suffix="layer_norm", target_module=T5LayerNorm),
                policy=policy,
                target_key=T5LayerSelfAttention(),
            )
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(suffix="final_layer_norm", target_module=T5LayerNorm),
                policy=policy,
                target_key=T5Stack(),
            )
        except (ImportError, ModuleNotFoundError):
            pass

        # use jit operator
        if self.shard_config.enable_jit_fused:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_jit_fused_T5_layer_ff_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=T5LayerFF(),
            )
            self.append_or_create_method_replacement(
                description={
                    "forward": get_T5_layer_self_attention_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=T5LayerSelfAttention(),
            )

        return policy

    def postprocess(self):
        return self.model

class T5Embedder:
    available_models = ["t5-v1_1-xxl"]
    bad_punct_regex = re.compile(r"[" + "#®•©™&@·º½¾¿¡§~" + "\)" + "\(" +
                                 "\]" + "\[" + "\}" + "\{" + "\|" + "\\" +
                                 "\/" + "\*" + r"]{1,}")  # noqa
    def __init__(self, from_pretrained, device, use_text_preprocessing=True, t5_model_kwargs=None, torch_dtype=None, model_max_length=120,):
        self.device = torch.device(device)


        self.torch_dtype = torch_dtype or torch.bfloat16
        if t5_model_kwargs is None:
            t5_model_kwargs = {
                "low_cpu_mem_usage": True,
                "torch_dtype": self.torch_dtype
            }
            t5_model_kwargs["device_map"] = {
                "shared": self.device,
                "encoder": self.device
            }

        self.use_text_preprocessing = use_text_preprocessing
        self.tokenizer = AutoTokenizer.from_pretrained(from_pretrained)
        self.model = T5EncoderModel.from_pretrained(from_pretrained).eval().to(self.device)
        self.model_max_length = model_max_length

    def get_text_embeddings(self, texts):
        texts = [self.text_preprocessing(text) for text in texts]

        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.model_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        text_tokens_and_mask["input_ids"] = text_tokens_and_mask["input_ids"]
        text_tokens_and_mask["attention_mask"] = text_tokens_and_mask[
            "attention_mask"]

        with torch.no_grad():
            text_encoder_embs = self.model(
                input_ids=text_tokens_and_mask["input_ids"].to(self.device),
                attention_mask=text_tokens_and_mask["attention_mask"].to(
                    self.device),
            )["last_hidden_state"].detach()
        return text_encoder_embs, text_tokens_and_mask["attention_mask"].to(self.device)

    def text_preprocessing(self, text):
        if self.use_text_preprocessing:
            text = self.clean_caption(text)
            text = self.clean_caption(text)
            return text
        else:
            return text.lower().strip()

    @staticmethod
    def basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)",
                         "", caption)

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(self.bad_punct_regex, r" ",
                         caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = self.basic_clean(caption)

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(
            r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "",
            caption)
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ",
                         caption)  

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()

class T5Encoder:
    def __init__(self, from_pretrained=None, model_max_length=120, device="cuda", dtype=torch.bfloat16, shardformer=False, ):
        assert from_pretrained is not None, "Por favor especifica el path del modelo T5"

        self.t5 = T5Embedder(
            from_pretrained=from_pretrained,
            device=device,
            torch_dtype=dtype,
            model_max_length=model_max_length
        )

        self.t5.model.to(dtype=dtype)
        self.y_embedder = None
        self.model_max_length = model_max_length
        self.output_dim = self.t5.model.config.d_model

        if shardformer:
            self.shardformer_t5()
    
    def shardformer_t5(self):
        from colossalai.shardformer import ShardConfig, ShardFormer

        shard_config = ShardConfig(
            tensor_parallel_process_group=None,
            pipeline_stage_manager=None,
            enable_tensor_parallelism=False,
            enable_fused_normalization=False,
            enable_flash_attention=False,
            enable_jit_fused=True,
            enable_sequence_parallelism=False,
        )

        shard_former = ShardFormer(shard_config=shard_config)
        optim_model, _ = shard_former.optimize(self.t5.model, policy=T5EncoderPolicy())

        self.t5.model = optim_model.half()

        requires_grad(self.t5.model, False)

    def encode(self, text):
        caption_embs, emb_masks = self.t5.get_text_embeddings(text)
        caption_embs = caption_embs[:, None]
        return dict(y=caption_embs, mask=emb_masks)
    
    def null(self, n):
        null_y = self.y_embedder.y_embedding[None].repeat(n, 1, 1)[:, None]
        return null_y