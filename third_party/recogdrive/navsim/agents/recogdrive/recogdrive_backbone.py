from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from .utils.conversation import get_conv_template
from .recogdrive_conditioning import VLMConditioning

IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'

system_message = """
You are a vehicle trajectory prediction model for autonomous driving. Your task is to predict the ego vehicle's 4-second trajectory based on the following inputs: multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands. The input provides a 2-second history, and your output should ensure a safe trajectory for the next 4 seconds. Your predictions must adhere to the following metrics:
1. **No at-fault Collisions (NC)**: Avoid collisions with other objects/vehicles.
2. **Drivable Area Compliance (DAC)**: Stay within the drivable area.
3. **Time to Collision (TTC)**: Maintain a safe distance from other vehicles.
4. **Ego Progress (EP)**: Ensure the ego vehicle moves forward without being stuck.
5. **Comfort (C)**: Avoid sharp turns and sudden decelerations.
6. **Driving Direction Compliance (DDC)**: Align with the intended driving direction.
For evaluation, use the **PDM Score**, which combines these metrics: **PDM Score** = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12.
Your predictions will be evaluated through a non-reactive 4-second simulation with an LQR controller and background actors following their recorded trajectories. The better your predictions, the higher your score.
"""

class RecogDriveBackbone(nn.Module):
    """
    A simplified vision-language model backbone with direct loading logic
    for different model architectures (InternVL, Qwen-VL).
    """
    def __init__(self,
                 model_type: str,
                 checkpoint_path: str,
                 device: str = "cuda"):
        """
        Initializes and loads the specified model and its preprocessor/tokenizer.

        Args:
            model_type (str): The type of model to load. Supported: 'internvl', 'qwen'.
            checkpoint_path (str): The path to the model checkpoint.
            device (str): The device to load the model onto ('cuda', 'cpu').
        """
        super().__init__()

        self.model = None
        self.tokenizer = None  
        self.model_type = model_type.lower()
        self.device = device

        print(f"Initializing backbone of type: '{self.model_type}' from path: '{checkpoint_path}'")

        if self.model_type == 'internvl':
            # --- Load InternVL Model and Tokenizer ---
            self.model = AutoModel.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_flash_attn=True,
                device_map=self.device
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
                use_fast=False
            )
            # Load model-specific configuration
            self._configure_internvl()
            self.num_image_token = 256

        elif self.model_type == 'qwen':
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True
            )
            self.tokenizer = AutoProcessor.from_pretrained(
                checkpoint_path,
                trust_remote_code=True
            )
            
        else:
            raise ValueError(f"Unsupported model_type: '{self.model_type}'. Please choose 'internvl' or 'qwen'.")


        print(f"Backbone '{self.model_type}' loaded successfully on device '{self.device}'.")

    def _configure_internvl(self):
        """Applies specific configurations required for the InternVL model."""
        self.model.system_message = system_message
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self.img_context_token_id
        print("InternVL model configured.")
    
    def _prepare_inputs(self, pixel_values, questions, num_patches_list, system_prompt=None):
        if not self.model:
            raise RuntimeError("Backbone model has not been initialized. Call initialize() on the agent first.")
        
        if len(questions) != len(num_patches_list) or sum(num_patches_list) != pixel_values.shape[0]:
            raise ValueError("Image patches must match the questions and per-sample patch counts")
        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            
            template = get_conv_template("internvl2_5")
            template.system_message = system_message if system_prompt is None else system_prompt
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)
        self.tokenizer.padding_side = 'left'
        # Keep the historical minimum width without truncating longer prompts.
        model_inputs = self.tokenizer(queries, return_tensors='pt', padding=True)
        padding = max(0, 2800 - model_inputs['input_ids'].shape[1])
        if padding:
            model_inputs['input_ids'] = torch.nn.functional.pad(
                model_inputs['input_ids'], (padding, 0), value=self.tokenizer.pad_token_id)
            model_inputs['attention_mask'] = torch.nn.functional.pad(model_inputs['attention_mask'], (padding, 0))
        device = next(self.model.parameters()).device
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return input_ids, attention_mask, position_ids

    def forward(self, pixel_values: torch.Tensor, questions: List[str], num_patches_list: List[int], *, system_prompt=None):
        input_ids, attention_mask, position_ids = self._prepare_inputs(
            pixel_values, questions, num_patches_list, system_prompt)
        model_dtype = next(self.model.parameters()).dtype
        num_patches = pixel_values.size(0)
        image_flags = torch.ones(num_patches, dtype=torch.long, device=input_ids.device)
        outputs = self.model(
                pixel_values=pixel_values.to(device=input_ids.device, dtype=model_dtype),
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_flags=image_flags,
                output_hidden_states=True,
                return_dict=True,
        )
        outputs['context_valid_mask'] = attention_mask.bool()
        return outputs

    def rotary_contract(self):
        decoder = self.model.language_model.model
        rotary = decoder.layers[0].self_attn.rotary_emb
        return {"implementation": type(rotary).__name__, "head_dim": rotary.inv_freq.numel() * 2,
                "theta": float(rotary.base)}

    @torch.no_grad()
    def extract_conditioning(self, pixel_values, questions, num_patches_list, *, layers=(), action_horizon=8):
        """Prefill InternVL's decoder once, bypassing its unused vocabulary head."""
        if self.model_type != 'internvl':
            raise ValueError("Online stage-2 conditioning currently requires InternVL/Qwen2")
        self.eval()
        input_ids, mask, positions = self._prepare_inputs(pixel_values, questions, num_patches_list)
        decoder = self.model.language_model.model
        if any(type(i) is not int or not 0 <= i < len(decoder.layers) for i in layers):
            raise ValueError("Selected KV layer is outside the VLM decoder")
        embeddings = self.model.language_model.get_input_embeddings()(input_ids).clone()
        # Match InternVL's image-token replacement exactly; reject mismatched images.
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            visual = self.model.extract_feature(pixel_values.to(device=embeddings.device, dtype=embeddings.dtype))
            selected = input_ids == self.img_context_token_id
            flat_visual = visual.reshape(-1, embeddings.shape[-1])
            if int(selected.sum()) != flat_visual.shape[0]:
                raise ValueError("VLM image tokens do not match extracted image embeddings")
            embeddings[selected] = flat_visual.to(embeddings.dtype)
            output = decoder(inputs_embeds=embeddings, attention_mask=mask, position_ids=positions,
                             use_cache=bool(layers), output_hidden_states=False, return_dict=True)
        cache = output.past_key_values
        if layers and hasattr(cache, "to_legacy_cache"):
            cache = cache.to_legacy_cache()
        selected_cache = tuple((cache[i][0].detach(), cache[i][1].detach()) for i in layers)
        query_positions = mask.sum(-1)[:, None] + torch.arange(action_horizon, device=mask.device)[None, :]
        factors = None
        if layers:
            rotary = decoder.layers[layers[0]].self_attn.rotary_emb
            cos, sin = rotary(selected_cache[0][1], seq_len=mask.shape[1] + action_horizon)
            factors = (cos[query_positions].unsqueeze(1), sin[query_positions].unsqueeze(1))
        return output.last_hidden_state.detach(), VLMConditioning(
            valid_mask=mask.bool(), kv_cache=selected_cache,
            query_position_ids=query_positions, query_rope=factors)

    
