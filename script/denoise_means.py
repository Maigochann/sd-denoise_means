import os
from typing import Callable, List, Union

import gradio as gr

from modules import scripts
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img
from modules.script_callbacks import (CFGDenoisedParams, CFGDenoiserParams,
									  on_cfg_denoised, on_cfg_denoiser)

NAME = 'denoise means'

class Script(scripts.Script):
	def __init__(self):
		pass

	def title(self):
		return NAME

	
	def show(self, is_img2img):
		return scripts.AlwaysVisible

	def ui(self, is_img2img):
		with gr.Group():
			with gr.Accordion(NAME, open=False):
				enabled = gr.Checkbox(value=False, label="Enabled")
		
		return [enabled]
    # 均值降噪
	def cb(self, params: CFGDenoiserParams):
		if self.enabled:
			params.x = params.x-params.x.mean(dim=(-2,-1)).unsqueeze(-1).unsqueeze(-1).expand(params.x.shape)
	
	def process(self, p: StableDiffusionProcessing, enabled: bool):
		self.enabled = enabled

		if not self.enabled:
			return
		
		if not hasattr(self, 'callbacks_added'):
			on_cfg_denoiser(self.cb)
			self.callbacks_added = True

		p.extra_generation_params.update({
			f'{NAME} enabled': enabled,
		})

		return

def __set_value(p: StableDiffusionProcessing, script: type, index: int, value):
	args = list(p.script_args)
	
	if isinstance(p, StableDiffusionProcessingTxt2Img):
		all_scripts = scripts.scripts_txt2img.scripts
	else:
		all_scripts = scripts.scripts_img2img.scripts
	
	froms = [x.args_from for x in all_scripts if isinstance(x, script)]
	for idx in froms:
		assert idx is not None
		args[idx + index] = value
	
	p.script_args = type(p.script_args)(args)


def to_bool(v: str):
	if len(v) == 0: return False
	v = v.lower()
	if 'true' in v: return True
	if 'false' in v: return False
	
	try:
		w = int(v)
		return bool(w)
	except:
		acceptable = ['True', 'False', '1', '0']
		s = ', '.join([f'`{v}`' for v in acceptable])
		raise ValueError(f'value must be one of {s}.')

__init = False

def init_xyz(script: type, ext_name: str):
	global __init
	
	if __init:
		return
	
	for data in scripts.scripts_data:
		name = os.path.basename(data.path)
		if name == 'xy_grid.py' or name == 'xyz_grid.py':
			AxisOption = data.module.AxisOption
			
			def define(param: str, index: int, type: Callable, choices: List[str] = []):
				def fn(p, x, xs):
					__set_value(p, script, index, x)
				if len(choices) == 0:
					data.module.axis_options.append(AxisOption(f'{ext_name} {param}', type, fn))
				else:
					data.module.axis_options.append(AxisOption(f'{ext_name} {param}', type, fn, choices=lambda: choices))
			
			define('Enabled', 0, to_bool, choices=['false', 'true'])
			
	__init = True

init_xyz(Script,f'[{NAME}]')