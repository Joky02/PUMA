import os.path
import pickle
import multiprocessing as mp
import numpy as np
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from functools import partial
import logging
import torch
import pyscipopt as scip
from functools import partial

from src.utils import get_A_b_c


def preprocess_(file: str, config: DictConfig):
	"""
	Preprocess a single instance file.
	"""

	file_path = os.path.join(config.paths.train_data_dir, file)

	# vars:  [obj coeff, norm_coeff, degree, Bin?]
	m = scip.Model()
	m.hideOutput(True)
	m.readProblem(file_path)

	ncons = m.getNConss()
	nvars = m.getNVars()

	vars = m.getVars()
	vars.sort(key=lambda v: v.name)

	v_nodes = []

	b_vars = []

	ori_start = 6
	emb_num = 15

	for i in range(len(vars)):
		tp = [0] * ori_start
		tp[3] = 0
		tp[4] = 1e+20
		if vars[i].vtype() == 'BINARY':
			tp[ori_start - 1] = 1
			b_vars.append(i)

		v_nodes.append(tp)
	v_map = {}

	for indx, v in enumerate(vars):
		v_map[v.name] = indx

	obj = m.getObjective()
	for e in obj:
		vnm = e.vartuple[0].name
		v = obj[e]
		v_indx = v_map[vnm]
		v_nodes[v_indx][0] = v

	indices_spr = [[], []]
	values_spr = []

	cons = []
	for cind, c in enumerate(m.getConss()):
		coeff = m.getValsLinear(c)
		if len(coeff) != 0:
			cons.append(c)

	cons_map = [[x, len(m.getValsLinear(x))] for x in cons]

	cons_map = sorted(cons_map, key=lambda x: [x[1], str(x[0])])
	cons = [x[0] for x in cons_map]

	c_nodes = []
	for cind, c in enumerate(cons):
		coeff = m.getValsLinear(c)
		rhs = m.getRhs(c)
		lhs = m.getLhs(c)
		sense = 0

		if rhs == lhs:
			sense = 2
		elif rhs >= 1e+20:
			sense = 1
			rhs = lhs

		summation = 0
		for k in coeff:
			v_indx = v_map[k]
			if coeff[k] != 0:
				indices_spr[0].append(cind)
				indices_spr[1].append(v_indx)
				values_spr.append(coeff[k])
			v_nodes[v_indx][2] += 1
			v_nodes[v_indx][1] += coeff[k] / ncons
			v_nodes[v_indx][3] = max(v_nodes[v_indx][3], coeff[k])
			v_nodes[v_indx][4] = min(v_nodes[v_indx][4], coeff[k])
			summation += coeff[k]
		llc = max(len(coeff), 1)
		c_nodes.append([summation / llc, llc, rhs, sense])
	v_nodes = torch.as_tensor(v_nodes, dtype=torch.float32)
	c_nodes = torch.as_tensor(c_nodes, dtype=torch.float32)
	b_vars = torch.as_tensor(b_vars, dtype=torch.int32)
	indices_spr = torch.as_tensor(indices_spr, dtype=torch.int64)
	values_spr = torch.as_tensor(values_spr, dtype=torch.float32)

	clip_max = [20000, 1, torch.max(v_nodes, 0)[0][2].item()]
	clip_min = [0, -1, 0]

	v_nodes[:, 0] = torch.clamp(v_nodes[:, 0], clip_min[0], clip_max[0])

	maxs = torch.max(v_nodes, 0)[0]
	mins = torch.min(v_nodes, 0)[0]
	diff = maxs - mins
	for ks in range(diff.shape[0]):
		if diff[ks] == 0:
			diff[ks] = 1
	v_nodes = v_nodes - mins
	v_nodes = v_nodes / diff
	v_nodes = torch.clamp(v_nodes, 1e-5, 1)

	maxs = torch.max(c_nodes, 0)[0]
	mins = torch.min(c_nodes, 0)[0]
	diff = maxs - mins
	c_nodes = c_nodes - mins
	c_nodes = c_nodes / diff
	c_nodes = torch.clamp(c_nodes, 1e-5, 1)

	c_nodes[torch.isnan(c_nodes)] = 1
	c_nodes[torch.isinf(c_nodes)] = 1  # remove nan value

	# save the preprocessed data
	edge_indices = torch.LongTensor(np.array(indices_spr, dtype=int))
	edge_features = torch.FloatTensor(values_spr).reshape(-1, 1)
	graph = [c_nodes, edge_indices, edge_features, v_nodes]
	sample_path = os.path.join(config.paths.data_samples_dir, file.split(".")[0] + '.pkl')
	pickle.dump(graph, open(sample_path, 'wb'))

	# save the tensor data
	sample_tensor_path = os.path.join(config.paths.data_tensors_dir, file.split(".")[0] + '.pkl')
	A, b, c = get_A_b_c(file_path)
	pickle.dump((A, b, c), open(sample_tensor_path, 'wb'))

@hydra.main(version_base=None, config_path="config", config_name="preprocess")
def preprocess(config: DictConfig):
	logging.basicConfig(
		format="[%(asctime)s]: %(message)s",
		level=logging.DEBUG
	)

	os.makedirs(config.paths.data_samples_dir, exist_ok=True)
	os.makedirs(config.paths.data_solution_dir, exist_ok=True)
	os.makedirs(config.paths.data_solve_log_dir, exist_ok=True)
	os.makedirs(config.paths.data_tensors_dir, exist_ok=True)

	files = os.listdir(config.paths.train_data_dir)

	logging.info(f"Preprocessing the dataset {config.dataset.name} ({config.dataset.full_name}).")

	func = partial(preprocess_, config=config)
	with mp.Pool(config.num_workers) as pool:
		for _ in tqdm(pool.imap(func, files), total=len(files), desc="Collect Sample"):
			pass

	logging.info(f"Preprocessing done.")
	logging.info(f"The preprocessed data files are saved in {config.paths.preprocess_dir}.")


if __name__ == '__main__':
	preprocess()
