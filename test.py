import gurobipy as gp
import os
import numpy as np
import torch
import hydra
from omegaconf import DictConfig

from src.solver_wrapper import BaseModelWrapper
from src.utils import set_seed, set_cpu_num, get_A_b_c, solve_problem
import src.tb_writter as tb_writter
from src.model import GATPredictor, GNNPredictor, BipartiteNodeData
import logging
from functools import partial
import torch.multiprocessing as mp
from tqdm import tqdm
import json
import pyscipopt as scip

mp.set_start_method('spawn', force=True)



def get_varname(var):
	"""Get a variable's name uniformly across Gurobi, SCIP, etc."""
	return getattr(var, 'VarName', getattr(var, 'name', str(var)))


def add_trust_region(m: BaseModelWrapper, best_x: list, delta: float, v_map: dict):
	error_expr = 0
	for i, v in enumerate(m.get_vars()):

		v_indx = v_map[get_varname(v)]
		# set MIP start
		m.set_start(v, best_x[0][v_indx])
		# add alpha var
		alpha = m.add_var(name=f'alp_{i}', lb=0, ub=1, obj=0)
		# build absolute difference
		if best_x[0][v_indx] == 0:
			m.add_constr(alpha >= v, name=f'tr1_{i}')
		elif best_x[0][v_indx] == 1:
			m.add_constr(alpha >= 1 - v, name=f'tr2_{i}')
		# accumulate error
		error_expr += alpha
	m.add_constr(error_expr <= delta, name="sum_alpha")

def add_multi_trust_region(m: BaseModelWrapper, best_x: np.array, delta: float, v_map: dict):
	error_expr = 0
	y_sum = 0
	M = 100000
	vars = m.get_vars()
	for j in range(best_x.shape[0]):
		y = m.add_var(name=f'y_{j}', lb=0, ub=1, obj=0, vtype='B')
		y_sum += y
		for i, v in enumerate(vars):
			v_indx = v_map[get_varname(v)]
			# set MIP start
			# add alpha var
			alpha = m.add_var(name=f'alp_{j}_{i}', lb=0, ub=1, obj=0)
			# build absolute difference
			if best_x[j][v_indx] == 0:
				m.add_constr(alpha >= v, name=f'tr1_{j}_{i}')
			elif best_x[j][v_indx] == 1:
				m.add_constr(alpha >= 1 - v, name=f'tr2_{j}_{i}')

			# accumulate error
			error_expr += alpha
		m.add_constr(error_expr <= delta + (1 - y) * M, name="sum_alpha")
	m.add_constr(y_sum >= 1, name="sum_y")



def predict(test_ins_name, model: GNNPredictor, test_data_dir, mu, delta, num_start):
	"""Serial CPU+GPU preprocessing for a single instance.

	Returns (best_x, delta_abs, v_map) so that the parallel-pool stage only
	needs to run the solver.
	"""
	ins_name_to_read = os.path.join(test_data_dir, test_ins_name)

	# vars:  [obj coeff, norm_coeff, degree, Bin?]
	m = scip.Model()
	m.hideOutput(True)
	m.readProblem(ins_name_to_read)

	ncons = m.getNConss()
	nvars = m.getNVars()
	delta_val = float(delta)
	if 0.0 < delta_val < 1.0:
		delta_abs = int(round(delta_val * nvars))
	else:
		delta_abs = int(round(delta_val))
	delta_abs = max(delta_abs, 0)

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

	edge_indices = torch.LongTensor(np.array(indices_spr, dtype=int))
	edge_features = torch.FloatTensor(values_spr).reshape(-1, 1)

	graph = BipartiteNodeData(c_nodes, edge_indices, edge_features, v_nodes)
	A_sp, b, c = get_A_b_c(ins_name_to_read, sparse=True)

	import scipy.sparse as sp
	A_ind = A_sp.indices().numpy()
	A_val = A_sp.values().numpy()
	A_shape = tuple(A_sp.shape)
	A_csr = sp.csr_matrix((A_val, (A_ind[0], A_ind[1])), shape=A_shape)
	del A_sp, A_ind, A_val
	b = b.numpy().reshape(-1)
	c = c.numpy()

	graph = graph.cuda()
	with torch.no_grad():
		logits = model.forward(graph)[0]
	pred = logits.sigmoid().cpu().numpy()
	del graph, logits
	torch.cuda.empty_cache()
	x = np.random.binomial(1, pred.squeeze(), size=(1000, len(pred))).astype(np.float32)
	cons = np.maximum(A_csr.dot(x.T) - b[:, None], 0).sum(0)
	idx = np.where(cons == 0)[0]
	if len(idx) > 0:
		sorted_idx = np.argsort(x[idx] @ c)

		# Scan sorted_idx and keep the distinct rows.
		unique_idx = []
		seen = set()
		for i in sorted_idx:
			# Convert the row to a tuple so it can be hashed.
			row = tuple(x[idx][i].tolist())
			if row not in seen:
				seen.add(row)
				unique_idx.append(i)
			if len(unique_idx) >= num_start:
				break

		# Take the selected num_start rows.
		best_x = x[idx][unique_idx]
	else:
		best_x = np.argmin((x @ c).squeeze() + mu * cons)
		best_x = x[best_x].reshape(1, -1)

	return best_x, delta_abs, v_map


def run_solver(test_ins_name, test_data_dir, log_dir, json_dir, solver, time_limit, num_start, predictions):
	"""Parallel-pool stage: apply trust region and run the solver."""
	best_x, delta_abs, v_map = predictions[test_ins_name]
	if num_start == 1:
		modifier = partial(add_trust_region, best_x=best_x, delta=delta_abs, v_map=v_map)
	else:
		modifier = partial(add_multi_trust_region, best_x=best_x, delta=delta_abs, v_map=v_map)
	solve_problem(test_ins_name=test_ins_name,
				  test_data_dir=test_data_dir,
				  log_dir=log_dir,
				  json_dir=json_dir,
				  solver=solver,
				  time_limit=time_limit,
				  model_modifier=modifier,
				  )


@hydra.main(version_base=None, config_path="config", config_name="test")
def test(config: DictConfig):
	# Initialize settings
	set_seed(config.seed)
	set_cpu_num(config.num_workers + 1)
	tb_writter.set_logger(config.paths.tensorboard_dir)

	# Create output directories
	test_dir = config.paths.test_dir
	log_dir = os.path.join(test_dir, "logs")
	json_dir = os.path.join(test_dir, "jsons")
	for directory in [test_dir, log_dir, json_dir]:
		os.makedirs(directory, exist_ok=True)

	# Load and prepare model
	model_path = os.path.join(config.model_dir, "models", "model.pth")

	
	model_name = config.model.get('name', 'gnn')

	if model_name == 'gat':
		logging.info("Initializing GATPredictor model for testing.")
		model = GATPredictor(config.model)
	elif model_name == 'gnn':
		logging.info("Initializing GNNPredictor model for testing.")
		model = GNNPredictor(config.model)
	else:
		raise ValueError(f"Unknown model name specified in config: {model_name}")
 
	model.load_state_dict(torch.load(model_path, map_location='cuda:0'), strict=False)
	model.eval()
	model = model.cuda()

	# Get test files
	files = os.listdir(config.paths.test_data_dir)
	total_files = len(files)

	# Stage 1: serial prediction on GPU (main process only, no GPU contention)
	logging.info(f"Stage 1: running predictions for {total_files} instances on GPU.")
	predictions = {}
	for ins in tqdm(files, desc="Predicting"):
		predictions[ins] = predict(
			test_ins_name=ins,
			model=model,
			test_data_dir=config.paths.test_data_dir,
			mu=config.mu,
			delta=config.model.delta,
			num_start=config.num_start,
		)

	model = model.cpu()
	torch.cuda.empty_cache()

	# Stage 2: parallel solver on CPU
	logging.info(f"Stage 2: parallel solving for {total_files} instances using {config.num_workers} workers.")
	solve_func = partial(run_solver,
						 test_data_dir=config.paths.test_data_dir,
						 log_dir=log_dir,
						 json_dir=json_dir,
						 solver=config.solver,
						 time_limit=config.time_limit,
						 num_start=config.num_start,
						 predictions=predictions)

	with mp.Pool(config.num_workers) as pool:
		progress_iterator = tqdm(pool.imap(solve_func, files),
		                         total=total_files,
		                         desc="Solving")
		for i, _ in enumerate(progress_iterator):
			logging.info(f"Progress: [{i + 1}/{total_files}] - Completed solving for {files[i]}")

	logging.info("All solving tasks completed.")


if __name__ == "__main__":
	test()
