import os
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import unbatch
import hydra
from omegaconf import DictConfig
import logging
from src.utils import custom_collate_fn, set_seed, set_cpu_num, gumbel_sample
from tqdm import tqdm
from tensorboardX import SummaryWriter
import src.tb_writter as tb_writter
from src.model import GraphDataset, GNNPredictor, GATPredictor
import torch.optim as optim
import math
import numpy as np


class Trainer:
	def __init__(self,
	             model: GNNPredictor,
	             train_set: GraphDataset,
	             valid_set: GraphDataset,
	             paths: DictConfig,
	             config: DictConfig,
	             ) -> None:

		self.model = model
		self.batch_size = config.batch_size

		# Configure data loaders
		loader_kwargs = {
			'batch_size': self.batch_size,
			# 'follow_batch': ["constraint_features", "variable_features"],
			'num_workers': 1
		}

		self.train_loader = DataLoader(
			train_set,
			shuffle=True,
			**loader_kwargs
		)

		self.valid_loader = DataLoader(
			valid_set,
			shuffle=False,
			**loader_kwargs
		)

		self.train_loader.collate_fn = custom_collate_fn
		self.valid_loader.collate_fn = custom_collate_fn

		# Configure optimizer
		output_params = {id(p) for layer in (model.vars_output_layer, model.cons_output_layer)
		                 for p in layer.parameters()}
		other_params = [p for p in model.parameters() if id(p) not in output_params]

		params_dict = [
			{'params': model.vars_output_layer.parameters(), 'lr': config.optim.lr_o},
			{'params': other_params, 'lr': config.optim.lr_i}
		]

		optimizer_map = {
			"adam": lambda: optim.Adam(params_dict, weight_decay=config.optim.weight_decay),
			"sgd": lambda: optim.SGD(params_dict, momentum=0.9, weight_decay=1e-4)
		}
		self.optimizer = optimizer_map[config.optim.optimizer]()




		dummy = torch.zeros(1, requires_grad=False)
		self.optimizer_mu = torch.optim.SGD([dummy], lr=config.mu.max)

		# Use CosineAnnealingLR to anneal lr(mu) from mu_max down to mu_min
		self.scheduler_mu = torch.optim.lr_scheduler.CosineAnnealingLR(
			self.optimizer_mu,
			T_max=config.mu.T,
			eta_min=config.mu.min,
		)






		# Configure learning rate scheduler
		scheduler_map = {
			"exp": lambda: optim.lr_scheduler.ExponentialLR(
				self.optimizer, gamma=config.optim.lr.anneal_factor
			),
			"cos": lambda: optim.lr_scheduler.CosineAnnealingLR(
				self.optimizer, T_max=config.optim.lr.cos_T,
				eta_min=config.optim.lr.cos_min
			),
			"cosrestart": lambda: optim.lr_scheduler.CosineAnnealingWarmRestarts(
				self.optimizer, T_0=config.optim.lr.cos_T,
				T_mult=2, eta_min=config.optim.lr.cos_min
			)
		}
		self.lr_scheduler = scheduler_map[config.optim.lr.scheduler]()

		# Store configuration parameters
		self.p_flip = config.p_flip.init

		self.p_flip_step_size = config.p_flip.step_size

		self.mu = config.mu.init
		self.mu_step = config.mu.step
		self.mu_step_size = config.mu.step_size
		self.mu_max = config.mu.max
		self.mu_min = config.mu.min
		self.mu_value = config.mu.value

		self.loss_config = config.loss_config
		self.num_samples = config.num_samples
		self.p_samples = int(config.get('p_samples', 10))
		self.step = 0
		self.epoch = 0
		self.num_epochs = config.num_epochs

		# Setup model save directory
		self.model_save_dir = paths.model_save_dir
		os.makedirs(self.model_save_dir, exist_ok=True)

	def train(self):
		best_valid_best = float('inf')
		model_path = os.path.join(self.model_save_dir, "model.pth")

		for epoch in range(self.num_epochs):
			self.epoch = epoch

			# Run training and validation
			train_metrics = self.run_train_epoch()
			valid_metrics = self.run_valid_epoch()

			# Log metrics
			self._log_metrics("Train", epoch, train_metrics)
			self._log_metrics("Valid", epoch, valid_metrics)

			# Save best model
			if valid_metrics["best"] <= best_valid_best:
				best_valid_best = valid_metrics["best"]
				torch.save(self.model.state_dict(), model_path)
				logging.info(f"Best model saved at epoch {epoch}.")

	def _log_metrics(self, phase, epoch, metrics):
		msg = (
			f"Epoch {epoch} {phase} "
			f"loss: {metrics['loss']:0.3f} "
			f"Obj: {metrics['obj']:0.3f} "
			f"{phase.lower()}_cons: {metrics['cons']:0.3f}"
			+ (f" Best: {metrics['best']:0.3f}" if phase == "Valid" else "")
		)
		if 'sol_entropy' in metrics:
			msg += f" sol_entropy: {metrics['sol_entropy']:0.3f}"
		if 'sol_entropy_norm' in metrics:
			msg += f" sol_entropy_norm: {metrics['sol_entropy_norm']:0.3f}"
		logging.info(msg)

	def run_train_epoch(self):
		self.mu = self.optimizer_mu.param_groups[0]['lr']
		self.model.train()
		data_loader = self.train_loader

		epoch_loss, epoch_obj, epoch_cons = 0, 0, 0
		epoch_sol_entropy, epoch_sol_entropy_norm = 0, 0
		num_samples = 0

		for batch in tqdm(data_loader, desc="Train"):
			batch = batch.cuda()

			# Forward pass
			vars_o, _ = self.model.forward(batch)
			vars_o = vars_o.reshape(-1, 1)

			logits = unbatch(vars_o, batch=batch.variable_features_batch)
			data_list = batch.to_data_list()


			batch_loss = torch.zeros(1, device=vars_o.device)
			batch_obj = torch.zeros(1, device=vars_o.device)
			batch_cons = torch.zeros(1, device=vars_o.device)
			batch_best = 0
			batch_best_obj = 0
			batch_mean_obj = 0
			batch_sol_entropy = torch.zeros(1, device=vars_o.device)
			batch_sol_entropy_norm = torch.zeros(1, device=vars_o.device)

			problem_tensors_list = batch.problem_tensors

			for i, g in enumerate(data_list):
				# unpack the corresponding A, b, c from problem_tensors_list
				A, b, c = problem_tensors_list[i] 
				A, b, c = A.cuda(), b.cuda(), c.cuda()
				p = torch.sigmoid(logits[i]).squeeze()
				# Sample solutions
				x = gumbel_sample(logits[i], self.num_samples, 1.0).float().reshape(self.num_samples, -1)

				s = torch.bernoulli(self.p_flip * torch.ones_like(x)).float()

				entropy = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8)).sum()
				# Solution distribution entropy (analytical): entropy of flipped marginal probability
				p_tilde = (1 - self.p_flip) * p + self.p_flip * (1 - p)
				p_tilde = p_tilde.clamp(1e-7, 1 - 1e-7)
				sol_entropy = -(p_tilde * torch.log(p_tilde) + (1 - p_tilde) * torch.log(1 - p_tilde)).sum()
				batch_sol_entropy += sol_entropy
				batch_sol_entropy_norm += sol_entropy / p_tilde.numel()

				if self.loss_config == "normalize":
					K = self.num_samples
					M = self.p_samples
					tau = 1  # soft-min temperature

					norm_c = torch.norm(c)  # scalar
					A_norm = torch.norm(A, dim=1) + 1e-12  # [m]

					loss_per_k = []

					s_k = torch.bernoulli(torch.full((M, x.size(1)), self.p_flip, device=x.device))  # [M, n]
					# iterate over each first-level sample x[k]
					all_raw_cons = []
					
					all_obj = []
					for k in range(K):
						x_k = x[k]  # [n]
						# second-level sampling: perturb x_k again
						x_k_rep = x_k.unsqueeze(0).expand(M, -1)  # [M, n]
						x2 = (1 - s_k) * x_k_rep + s_k * (1 - x_k_rep)  # [M, n]

						x2 = torch.cat([x_k.reshape(1, -1), x2], dim=0)  # [M+1, n]
						

						# compute obj_{k,u}
						obj = (x2 * c.unsqueeze(0).expand(M + 1, -1)).sum(dim=1)  # [M+1]
						all_obj.append(obj)  # collect all obj

						# compute raw_cons_{k,u}: [m, M]
						raw_cons = torch.relu(A @ x2.T - b)
						all_raw_cons.append(raw_cons)  # collect all raw_cons
						# normalized violation cons_norm_{k,u}: [M]
						num_nonzero = torch.count_nonzero(raw_cons)
						cons_norm = raw_cons.div(A_norm.unsqueeze(1)).sum(dim=0) / num_nonzero if num_nonzero > 0 else 0

						# L_{k,u} for each (k,u)
						L_ku = (obj.div(norm_c) if torch.norm(c) > 0 else 0) + self.mu * cons_norm  # [M]
      
						# soft-min over the u dimension to get L_k
						L_k = -tau * torch.logsumexp(-L_ku / tau, dim=0) # scalar
						loss_per_k.append(L_k)

					# average L_k over all first-level x[k] for the final loss
					loss_softmin = torch.stack(loss_per_k).mean()  # scalar

				elif self.loss_config == "sum":
					loss = obj + self.mu * cons_pos.sum()
				elif self.loss_config == "mean":
					loss = obj + self.mu * cons_pos.mean()
				elif self.loss_config == "nonzero_mean":
					num_nonzero = torch.count_nonzero(cons_pos)
					loss = obj + self.mu * cons_pos.mean() / num_nonzero if num_nonzero > 0 else obj

				# Validation metrics
				with torch.no_grad():
					xx = gumbel_sample(logits[i], 1000, 1.0).float().reshape(1000, -1)
					s = torch.bernoulli(self.p_flip * torch.ones_like(xx)).float()
					xx = (1 - s) * xx + s * (1 - xx)
					idx = torch.where(torch.relu(A @ xx.T - b).sum(0) == 0)[0]
					best = (xx @ c)[idx].min().item() if len(idx) > 0 else float('inf')
					best_obj = (xx @ c).min().item()
					mean_obj = (xx @ c).mean().item()
	 
					s = torch.bernoulli(self.p_flip * torch.ones_like(x)).float()
					x_ = (1 - s) * x + s * (1 - x)  # perturbed sample
					hamming_matrix = (torch.mm(x_, x_.t()) + torch.mm(1 - x_, (1 - x_).t())) / x_.size(1) 
					n = hamming_matrix.shape[0]
					dis = (torch.sum(hamming_matrix) - torch.sum(torch.diag(hamming_matrix))) / (n * (n - 1))
	 
					dis = 1 - dis
	 
					p = (1 - self.p_flip) * p + self.p_flip * (1 - p)  # Adjusted probability based on p_flip
	 
					metric = torch.min(torch.abs(p), torch.abs(1 - p))  # For each element, compute min(|x|, |1 - x|)
					mean_metric = metric.mean()
					std_dev_metric = metric.std()

				batch_obj += torch.stack(all_obj).mean()
				batch_cons += torch.cat(all_raw_cons, dim=1).mean()
				batch_loss += loss_softmin
				batch_best += best
				batch_best_obj += best_obj
				batch_mean_obj += mean_obj



			# Update running totals
			epoch_loss += batch_loss.item()
			epoch_obj += batch_obj.item()
			epoch_cons += batch_cons.item()
			epoch_sol_entropy += batch_sol_entropy.item()
			epoch_sol_entropy_norm += batch_sol_entropy_norm.item()
			num_samples += len(batch)

			# Normalize batch metrics
			batch_loss = batch_loss / len(batch)
			batch_obj = batch_obj / len(batch)
			batch_cons = batch_cons / len(batch)
			batch_best = batch_best / len(batch)
			batch_best_obj = batch_best_obj / len(batch)
			batch_mean_obj = batch_mean_obj / len(batch)
			batch_sol_entropy = batch_sol_entropy / len(batch)
			batch_sol_entropy_norm = batch_sol_entropy_norm / len(batch)

			# Backward pass
			batch_loss.backward()
			self.optimizer.step()
			self.optimizer.zero_grad()

			# Logging
			self.step += 1
			tb_writter.set_step(self.step)
			try:
				tb_writter.add_scalar("Loss/loss", batch_loss.item(), self.step)
				tb_writter.add_scalar("Loss/obj", batch_obj.item(), self.step)
				tb_writter.add_scalar("Loss/cons", batch_cons.item(), self.step)
				tb_writter.add_scalar("Loss/entropy", entropy, self.step)
				tb_writter.add_scalar("Params/mu", self.mu, self.step)
				tb_writter.add_scalar("Params/lr_i", self.lr_scheduler.get_last_lr()[1], self.step)
				tb_writter.add_scalar("Params/lr_o", self.lr_scheduler.get_last_lr()[0], self.step)
				tb_writter.add_scalar("Output/Best", batch_best, self.step)
				tb_writter.add_scalar("Output/Best_obj", batch_best_obj, self.step)
				tb_writter.add_scalar("Output/Mean_obj", batch_mean_obj, self.step)
				tb_writter.add_scalar("Output/sol_entropy", batch_sol_entropy.item(), self.step)
				tb_writter.add_scalar("Output/sol_entropy_norm", batch_sol_entropy_norm.item(), self.step)
				tb_writter.add_scalar("Output/prob", torch.sigmoid(vars_o).mean(), self.step)
				tb_writter.add_scalar("Output/logits", vars_o.mean(), self.step)
				tb_writter.add_scalar("Output/prob_min", torch.sigmoid(vars_o).min(), self.step)
				tb_writter.add_scalar("Output/logits_min", vars_o.min(), self.step)
				tb_writter.add_scalar("Output/prob_max", torch.sigmoid(vars_o).max(), self.step)
				tb_writter.add_scalar("Output/logits_max", vars_o.max(), self.step)
				tb_writter.add_histogram("Prediction/samples", x, self.step)
				tb_writter.add_histogram("Prediction/pred", torch.sigmoid(vars_o), self.step)
				tb_writter.add_histogram("Prediction/logits", vars_o, self.step)
			except Exception:
				pass


		# Update learning rate and mu
		self.lr_scheduler.step()

		self.scheduler_mu.step()


		self.p_flip *= self.p_flip_step_size

		return {
			"loss": epoch_loss / num_samples,
			"obj": epoch_obj / num_samples,
			"cons": epoch_cons / num_samples,
			"sol_entropy": epoch_sol_entropy / num_samples,
			"sol_entropy_norm": epoch_sol_entropy_norm / num_samples,
		}

	def run_train_epoch_self_supervised(self):
		self.mu = self.optimizer_mu.param_groups[0]['lr']
		self.model.train()
		data_loader = self.train_loader

		epoch_loss, epoch_obj, epoch_cons = 0, 0, 0
		epoch_sol_entropy, epoch_sol_entropy_norm = 0, 0
		num_samples = 0

		for batch in tqdm(data_loader, desc="Train"):
			batch = batch.cuda()

			# Forward pass
			vars_o, _ = self.model.forward(batch)
			vars_o = vars_o.reshape(-1, 1)

			logits = unbatch(vars_o, batch=batch.variable_features_batch)
			batch = batch.to_data_list()

			batch_loss = torch.zeros(1, device=vars_o.device)
			batch_obj = torch.zeros(1, device=vars_o.device)
			batch_cons = torch.zeros(1, device=vars_o.device)
			batch_best = 0
			batch_best_obj = 0
			batch_mean_obj = 0
			batch_sol_entropy = torch.zeros(1, device=vars_o.device)
			batch_sol_entropy_norm = torch.zeros(1, device=vars_o.device)

			# Process each graph in batch
			for i, g in enumerate(batch):
				p = torch.sigmoid(logits[i]).squeeze()
				# Sample solutions
				x = gumbel_sample(logits[i], self.num_samples, 1.0).float().reshape(self.num_samples, -1)

				s = torch.bernoulli(self.p_flip * torch.ones_like(x)).float()

				# Get problem matrices
				A = g.A.cuda()
				b = g.b.cuda()
				c = g.c.cuda()

				entropy = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8)).sum()
				# Solution distribution entropy (analytical): entropy of flipped marginal probability
				p_tilde = (1 - self.p_flip) * p + self.p_flip * (1 - p)
				p_tilde = p_tilde.clamp(1e-8, 1 - 1e-8)
				sol_entropy = -(p_tilde * torch.log(p_tilde) + (1 - p_tilde) * torch.log(1 - p_tilde)).sum()
				batch_sol_entropy += sol_entropy
				batch_sol_entropy_norm += sol_entropy / p_tilde.numel()

				if self.loss_config == "normalize" and torch.norm(c) > 0:
					K = self.num_samples
					alpha = 0.2
					logits_i = logits[i]

					x = gumbel_sample(logits_i, K, 1.0).float().reshape(K, -1)
					s = torch.bernoulli(self.p_flip * torch.ones_like(x)).float()
					x_flipped = (1 - s) * x + s * (1 - x)

					x_tot = torch.cat([x, x_flipped], dim=0)

					obj = (x_tot * c.unsqueeze(0).expand(2 * K, -1)).sum(dim=1)
					raw_cons = torch.relu(A @ x_tot.T - b)
					A_norm = torch.norm(A, dim=1) + 1e-12
					cons_norm = raw_cons.div(A_norm.unsqueeze(1)).sum(dim=0) / A_norm.numel()

					norm_c = torch.norm(c)
					loss_k = obj.div(norm_c) + self.mu * cons_norm

					num_elite = max(1, int(alpha * 2 * K))
					elite_idx = torch.topk(-loss_k, num_elite, sorted=False).indices
					x_elite = x_tot[elite_idx].detach()

					logits_repeat = logits_i.reshape(1, -1).expand(num_elite, -1)
					loss_softmin = torch.nn.functional.binary_cross_entropy_with_logits(logits_repeat, x_elite)
				else:
					loss_softmin = batch_loss
					obj = torch.zeros(1, device=vars_o.device)
					raw_cons = torch.zeros(1, device=vars_o.device)

				with torch.no_grad():
					xx = gumbel_sample(logits[i], 1000, 1.0).float().reshape(1000, -1)
					s = torch.bernoulli(self.p_flip * torch.ones_like(xx)).float()
					xx = (1 - s) * xx + s * (1 - xx)
					idx = torch.where(torch.relu(A @ xx.T - b).sum(0) == 0)[0]
					best = (xx @ c)[idx].min().item() if len(idx) > 0 else float('inf')
					best_obj = (xx @ c).min().item()
					mean_obj = (xx @ c).mean().item()

				batch_obj += obj.mean() if hasattr(obj, 'mean') else obj
				batch_cons += 100
				batch_loss += loss_softmin
				batch_best += best
				batch_best_obj += best_obj
				batch_mean_obj += mean_obj

			# Update running totals
			epoch_loss += batch_loss.item()
			epoch_obj += batch_obj.item()
			epoch_cons += batch_cons.item()
			epoch_sol_entropy += batch_sol_entropy.item()
			epoch_sol_entropy_norm += batch_sol_entropy_norm.item()
			num_samples += len(batch)

			# Normalize batch metrics
			batch_loss = batch_loss / len(batch)
			batch_obj = batch_obj / len(batch)
			batch_cons = batch_cons / len(batch)
			batch_best = batch_best / len(batch)
			batch_best_obj = batch_best_obj / len(batch)
			batch_mean_obj = batch_mean_obj / len(batch)
			batch_sol_entropy = batch_sol_entropy / len(batch)
			batch_sol_entropy_norm = batch_sol_entropy_norm / len(batch)

			# Backward pass
			batch_loss.backward()
			self.optimizer.step()
			self.optimizer.zero_grad()

			# Logging
			self.step += 1
			tb_writter.set_step(self.step)
			try:
				tb_writter.add_scalar("Loss/loss", batch_loss.item(), self.step)
				tb_writter.add_scalar("Loss/obj", batch_obj.item(), self.step)
				tb_writter.add_scalar("Loss/cons", batch_cons.item(), self.step)
				tb_writter.add_scalar("Loss/entropy", entropy, self.step)
				tb_writter.add_scalar("Params/mu", self.mu, self.step)
				tb_writter.add_scalar("Params/lr_i", self.lr_scheduler.get_last_lr()[1], self.step)
				tb_writter.add_scalar("Params/lr_o", self.lr_scheduler.get_last_lr()[0], self.step)
				tb_writter.add_scalar("Output/Best", batch_best, self.step)
				tb_writter.add_scalar("Output/Best_obj", batch_best_obj, self.step)
				tb_writter.add_scalar("Output/Mean_obj", batch_mean_obj, self.step)
				tb_writter.add_scalar("Output/sol_entropy", batch_sol_entropy.item(), self.step)
				tb_writter.add_scalar("Output/sol_entropy_norm", batch_sol_entropy_norm.item(), self.step)
				tb_writter.add_scalar("Output/prob", torch.sigmoid(vars_o).mean(), self.step)
				tb_writter.add_scalar("Output/logits", vars_o.mean(), self.step)
				tb_writter.add_scalar("Output/prob_min", torch.sigmoid(vars_o).min(), self.step)
				tb_writter.add_scalar("Output/logits_min", vars_o.min(), self.step)
				tb_writter.add_scalar("Output/prob_max", torch.sigmoid(vars_o).max(), self.step)
				tb_writter.add_scalar("Output/logits_max", vars_o.max(), self.step)
				tb_writter.add_histogram("Prediction/samples", x, self.step)
				tb_writter.add_histogram("Prediction/pred", torch.sigmoid(vars_o), self.step)
				tb_writter.add_histogram("Prediction/logits", vars_o, self.step)
			except Exception:
				pass

		# Update learning rate and mu
		self.lr_scheduler.step()
		self.scheduler_mu.step()
		self.p_flip *= self.p_flip_step_size

		return {
			"loss": epoch_loss / num_samples,
			"obj": epoch_obj / num_samples,
			"cons": epoch_cons / num_samples,
			"sol_entropy": epoch_sol_entropy / num_samples,
			"sol_entropy_norm": epoch_sol_entropy_norm / num_samples,
		}


	@torch.no_grad()
	def run_valid_epoch(self):
		self.model.eval()
		data_loader = self.valid_loader

		epoch_metrics = {
			'loss': 0, 'obj': 0, 'cons': 0, 'best': 0
		}
		num_samples = 0

		for batch in tqdm(data_loader, desc="Valid"):
			batch = batch.cuda()

			# Predict binary distribution
			vars_o, cons_o = self.model.forward(batch)
			vars_o = vars_o.reshape(-1, 1)

			logits = unbatch(vars_o, batch=batch.variable_features_batch)
			data_list = batch.to_data_list()
			batch_metrics = {
				'loss': 0, 'obj': 0, 'cons': 0, 'best': 0,
				'best_obj': 0, 'mean_obj': 0
			}

			problem_tensors_list = batch.problem_tensors

			for i, (g, logit) in enumerate(zip(data_list, logits)):
				# unpack the corresponding A, b, c from problem_tensors_list
				A, b, c = problem_tensors_list[i] 
				A, b, c = A.cuda(), b.cuda(), c.cuda()
				x = gumbel_sample(logit, self.num_samples, 1.0).float().reshape(self.num_samples, -1)

				# Get problem matrices/vectors
				# Calculate metrics
				p = torch.sigmoid(logit).squeeze()
				obj = (p * c).sum().item()
				cons_pos = torch.relu(A @ x.T - b).mean(dim=1, keepdim=True)

				# Calculate loss
				loss = (obj + self.mu * cons_pos.sum()).item()

				# Additional sampling for statistics
				xx = gumbel_sample(logit, 1000, 1.0).float().reshape(1000, -1)
				feasible_idx = torch.where(torch.relu(A @ xx.T - b).sum(0) == 0)[0]
				best = (xx @ c)[feasible_idx].min().item() if len(feasible_idx) > 0 else 1000

				# Update batch metrics
				metrics = {
					'obj': obj,
					'cons': cons_pos.sum().item(),
					'loss': loss,
					'best': best,
					'best_obj': (xx @ c).min().item(),
					'mean_obj': (xx @ c).mean().item()
				}
				for k, v in metrics.items():
					batch_metrics[k] += v

			# Update epoch metrics
			batch_size = len(batch)
			num_samples += batch_size

			for k in ['loss', 'obj', 'cons', 'best']:
				epoch_metrics[k] += batch_metrics[k]

			# Log batch metrics
			normalized_metrics = {k: v / batch_size for k, v in batch_metrics.items()}

		# Log epoch metrics
		for k, v in epoch_metrics.items():
			normalized_v = v / num_samples
			tb_writter.add_scalar(f"Valid/{k}", normalized_v, self.epoch)
			epoch_metrics[k] = normalized_v




		return epoch_metrics


@hydra.main(version_base=None, config_path="config", config_name="train")
def train(config: DictConfig):
	"""
	Train the model.
	"""

	# Initialize settings
	set_seed(config.seed)
	set_cpu_num(config.num_workers + 1)
	torch.cuda.set_device(config.cuda)
	tb_writter.set_logger(config.paths.tensorboard_dir)

	# Get all sample files and split into train/valid
	sample_files = [os.path.join(config.paths.data_samples_dir, f)
	                for f in os.listdir(config.paths.data_samples_dir)]
	split_idx = int(0.80 * len(sample_files))
	train_files, valid_files = sample_files[:split_idx], sample_files[split_idx:]

	# Create datasets
	train_data = GraphDataset(train_files)
	valid_data = GraphDataset(valid_files)

	# Initialize model and move to GPU
	device = torch.device(f'cuda:{config.cuda}')

	model_name = config.model.get('name', 'gnn')

	if model_name == 'gat':
		logging.info(f"Initializing GATPredictor model with {config.model.num_heads} heads.")
		model = GATPredictor(config.model).to(device)
	elif model_name == 'gnn':
		logging.info("Initializing GNNPredictor model.")
		model = GNNPredictor(config.model).to(device)
	else:
		raise ValueError(f"Unknown model name specified in config: {model_name}")


	# Create and run trainer
	trainer = Trainer(
		model=model,
		train_set=train_data,
		valid_set=valid_data,
		paths=config.paths,
		config=config.trainer,
	)

	trainer.train()


if __name__ == "__main__":
	train()
