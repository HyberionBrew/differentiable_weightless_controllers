# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from utils.buffers import ReplayBuffer

from wnn_models import WNN
from typing import Optional, Literal
from thermometer import ThermometerGaussian
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ddpg_1411_wnn"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the Atari game"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    
    
    network_type: Literal["wnn", "float", "quant"] = "wnn"
    """Type of network"""
    
    size: int = 256
    """the size of the input for the wnn"""
    bits: int = 63
    """the number of bits for the wnn"""

    n_bit_quantization: Optional[int] = 8
    """Quantization of the core network (all expect input/output)"""
    initial_quantization: Optional[int] = 8
    """Quantization of the floating point inputs, bit-width"""
    last_bit_width: int = 8
    """Last requantiation bit-width"""
    l : int = 2
    enable_quant_step: int = 0
    """When to enable quantization, for DDPG, always enabled at step 0, regardless of the input here"""
    ptq_calibration: bool = True
    """If used, warmup can be performed (use always)"""
    running_normalization: bool = True
    """Running Normalization"""
    
    save_path: str = "models"
    """The directory to save the final model to"""

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias

class WNNActor(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        
        bits = args.bits
        obs_dim = np.array(env.single_observation_space.shape).prod()
        act_dim = np.prod(env.single_action_space.shape)
        thermo = ThermometerGaussian(n_bits=bits, device='cuda')
            
        min_values = torch.ones((obs_dim,)) * -10
        max_values = torch.ones((obs_dim,)) * 10
            
        thermo.fit(torch.zeros((1, obs_dim)), min_value=min_values, max_value=max_values)
        self.fc_mean = WNN(obs_dim=np.array(env.single_observation_space.shape).prod(), 
                act_dim=np.prod(env.single_action_space.shape), 
                sizes=[args.size] * args.l, 
                thermometer=thermo, bits=bits,
                ) 
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )


    def forward(self, x):
        x = self.fc_mean(x)
        x = torch.tanh(x)
        return x * self.action_scale + self.action_bias
    
from brevitas.nn import QuantLinear, QuantReLU, QuantIdentity, QuantTanh
from quantized_models import QuantizedModel
class ActorQuant(nn.Module):
    def __init__(self, env, args, n_bits, initial_quantization):
        super().__init__()
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.n_bits = n_bits
        self.initial_quantization = initial_quantization
        self.fc_mean = QuantizedModel(obs_actor=env.single_observation_space.shape[0], 
                                    act=env.single_action_space.shape[0], hidden=[256, 256],
                                    activation_fn=QuantReLU, 
                                    act_bit_width=n_bits, weight_bit_width=n_bits, initial_quantization=initial_quantization, 
                                    last_bit_width=args.last_bit_width,
                                    thermometer=None, 
                                    squash=False,
                                    ptq_calibration=args.ptq_calibration,)

    def forward(self, x):
        x = self.fc_mean(x)
        x = torch.tanh(x)
        return x * self.action_scale + self.action_bias
    def enable_quant(self):
        self.fc_mean.enable_quant()
        
        
class ObsNorm(nn.Module):
    def __init__(self, shape, device, eps=1e-8):
        super(ObsNorm, self).__init__()
        mean = torch.zeros(shape, device=device)
        var  = torch.ones(shape, device=device)
        count = torch.tensor(eps, device=device)

        # register the buffers
        self.register_buffer("mean", mean)
        self.register_buffer("var", var)
        self.register_buffer("count", count)

    @torch.no_grad()
    def update(self, x):  # x: [B, obs_dim] tensor
        b = torch.as_tensor(x, device=self.mean.device, dtype=torch.float32)
        batch_mean = b.mean(0)
        batch_var  = b.var(0, unbiased=False)
        batch_count = torch.tensor(b.shape[0], device=self.mean.device, dtype=torch.float32)
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot
        self.mean, self.var, self.count = new_mean, M2 / tot, tot

    def norm(self, x):
        x = (x - self.mean) / torch.sqrt(self.var + 1e-8)
        # clamp in -10, 10
        return torch.clamp(x, -10, 10)

@torch.no_grad()
def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float):
    # 1) Soft-update parameters
    src_params = dict(source.named_parameters())
    for name, tgt_p in target.named_parameters():
        src_p = src_params.get(name, None)
        if src_p is None:
            # param missing/mismatch: warn once
            print(f"[WARN] Param '{name}' not found in source. Skipping.")
            continue
        # in-place: tgt = (1 - tau) * tgt + tau * src
        tgt_p.mul_(1.0 - tau).add_(src_p, alpha=tau)

    # 2) Hard-copy buffers (covers BatchNorm running stats & Brevitas quant stats)
    src_bufs = dict(source.named_buffers())
    for name, tgt_b in target.named_buffers():
        src_b = src_bufs.get(name, None)
        if src_b is None:
            print(f"[WARN] Buffer '{name}' not found in source. Skipping.")
            continue
        if tgt_b.shape != src_b.shape or tgt_b.dtype != src_b.dtype:
            print(f"[WARN] Buffer shape/dtype mismatch for '{name}'. "
                  f"tgt={tgt_b.shape},{tgt_b.dtype} src={src_b.shape},{src_b.dtype}. Copying may fail.")
        tgt_b.copy_(src_b)
        
        
def hard_update(target: torch.nn.Module, source: torch.nn.Module):
    # Full copy (use at initialization)
    notac = target.load_state_dict(source.state_dict(), strict=False)  # strict=False to tolerate non-persistent buffers
    # Also copy any non-persistent buffers not in state_dict
    print(notac)
    print(dict(source.named_buffers()))
    print('...')
    with torch.no_grad():
        s_bufs = dict(source.named_buffers())  # includes non-persistent
        for n, tb in target.named_buffers():
            if n in s_bufs:
                tb.copy_(s_bufs[n])
@torch.no_grad()
def sanity_check(source, target, tau):
    # After a soft_update, params should satisfy: tgt ≈ (1-τ)*tgt_prev + τ*src.
    # We can’t access tgt_prev now, so we instead check that source/target have matching keys.
    missing_p = set(dict(source.named_parameters())) ^ set(dict(target.named_parameters()))
    missing_b = set(dict(source.named_buffers())) ^ set(dict(target.named_buffers()))
    print(dict(source.named_buffers()))
    if missing_p:
        print("[CHECK] Param key mismatch:", sorted(missing_p))
    if missing_b:
        print("[CHECK] Buffer key mismatch:", sorted(missing_b))

    # Report max abs diff on buffers (should be small only if you just copied)
    max_buf_diff = 0.0
    for n, sb in source.named_buffers():
        tb = dict(target.named_buffers()).get(n)
        if tb is not None and sb.dtype.is_floating_point and tb.dtype.is_floating_point and sb.shape == tb.shape:
            max_buf_diff = max(max_buf_diff, (tb - sb).abs().max().item())
    print(f"[CHECK] Max abs diff (buffers): {max_buf_diff:.3e}")
    

@torch.no_grad()
def evaluate_actor(
    actor: nn.Module,
    env_id: str,
    normalizer: nn.Module,
    device: torch.device,
    episodes: int = 10,
    seed: int = 123,
    deterministic: bool = True,
    record_stats: bool = True,
    global_step = None,
    writer=None,
):
    eval_env = gym.make(env_id)
    eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env)

    actor.eval()  # important: turn off dropout, etc.
    returns = []
    lengths = []

    for ep in range(episodes):
        obs, _ = eval_env.reset(seed=seed + ep)
        done = False
        truncated = False
        ep_ret = 0.0
        ep_len = 0

        while not (done or truncated):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            if normalizer is not None:
                obs_t = normalizer.norm(obs_t)
            if deterministic:
                
                # use mean action for evaluation
                mean = actor(obs_t)
                action = mean[0].cpu().numpy()
            else:
                action, _, _ = actor(obs_t)
                action = action[0].cpu().numpy()

            obs, r, done, truncated, infos = eval_env.step(action)
            ep_ret += float(r); ep_len += 1
            if done or truncated:
                returns.append(infos["episode"]["r"][0]); lengths.append(infos["episode"]["l"][0])
        #returns.append(ep_ret); lengths.append(ep_len)

    eval_env.close()
    actor.train()  # restore
    
    ret_mean = float(np.mean(returns)); ret_std = float(np.std(returns))
    len_mean = float(np.mean(lengths))

    if record_stats and writer is not None:
        writer.add_scalar("eval/episodic_return_mean", ret_mean, global_step or 0)
        writer.add_scalar("eval/episodic_return_std", ret_std, global_step or 0)
        writer.add_scalar("eval/episodic_length_mean", len_mean, global_step or 0)

    return ret_mean, ret_std, returns, lengths

if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            group=f"{args.bits}_{args.size}_{args.env_id}_{args.exp_name}_{args.network_type}_{args.initial_quantization}_{args.n_bit_quantization}_{args.last_bit_width}",
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    if args.network_type == "wnn":
        actor = WNNActor(envs, args).to(device)
    elif args.network_type=='quant':
        actor = ActorQuant(envs, args, n_bits=args.n_bit_quantization, initial_quantization= args.initial_quantization).to(device)
    elif args.network_type == "float":
        actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    ret_mean, ret_std = 0.0, 0.0
    if args.network_type == "wnn":
        target_actor = WNNActor(envs, args).to(device)
    elif args.network_type == "float":
        target_actor = Actor(envs).to(device)
    elif args.network_type=='quant':
        target_actor = ActorQuant(envs, args, n_bits=args.n_bit_quantization, initial_quantization= args.initial_quantization).to(device)
    print("Using WNN Actor")
    if args.network_type == "quant":
        actor.enable_quant()
        target_actor.enable_quant()
    hard_update(target_actor, actor)  # hard update at the beginning
    sanity_check(actor, target_actor, tau=1.0)
    
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    
    obs_shape = np.array(envs.single_observation_space.shape).prod()
    if args.running_normalization:
        obs_norm = ObsNorm(obs_shape, device=device)
    else:
        obs_norm = None
    

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        
        if args.network_type == "quant":
            if global_step == args.enable_quant_step:
                print('enabling quant')
                actor.enable_quant()
                target_actor.enable_quant()
                
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                # added normalization
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                if args.running_normalization:
                    obs_t = obs_norm.norm(obs_t)
                actions = actor(obs_t)
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
        if args.running_normalization:
            obs_norm.update(next_obs_t)
        
        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            if args.running_normalization:
                o  = obs_norm.norm(data.observations)
                no = obs_norm.norm(data.next_observations)
            else:
                o  = data.observations
                no = data.next_observations
            with torch.no_grad():
                target_actor.eval()
                next_state_actions = target_actor(no)
                qf1_next_target = qf1_target(no, next_state_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)

            qf1_a_values = qf1(o, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(o, actor(o)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                soft_update(target_actor, actor, args.tau)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 10000 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        eval_every = args.total_timesteps // 20
        if global_step % eval_every == 0 and not global_step == args.total_timesteps - 1:
            ret_mean, ret_std, _,_ = evaluate_actor(actor, args.env_id, obs_norm, device=device, writer=writer, global_step=global_step+1)
        # if last timestep , do 1000 eval episodes
        if global_step == args.total_timesteps - 1:
            ret_mean, ret_std, _,_ = evaluate_actor(actor, args.env_id, obs_norm, device=device, episodes=1000, writer=writer, global_step=global_step+1)

    envs.close()
    writer.close()
    
    os.makedirs(args.save_path, exist_ok=True)
    if args.network_type == 'wnn':
        torch.save({
            "fc_mean": actor.fc_mean.state_dict(),
            "obs_norm": obs_norm.state_dict()
        }, f"{args.save_path}/model_ddpg_{args.env_id}_{args.size}_{args.bits}_{args.seed}.pth")
        # write the return mean and std to a txt file
        with open(f"{args.save_path}/model_ddpg_{args.env_id}_{args.size}_{args.bits}_{args.seed}_return.txt", "w") as f:
            f.write(f"{ret_mean},{ret_std}\n")
        print('saved wnn model')
    if args.network_type == 'float':
        torch.save({
            "actor": actor.state_dict(),
            "obs_norm": obs_norm.state_dict()
        }, f"{args.save_path}/model_ddpg_float_{args.env_id}_{args.size}_{args.seed}.pth")
        print('saved float model')
    if args.network_type == 'quant':
        torch.save({
            "actor": actor.state_dict(),
            "obs_norm": obs_norm.state_dict()
        }, f"{args.save_path}/model_ddpg_quant_{args.env_id}_{args.size}_{args.n_bit_quantization}_{args.initial_quantization}_{args.last_bit_width}_{args.seed}.pth")
        print('saved quant model')