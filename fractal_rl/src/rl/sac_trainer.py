"""Off-policy SAC training on the FractalEnv, using twin critics and replay."""
import csv
from copy import deepcopy
from pathlib import Path
import random
import warnings
import numpy as np
import torch
from torch.nn import functional as F
from src.env.fractal_env import FractalEnv
from src.rl.replay import ReplayBuffer
from src.rl.sac import SACActor, TwinQ
from src.rl.jepa_regularizers import regularizer
from src.rl.trainer import evaluate
from src.terrain.pink_noise import generate
from src.utils.config import save_config


def update(actor, critics, target, replay, actor_opt, critic_opt, log_alpha, alpha_opt, cfg, device):
    terrain, state, action, reward, next_terrain, next_state, terminated = replay.sample(cfg['sac']['batch_size'], device)
    alpha = log_alpha.exp().detach()
    with torch.no_grad():
        next_action, _, next_logp, _ = actor.act(next_terrain, next_state)
        tq1, tq2 = target(next_terrain, next_state, next_action)
        target_value = reward + float(cfg['sac']['gamma']) * (1 - terminated) * (torch.minimum(tq1, tq2) - alpha * next_logp)
    q1, q2 = critics(terrain, state, action)
    critic_loss = F.mse_loss(q1, target_value) + F.mse_loss(q2, target_value)
    critic_jepa_coef = float(cfg['training'].get('critic_jepa_coef', 0.0))
    if critic_jepa_coef > 0:
        critic_sequence = replay.sample_sequence(cfg['sac']['batch_size'], critics.q1.jepa_horizon, device)
        critic_jepa_loss = (critics.q1.jepa_loss(*critic_sequence, target.q1)
                            + critics.q2.jepa_loss(*critic_sequence, target.q2)) * 0.5
        critic_loss = critic_loss + critic_jepa_coef * critic_jepa_loss
    else:
        critic_jepa_loss = critic_loss.detach().new_zeros(())
    sharing = cfg['training'].get('encoder_sharing', 'separate')
    shared_critic_updates = sharing in ('fully_shared', 'critic_trained')
    # The actor optimizer owns the registered shared backbone.  In both shared
    # modes it steps that backbone from the critic loss; critic_opt owns Q heads.
    if shared_critic_updates:
        actor_opt.zero_grad(set_to_none=True)
    critic_opt.zero_grad(set_to_none=True); critic_loss.backward(); critic_opt.step()
    if shared_critic_updates:
        actor_opt.step()
    critics.set_value_requires_grad(False)
    jepa_coef = float(cfg['training'].get('jepa_coef', 0.0))
    inverse_coef = float(cfg['training'].get('jepa_inverse_coef', 0.0))
    action_margin_coef = float(cfg['training'].get('jepa_action_margin_coef', 0.0))
    action_margin = float(cfg['training'].get('jepa_action_margin', 0.0))
    variance_coef = float(cfg['training'].get('jepa_variance_coef', 0.0))
    actor_encoder_detached = sharing == 'critic_trained'
    embedding = actor.encode(terrain, state) if (actor.jepa_enabled or variance_coef > 0) else None
    if embedding is not None and actor_encoder_detached:
        embedding = embedding.detach()
    new_action, _, logp, _ = (actor.act_from_embedding(embedding)
                              if embedding is not None else actor.act(terrain, state))
    if actor_encoder_detached:
        critics.set_shared_encoder_detach(True)
    q1_pi, q2_pi = critics(terrain, state, new_action)
    if actor_encoder_detached:
        critics.set_shared_encoder_detach(False)
    sac_actor_loss = (alpha * logp - torch.minimum(q1_pi, q2_pi)).mean()
    # Replay stores the actual tanh-squashed action sent to the environment,
    # so JEPA predicts the transition that was actually observed rather than a
    # fresh actor sample.
    if actor.jepa_enabled:
        sequence_terrain, sequence_state, action_sequence, future_terrain, future_state = replay.sample_sequence(
            cfg['sac']['batch_size'], actor.jepa_horizon, device)
        sequence_embedding = actor.encode(sequence_terrain, sequence_state)
        if actor_encoder_detached:
            sequence_embedding = sequence_embedding.detach()
        jepa_loss, prediction, target_embedding = actor.jepa_loss(
            sequence_terrain, sequence_state, action_sequence, future_terrain, future_state,
            embedding=sequence_embedding, return_details=True)
        # A recorded action sequence must predict its own successor better than
        # a shuffled sequence from another transition.  Unlike a residual
        # predictor, this directly penalizes action-blind identity solutions.
        permutation = torch.randperm(len(action_sequence), device=device)
        shuffled_prediction = actor.jepa_predictor(torch.cat([
            sequence_embedding, action_sequence[permutation].flatten(1)], dim=-1))
        real_distance = 1.0 - F.cosine_similarity(prediction, target_embedding.detach(), dim=-1)
        shuffled_distance = 1.0 - F.cosine_similarity(shuffled_prediction, target_embedding.detach(), dim=-1)
        action_margin_loss = torch.relu(action_margin + real_distance - shuffled_distance).mean()
        with torch.no_grad():
            jepa_shuffled = (1.0 - F.cosine_similarity(shuffled_prediction, target_embedding, dim=-1)).mean()
            # Compare two target embeddings, so this baseline measures only
            # temporal predictability rather than EMA-online encoder lag.
            current_target_embedding = actor.target_encode(sequence_terrain, sequence_state)
            jepa_identity = (1.0 - F.cosine_similarity(current_target_embedding, target_embedding, dim=-1)).mean()
        inverse_loss = actor.inverse_loss(sequence_embedding, target_embedding, action_sequence)
    else:
        jepa_loss = sac_actor_loss.new_zeros(())
        inverse_loss = action_margin_loss = jepa_shuffled = jepa_identity = sac_actor_loss.detach().new_zeros(())
    regularizer_name = cfg['training'].get('jepa_regularizer', 'variance')
    regularizer_coef = cfg['training'].get('jepa_regularizer_coef')
    regularizer_coef = variance_coef if regularizer_coef is None else float(regularizer_coef)
    if regularizer_coef > 0:
        embedding_std_by_feature = torch.sqrt(embedding.var(dim=0, unbiased=False) + 1e-4)
        variance_loss = regularizer(regularizer_name, embedding, float(cfg['training'].get('jepa_variance_target', .1)),
                                   int(cfg['training'].get('jepa_regularizer_slices', 64)),
                                   cfg['training'].get('jepa_regularizer_gamma'),
                                   int(cfg['training'].get('jepa_regularizer_quadrature', 32)))
        embedding_std = embedding_std_by_feature.mean().detach()
    else:
        variance_loss = sac_actor_loss.new_zeros(())
        embedding_std = sac_actor_loss.detach().new_zeros(())
    total_actor_loss = (sac_actor_loss + jepa_coef * jepa_loss + inverse_coef * inverse_loss
                        + action_margin_coef * action_margin_loss
                        + regularizer_coef * variance_loss)
    actor_opt.zero_grad(set_to_none=True); total_actor_loss.backward(); actor_opt.step()
    critics.set_value_requires_grad(True)
    # Tune temperature from the updated policy, not the sample used for the
    # preceding actor gradient step.
    with torch.no_grad():
        _, _, updated_logp, _ = actor.act(terrain, state)
    alpha_loss = -(log_alpha * (updated_logp + float(cfg['sac']['target_entropy']))).mean()
    alpha_opt.zero_grad(set_to_none=True); alpha_loss.backward(); alpha_opt.step()
    with torch.no_grad():
        for t, p in zip(target.parameters(), critics.parameters()): t.lerp_(p, float(cfg['sac']['tau']))
    ratio = jepa_loss.detach() / jepa_shuffled.detach().clamp_min(1e-8)
    return (float(sac_actor_loss.item()), float(total_actor_loss.item()),
            float(critic_loss.item()), float(log_alpha.exp().item()),
            float(jepa_loss.item()), float(inverse_loss.item()), float(variance_loss.item()), float(embedding_std.item()),
            float(jepa_coef * jepa_loss.item()), float(regularizer_coef * variance_loss.item()),
            float(inverse_coef * inverse_loss.item()), float(action_margin_coef * action_margin_loss.item()),
            float(jepa_shuffled.item()), float(jepa_identity.item()), float(ratio.item()),
            float(critic_jepa_loss.item()), float(critic_jepa_coef * critic_jepa_loss.item()))


def train_sac(cfg, device='cpu'):
    tc, sc, seed = cfg['training'], cfg['sac'], int(cfg['training']['seed'])
    if int(sc['learning_starts']) < int(sc['batch_size']): raise ValueError('learning_starts must be >= batch_size')
    jepa_coef = float(tc.get('jepa_coef', 0.0))
    if jepa_coef < 0: raise ValueError('training.jepa_coef must be non-negative')
    jepa_inverse_coef = float(tc.get('jepa_inverse_coef', 0.0))
    if jepa_inverse_coef < 0: raise ValueError('training.jepa_inverse_coef must be non-negative')
    jepa_action_margin_coef = float(tc.get('jepa_action_margin_coef', 0.0))
    jepa_action_margin = float(tc.get('jepa_action_margin', 0.0))
    if jepa_action_margin_coef < 0 or jepa_action_margin < 0:
        raise ValueError('JEPA action-margin coefficient and margin must be non-negative')
    sharing = tc.get('encoder_sharing', 'actor_stop_gradient' if tc.get('share_encoder_with_critic', False) else 'separate')
    if sharing not in ('separate', 'actor_stop_gradient', 'fully_shared', 'critic_trained'):
        raise ValueError('training.encoder_sharing must be separate, actor_stop_gradient, fully_shared, or critic_trained')
    shared_encoder_with_critic = sharing != 'separate'
    if shared_encoder_with_critic and jepa_coef <= 0:
        warnings.warn('Sharing the actor encoder with critics without JEPA is an explicit SAC ablation.', stacklevel=2)
    critic_jepa_coef = float(tc.get('critic_jepa_coef', 0.0))
    if critic_jepa_coef < 0:
        raise ValueError('training.critic_jepa_coef must be non-negative')
    if critic_jepa_coef > 0 and shared_encoder_with_critic:
        raise ValueError('critic JEPA and shared stop-gradient critics are mutually exclusive')
    jepa_horizon = int(tc.get('jepa_horizon', 1))
    if jepa_horizon < 1: raise ValueError('training.jepa_horizon must be positive')
    jepa_variance_coef = float(tc.get('jepa_variance_coef', 0.0))
    if jepa_variance_coef < 0: raise ValueError('training.jepa_variance_coef must be non-negative')
    if jepa_coef > 0 and jepa_variance_coef <= 0:
        warnings.warn('SAC+JEPA without a variance floor; use only as an explicit ablation.', stacklevel=2)
    jepa_target_interval = int(tc.get('jepa_target_update_transitions', 64))
    if jepa_target_interval < 1: raise ValueError('training.jepa_target_update_transitions must be positive')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    root = Path(cfg['output']['dump_dir']); (root / 'checkpoints').mkdir(parents=True, exist_ok=True); (root / 'metrics').mkdir(parents=True, exist_ok=True); save_config(cfg, root / 'resolved_config.yaml')
    train_seeds = {int(cfg['terrain']['seed']) + i for i in range(int(cfg['environment']['training_terrain_count']))}
    if train_seeds & set(cfg['environment']['evaluation_terrain_seeds']): raise ValueError('training and evaluation terrain seeds overlap')
    bank = [generate(cfg['terrain'], s) for s in sorted(train_seeds)]; n = int(cfg['environment']['num_parallel_envs']); rng = np.random.default_rng(seed)
    envs = [FractalEnv(cfg, bank[i % len(bank)], seed + i) for i in range(n)]; obs = [env.reset()[0] for env in envs]
    cdd_scales = cfg['terrain'].get('cdd_scales')
    actor = SACActor(cfg['agent']['fov_size'], jepa_enabled=jepa_coef > 0,
                     jepa_horizon=jepa_horizon,
                     jepa_inverse_enabled=jepa_inverse_coef > 0, cdd_scales=cdd_scales).to(device)
    critics = TwinQ(cfg['agent']['fov_size'], shared_base=actor if shared_encoder_with_critic else None,
                    jepa_enabled=critic_jepa_coef > 0, jepa_horizon=jepa_horizon,
                    cdd_scales=cdd_scales, detach_shared=sharing in ('separate', 'actor_stop_gradient')).to(device)
    target = deepcopy(critics).to(device); target.requires_grad_(False)
    init_checkpoint = tc.get('init_checkpoint')
    if init_checkpoint:
        # A spectrum shift is a new MDP. Reuse learned representations and
        # value functions, but intentionally start fresh optimizers/replay.
        # This is transfer fine-tuning, not an exact training resumption.
        source = torch.load(init_checkpoint, map_location=device, weights_only=False)
        if source.get('algorithm') != 'sac':
            raise ValueError('training.init_checkpoint must be an SAC checkpoint')
        actor.load_state_dict(source['model'])
        critics.load_state_dict(source['critics'])
        target.load_state_dict(source.get('target_critics', source['critics']))
        target.requires_grad_(False)
        print(f'Initialized transfer run from {init_checkpoint}', flush=True)
    freeze_actor_encoder = bool(tc.get('freeze_actor_encoder', tc.get('freeze_encoder', False)))
    freeze_critic_encoder = bool(tc.get('freeze_critic_encoder', tc.get('freeze_encoder', False)))
    if shared_encoder_with_critic and freeze_actor_encoder != freeze_critic_encoder:
        raise ValueError('shared encoder cannot freeze actor and critic encoders differently')
    if freeze_actor_encoder:
        for module in (actor.cnn, actor.encoder):
            module.requires_grad_(False)
    if freeze_critic_encoder and not shared_encoder_with_critic:
        for q_network in (critics.q1, critics.q2):
            for module in (q_network.cnn, q_network.encoder):
                module.requires_grad_(False)
    if freeze_actor_encoder or freeze_critic_encoder:
        print(f'Encoder transfer freeze: actor={freeze_actor_encoder}, critic={freeze_critic_encoder}', flush=True)
    actor_opt = torch.optim.Adam((p for p in actor.parameters() if p.requires_grad), lr=float(sc['learning_rate']))
    critic_opt = torch.optim.Adam(critics.value_parameters() if shared_encoder_with_critic else critics.parameters(),
                                  lr=float(sc['learning_rate']))
    log_alpha = torch.tensor(np.log(float(sc['initial_alpha'])), device=device, requires_grad=True); alpha_opt = torch.optim.Adam([log_alpha], lr=float(sc['learning_rate']))
    replay, count, next_eval, next_save, speeds = ReplayBuffer(sc['buffer_size'], obs[0], seed), 0, int(tc['evaluation_interval']), int(tc['checkpoint_interval']), []
    losses = (float('nan'),) * 17
    loss_sum = np.zeros(17, dtype=np.float64)
    loss_count = 0
    transitions_since_jepa_target = 0
    with (root / 'metrics' / 'training.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['steps','rollout_mean_speed','evaluation_mean_speed','evaluation_std_speed','actor_loss','actor_total_loss','critic_loss','alpha','jepa_loss','jepa_inverse_loss','jepa_variance_loss','jepa_embedding_std','jepa_term','inverse_term','variance_term','action_margin_term','jepa_loss_shuffled','jepa_loss_identity','jepa_real_to_shuffled_ratio','critic_jepa_loss','critic_jepa_term']); writer.writeheader()
        while count < int(tc['total_timesteps']):
            active = min(n, int(tc['total_timesteps']) - count)
            if count < int(sc['learning_starts']): actions = rng.uniform(-1, 1, (active, 2)).astype(np.float32)
            else:
                with torch.no_grad(): actions = actor.act(torch.as_tensor(np.stack([o['terrain'] for o in obs[:active]]), device=device), torch.as_tensor(np.stack([o['state'] for o in obs[:active]]), device=device))[0].cpu().numpy()
            for i in range(active):
                previous = obs[i]; next_obs, reward, term, trunc, info = envs[i].step(actions[i]); replay.add(previous, actions[i], reward, next_obs, term, ended=term or trunc); speeds.append(info['speed'])
                if term or trunc: next_obs, _ = envs[i].reset(terrain=bank[int(rng.integers(len(bank)))])
                obs[i] = next_obs
            count += active
            if count >= int(sc['learning_starts']) and replay.size >= int(sc['batch_size']):
                for _ in range(active * int(sc.get('updates_per_step', 1))):
                    losses = update(actor, critics, target, replay, actor_opt, critic_opt, log_alpha, alpha_opt, cfg, device)
                    loss_sum += losses
                    loss_count += 1
                # Decouple the JEPA EMA lag from updates_per_step and batch
                # size. The target moves according to collected environment
                # transitions, not every replay gradient update.
                if actor.jepa_enabled:
                    transitions_since_jepa_target += active
                    while transitions_since_jepa_target >= jepa_target_interval:
                        actor.update_jepa_target(float(tc.get('jepa_target_tau', 0.01)))
                        transitions_since_jepa_target -= jepa_target_interval
            if count >= next_eval or count >= int(tc['total_timesteps']):
                if loss_count:
                    losses = tuple(loss_sum / loss_count)
                mean, std = evaluate(actor, cfg, device); writer.writerow(dict(steps=count, rollout_mean_speed=float(np.mean(speeds)), evaluation_mean_speed=mean, evaluation_std_speed=std, actor_loss=losses[0], actor_total_loss=losses[1], critic_loss=losses[2], alpha=losses[3], jepa_loss=losses[4], jepa_inverse_loss=losses[5], jepa_variance_loss=losses[6], jepa_embedding_std=losses[7], jepa_term=losses[8], variance_term=losses[9], inverse_term=losses[10], action_margin_term=losses[11], jepa_loss_shuffled=losses[12], jepa_loss_identity=losses[13], jepa_real_to_shuffled_ratio=losses[14], critic_jepa_loss=losses[15], critic_jepa_term=losses[16])); handle.flush(); speeds.clear(); next_eval = count + int(tc['evaluation_interval']); print(f'SAC steps={count} eval_speed={mean:.3f} jepa_loss={losses[4]:.4f} critic_jepa={losses[15]:.4f} ratio={losses[14]:.3f}', flush=True)
                loss_sum.fill(0.0)
                loss_count = 0
            if count >= next_save or count >= int(tc['total_timesteps']):
                torch.save({'algorithm':'sac','model':actor.state_dict(),'critics':critics.state_dict(),'target_critics':target.state_dict(),'steps':count,'config':cfg}, root / 'checkpoints' / f'sac_{count:09d}.pt'); next_save = count + int(tc['checkpoint_interval'])
    return actor
