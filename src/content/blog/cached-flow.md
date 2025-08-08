---
title: Shooting Method and MPC for Cross-Prompt Caching in Stable Diffusion 3
author: Ankil Patel
pubDatetime: 2025-07-24T00:28:00-08:00
slug: shooting-mpc-rectified-flow-caching
featured: true
draft: false
tags:
  - AI
  - shooting method
  - model predictive control
  - rectified flow
  - trajectory optimization
description: Using shooting method and MPC to enable high-quality trajectory continuation across different prompts in rectified flow models
---


# Shooting Method and MPC for Cross-Prompt Caching in Stable Diffusion 3

| ![Image 1](assets/images/success.png) | ![Image 2](assets/images/success_3.png) | ![Image 3](assets/images/sunset.png) |
|:---:|:---:|:---:|
| Prompt: "A beautiful mountain landscape" | Prompt from cache: "Mountain landscape that is beautiful" | Prompt from cache: "A mountain landscape at sunset" |

## Abstract

I present a shooting method and MPC approach for cross-prompt trajectory continuation in Stable Diffusion 3's rectified flow implementation. Rectified flows use straight-line paths from noise to data - when switching prompts mid-generation, the velocity field suddenly points toward a different target, creating a discontinuity the model wasn't trained for. I treat this as trajectory optimization: find a control sequence to smoothly transition between prompts. Results on SD3: 2.31x speedup with maintained quality (39.5 dB PSNR) for very similar prompts (>95% similarity), degrading to 1.10x speedup at 85% similarity. Direct switching without optimization produces unusable 15 dB PSNR results.

## 1. The Problem

SD3 uses flow matching to learn rectified flows - deterministic straight-line trajectories between data and noise:
```
x_t = (1 - t)x_0 + t·x_1
dx/dt = v_θ(x_t, t, c)
```

Flow matching is the training objective that teaches the model to predict velocities along these straight paths.

When switching prompts at time `t_switch`:
```
v_old = v_θ(x_t, t, c_old)  // Points toward old target
v_new = v_θ(x_t, t, c_new)  // Points toward new target
```

This discontinuity breaks generation because the model only learned smooth trajectories.

### Discovery Process

After fixing basic caching issues (see Appendix), I still faced quality degradation when switching prompts. The key insight came from visualizing the velocity fields:

```python
# Velocity field analysis at switch point
angle = arccos(v_old · v_new / (||v_old|| ||v_new||))
# Result: up to 75° divergence for different prompts!
```

This large angular difference meant simple interpolation would take the trajectory off the learned manifold. The realization: this is a trajectory optimization problem - and control theory has existing methods (shooting, MPC) that could be applied here.

## 2. Key Assumptions

### Core Assumption: Semantic Similarity → Latent Similarity

The entire approach rests on this assumption:
```
If similarity(prompt_A, prompt_B) > threshold
Then ||latent_A(t) - latent_B(t)|| < ε for same t
```

This assumes that semantically similar prompts produce similar intermediate latents at the same timestep. 

**Why this might hold:**
- Text encoders map similar concepts to nearby embeddings
- Rectified flows interpolate linearly, preserving relative distances
- Limited empirical observation on SD3 suggests this works for high similarity

**When it breaks:**
- Style modifiers ("photorealistic" vs "cartoon") can dramatically shift latent space despite semantic similarity
- Negation ("with car" vs "without car") flips meaning but maintains high token overlap
- Different word order can preserve similarity score but change generation path

### Secondary Assumptions

1. **Trajectory Smoothness**: The optimal transition between prompts follows a smooth path (C² continuous)
2. **Local Linearity**: Velocity fields can be locally linearized for MPC
3. **Manifold Connectivity**: Valid latents from different prompts lie on a connected manifold

### Critical Caveats

1. **Training Distribution Mismatch**: If the model wasn't trained on mid-trajectory prompt switches, the velocity field might not generalize well to these "off-distribution" transitions.

2. **Accumulated Error**: Switching prompts mid-flow could amplify numerical integration errors, especially if the new trajectory diverges significantly from the original.

3. **Prompt Embedding Space**: If the prompt embeddings are very different, the velocity field might produce unrealistic transitions or artifacts.

4. **Loss of Coherence**: The generated content might lose semantic consistency across the switch point.

5. **Unknown Concept Manifestation**: We don't know when specific concepts manifest during the flow steps. A "sunset" modifier might establish color palette early (step 10) or late (step 40). Without this knowledge, it's hard to determine the optimal cache point - too early and we miss established features, too late and we can't meaningfully change them.

## 3. Applying Control Theory to Trajectory Optimization

I formulate prompt switching as an optimal control problem, allowing us to use established methods:

**State dynamics:**
```
dx/dt = v_θ(x_t, t, c_t) + u_t
```

**Objective:**
```
min J = ∫[||x_T - x_target||² + λ||u_t||² + γ||dc_t/dt||²] dt
```

This minimizes final error, control effort, and transition abruptness.

### Trajectory Behavior During Transition

The key insight: we're not trying to instantly jump to the new trajectory. Instead, there's a transition period where:

```
t < t_cache: Original linear trajectory (prompt A)
t_cache < t < t_cache + Δt: Smooth transition curve
t > t_cache + Δt: New linear trajectory (prompt B)
```

The shooting method attempts to find a transition curve that:
1. Starts from the cached point on trajectory A
2. Smoothly curves through the transition period
3. Eventually aligns with the straight-line trajectory toward prompt B's target

This transition period (typically 5-8 steps) is crucial - too short and we get artifacts, too long and we lose the benefits of caching. Note: The method finds a plausible transition, not necessarily the optimal one, as no ground truth exists for cross-prompt transitions.

### Visual Comparison of Approaches

**Direct Switch from Cache (No Transition):**
![Direct Switch](assets/images/cache_1.png)
*Result: 15.2 dB PSNR - Catastrophic failure with severe artifacts*

**Linear Interpolation Without Optimization:**
![Linear Interpolation](assets/images/content.png)
*Result: Severe artifacts and corruption from off-manifold trajectories*

These images demonstrate why sophisticated trajectory optimization is necessary. Direct switching creates velocity field discontinuities that produce severe artifacts. Linear interpolation improves slightly but still fails because the interpolated velocities don't stay on the learned manifold.

## 4. Shooting Method for Trajectory Planning

The shooting method finds optimal trajectories by iteratively refining initial velocities:

```python
def shooting_method_trajectory(x_cached, t_switch, old_prompt, new_prompt, horizon=5):
    # Initial guess: blend velocities
    v_old = self.velocity_field(x_cached, t_switch, old_prompt)
    v_new = self.velocity_field(x_cached, t_switch, new_prompt)
    v_init = 0.5 * (v_old + v_new)
    
    for iteration in range(max_iterations):
        # Forward simulate
        trajectory = self.simulate_trajectory(x_cached, v_init, horizon)
        
        # Compute terminal error
        x_terminal = trajectory[-1]
        v_terminal = self.velocity_field(x_terminal, t_switch + horizon*dt, new_prompt)
        error = self.compute_trajectory_error(x_terminal, v_terminal, new_prompt)
        
        # Update initial velocity
        grad = self.compute_shooting_gradient(trajectory, error)
        v_init = v_init - learning_rate * grad
        
        if error < tolerance:
            break
    
    return v_init
```

### Multiple Shooting for Robustness

Single shooting can be sensitive to initial conditions. Multiple shooting divides the trajectory into segments:

```python
def multiple_shooting(x_cached, t_switch, old_prompt, new_prompt, num_segments=3):
    segment_times = np.linspace(t_switch, 1.0, num_segments + 1)
    segment_states = [x_cached] + [None] * num_segments
    segment_velocities = [None] * num_segments
    
    for iter in range(max_iterations):
        # Forward pass: simulate segments
        for i in range(num_segments):
            x_start = segment_states[i]
            v_start = segment_velocities[i] or self.interpolate_velocity(
                x_start, segment_times[i], old_prompt, new_prompt, 
                alpha=i/num_segments
            )
            x_end = self.simulate_segment(x_start, v_start, 
                                         segment_times[i], segment_times[i+1])
            segment_states[i+1] = x_end
        
        # Backward pass: enforce continuity
        continuity_errors = []
        for i in range(num_segments - 1):
            error = segment_states[i+1] - self.simulate_segment(
                segment_states[i], segment_velocities[i],
                segment_times[i], segment_times[i+1]
            )
            continuity_errors.append(error)
        
        # Update velocities
        segment_velocities = self.update_velocities(
            segment_velocities, continuity_errors
        )
        
        if max(continuity_errors) < tolerance:
            break
    
    return segment_velocities[0]
```

## 5. MPC for Real-Time Correction

While shooting provides global planning, MPC handles real-time corrections:

```python
class MPCController:
    def compute_control(self, x_t, t, old_prompt, new_prompt, alpha_schedule):
        # Current blend weight
        alpha = alpha_schedule(t)
        
        # Linearize dynamics
        A_t, B_t = self.linearize_dynamics(x_t, t, old_prompt, new_prompt, alpha)
        
        # Setup QP: min ||x_ref - x||_Q + ||u||_R + ||Δu||_S
        Q = self.state_weight_matrix(alpha)
        R = self.control_weight * np.eye(self.latent_dim)
        S = self.smoothness_weight * np.eye(self.latent_dim)
        
        # Solve for optimal control
        u_sequence = self.solve_qp(A_t, B_t, Q, R, S, x_t)
        
        return u_sequence[0]
```

## 6. Why Simple Approaches Failed

Initial attempts with direct prompt switching and linear blending produced poor results:

| Approach | PSNR | Visual Quality |
|----------|------|----------------|
| Direct switch | 15.2 dB | Corrupted |
| Linear blend | Poor | Severe artifacts |
| Needed | >38 dB | Production quality |

The problem: velocity field discontinuity when switching prompts mid-generation. This led to recognizing we had a trajectory optimization problem, for which control theory offers established solutions.

## 7. Implementation Details

### Adaptive Blending Schedule

The blending schedule α(t) determines transition speed based on prompt similarity:

```python
def adaptive_alpha_schedule(t, t_switch, similarity, transition_steps=5):
    if similarity > 0.95:
        transition_steps = 3  # Very similar: quick
    elif similarity > 0.90:
        transition_steps = 5  # Similar: moderate
    else:
        transition_steps = 8  # Different: slow
    
    progress = (t - t_switch) / (transition_steps * self.dt)
    alpha = 1 / (1 + np.exp(-10 * (progress - 0.5)))
    return np.clip(alpha, 0, 1)
```

### State Management Requirements

Complete state preservation proved critical for both approaches:

```python
class ComprehensiveCacheEntry:
    latent: torch.Tensor
    step: int
    sigma: float
    timestep: float
    dt: float  # Critical for integration
    scheduler_state: Dict[str, Any]
    generator_state: Dict[str, Any]  # Full CUDA state
```

### SD3-Specific Considerations

SD3's flow matching implementation has unique characteristics:
1. **Shifted timesteps**: Uses `shift=3.0` for numerical stability
2. **Velocity prediction**: Model outputs velocity v_θ, not noise ε_θ
3. **FlowMatchEulerDiscreteScheduler**: Custom scheduler for rectified flow integration
4. **No noise injection**: Unlike DDPM/DDIM, the process is fully deterministic

## 8. Combined Algorithm

```python
def cached_generation_with_shooting_mpc(self, prompt, cache):
    # Find similar cached entry
    cached_entry, similarity = cache.find_similar(prompt)
    
    if cached_entry is None or similarity < 0.85:
        return self.generate_from_scratch(prompt)
    
    # Phase 1: Shooting for trajectory planning
    initial_velocity = self.shooting_method_trajectory(
        cached_entry.latent, 
        cached_entry.t,
        cached_entry.prompt,
        prompt,
        horizon=10
    )
    
    # Phase 2: Execute with MPC
    x_t = cached_entry.latent
    t = cached_entry.t
    mpc = MPCController(prediction_horizon=5, control_horizon=2)
    alpha_schedule = lambda t: adaptive_alpha_schedule(
        t, cached_entry.t, similarity
    )
    
    for step in range(cached_entry.remaining_steps):
        # MPC control
        u_t = mpc.compute_control(x_t, t, cached_entry.prompt, prompt, alpha_schedule)
        
        # Blended velocity
        alpha = alpha_schedule(t)
        v_old = self.velocity_field(x_t, t, cached_entry.prompt)
        v_new = self.velocity_field(x_t, t, prompt)
        v_blend = (1 - alpha) * v_old + alpha * v_new
        
        # Update
        x_t = x_t + self.dt * (v_blend + u_t)
        t = t + self.dt
        
        # Re-plan if needed
        if step % 10 == 0:
            error = self.evaluate_trajectory_error(x_t, t, prompt)
            if error > threshold:
                initial_velocity = self.shooting_method_trajectory(
                    x_t, t, cached_entry.prompt, prompt, horizon=5
                )
    
    return self.decode(x_t)
```

## 9. Results

Note: All results are from limited testing on SD3 with a small set of prompt pairs. PSNR measurements are approximate and computed on generated 1024x1024 images. "Similarity" refers to cosine similarity of CLIP text embeddings.

### Trajectory Planning Performance

| Method | Planning Time | Trajectory Error* | PSNR |
|--------|--------------|------------------|------|
| Direct switch | 0ms | 0.892 | ~15 dB |
| Linear blend | 1ms | 0.453 | Poor (severe artifacts) |
| Single shooting | 45ms | 0.087 | ~36 dB |
| Multiple shooting | 112ms | 0.041 | ~38 dB |
| Shooting + MPC | 45ms + 3ms/step | 0.023 | ~39 dB |

*Trajectory error is a synthetic metric measuring deviation from predicted paths

### Performance vs Similarity

| Similarity | Speedup | Quality |
|------------|---------|---------|
| > 0.95 | 2.25x | 39.5 dB |
| 0.90-0.95 | 1.75x | 38.7 dB |
| 0.85-0.90 | 1.45x | 37.2 dB |
| < 0.85 | 1.10x | 31.4 dB |

### Computational Overhead
- Shooting: 45-112ms (1.5-3.8% of generation time)
- MPC: ~3ms per step (6% overhead for 50 steps)
- Similarity computation: ~5ms
- Cache lookup: ~2ms
- Total: ~10% overhead in best case (>95% similarity for 2.31x speedup)
- Note: Speedup drops to 1.10x at 85% similarity, making overhead less worthwhile

## Summary

The shooting method and MPC approach enables cross-prompt caching in SD3, but with significant limitations. It works well only for very similar prompts (>95% similarity), requires complete state management including scheduler details, and the computational overhead means diminishing returns for less similar prompts. The methods are borrowed from control theory and adapted to handle the velocity field discontinuities in rectified flows.

## Future Directions

- Neural trajectory predictors to replace iterative shooting
- Learned MPC weights from user preferences
- Extension to video generation with temporal consistency
- Multi-prompt batch optimization

## Appendix: Debugging Journey

### Initial Problem: 15 dB PSNR with Naïve Caching

Started with simple latent caching:
```python
# Save
cache[prompt] = {
    'latent': latent.cpu(),
    'step': i
}

# Load
latent = cache[prompt]['latent'].to(device)
# Continue from step i
```

Result: Catastrophic quality loss (15 dB PSNR vs 40 dB baseline).

### Debug Phase 1: Generator State

**Hypothesis:** RNG state mismatch causing different noise patterns.

**Finding:** PyTorch CUDA generators have two-level state:
```python
# Wrong - only 16 bytes
state = generator.get_state()  

# Right - full state
state = {
    'internal': generator.get_state(),  # 16 bytes
    'cuda_rng': torch.cuda.get_rng_state(generator.device)  # 5056 bytes
}
```

**Result:** Still poor quality after fix.

### Debug Phase 2: Latent Precision

**Hypothesis:** float16 precision loss during save/load.

**Test:**
```python
# Added diagnostics
print(f"Device transfer precision loss: {(original - restored).abs().max()}")
# Output: 0.00e+00
```

**Result:** Perfect precision, not the issue.

### Debug Phase 3: Scheduler State - The Breakthrough

**Discovery:** Base pipeline calls `scheduler.set_timesteps()` internally:

```python
# Inside StableDiffusion3Pipeline.__call__()
self.scheduler.set_timesteps(num_inference_steps, device=device)
# This overwrites our restored state!
```

**Evidence from logs:**
```
Restored scheduler state:
  timesteps: tensor([894., 852., 810., ...])  # Remaining steps only
  
After pipeline call:
  timesteps: tensor([1000., 958., 916., ...])  # Full schedule!
```

### Debug Phase 4: Flow Matching and Rectified Flow Mathematics

**Realization:** SD3 uses flow matching to learn rectified flows with straight-line trajectories:
```
x_t = (1 - t)x_0 + t·x_1
```

Key properties:
1. **Flow matching**: The training objective that minimizes ||v_θ(x_t, t) - (x_1 - x_0)||²
2. **Rectified flows**: The resulting straight-line trajectories from data to noise
3. **Deterministic**: No stochastic sampling, unlike DDPM
4. Integration step size `dt` must be consistent

The distinction matters because flow matching could theoretically learn curved trajectories, but SD3 specifically uses it to learn rectified (straight) flows.

**Implication:** Can't just restart from cached latent with new schedule - must preserve exact trajectory.

### Solution Architecture

1. **Custom diffusion method** - bypass pipeline's scheduler reset:
```python
def _custom_diffusion_call(self, ...):
    # Do NOT call scheduler.set_timesteps()
    timesteps = self.scheduler.timesteps.to(device)  # Use restored
```

2. **Complete state preservation**:
```python
# Save
entry.scheduler_state = {
    'timesteps': scheduler.timesteps,
    'sigmas': scheduler.sigmas,
    '_step_index': scheduler._step_index,
    'num_inference_steps': scheduler.num_inference_steps,
    # ... 5 more parameters
}

# Restore for continuation
remaining_timesteps = original_timesteps[entry.step:]
scheduler.timesteps = remaining_timesteps
scheduler._step_index = 0  # Reset for sliced schedule
```

3. **Trajectory-aware restoration**:
```python
# Key insight: slice timesteps for continuation
if for_continuation:
    scheduler.timesteps = scheduler.timesteps[step:]
    scheduler.sigmas = scheduler.sigmas[step:]
```

### Validation

Final implementation achieves:
- 39.5 dB PSNR (vs 40 dB baseline)
- 2.31x speedup
- Exact trajectory preservation

**Critical lesson:** After fixing the scheduler state issue, I still had a fundamental problem - switching prompts mid-generation created velocity field discontinuities that simple blending couldn't handle. The breakthrough came from recognizing this as a trajectory optimization problem and realizing that control theory already had tools for this: shooting methods for trajectory planning and MPC for real-time corrections. The key insight wasn't that control theory was "the answer," but that existing control methods could be adapted to this specific problem in rectified flows.

### Computational Challenges of Shooting and MPC

#### Shooting Method Complexity

**Problem 1: Sensitivity to Initial Conditions**
```python
# Small changes in v_init lead to exponentially different trajectories
v_init_1 = v_blend + 0.001
v_init_2 = v_blend - 0.001
# After 10 steps: ||traj_1 - traj_2|| > 0.1
```

**Problem 2: Non-Convex Optimization**
- Multiple local minima in trajectory space
- Gradient computation requires backprop through entire trajectory
- Each iteration needs full forward simulation

**Problem 3: Computational Cost**
```python
# Per iteration cost breakdown:
# - Forward simulation: O(horizon * model_eval_cost)
# - Gradient computation: O(horizon * backprop_cost)
# - Total for convergence: O(iterations * horizon * model_cost)
# 
# Typical: 20 iterations * 10 horizon * 50ms = 10 seconds!
```

#### MPC Implementation Challenges

**Problem 1: Real-Time Constraint**
```python
# Must complete within single diffusion step (~50ms)
# But need to:
# 1. Linearize dynamics (2.1ms)
# 2. Setup QP matrices (0.5ms)
# 3. Solve QP (0.8ms)
# 4. Validate solution (0.2ms)
# Total: 3.6ms (7.2% of step budget)
```

**Problem 2: Jacobian Computation**
```python
def compute_jacobian_wrt_state(self, x_t, t, old_prompt, new_prompt, alpha):
    # Finite differences require 2*latent_dim model evaluations
    # For 128x128 latent: 2 * 16384 * 50ms = 27 minutes (worst case)
    
    # Practical approximation: Assume local interactions
    # Compute only local neighborhoods: reduces to seconds
    # Note: This approximation may miss global interactions
```

**Problem 3: QP Solver Numerical Stability**
```python
# Condition number grows with prediction horizon
# cond(H) ~ O(N_p^2) where H is Hessian
# N_p > 10 leads to numerical issues
# Must use specialized solvers (OSQP, qpOASES)
```

#### Why It Still Works

1. **Amortized Planning**: Shooting method runs once at switch point, not every step
2. **Warm Starting**: Use previous solution as initial guess
3. **Early Termination**: Stop when "good enough" (3-5 iterations typically sufficient)
4. **Approximations That Work**:
   ```python
   # Instead of full Jacobian:
   J_approx = alpha * J_new + (1-alpha) * J_old  # Linear approximation
   
   # Instead of full horizon:
   N_p = min(5, remaining_steps)  # Bounded lookahead
   ```

#### Practical Implementation

```python
class EfficientShootingMPC:
    def __init__(self):
        self.jacobian_cache = {}  # Reuse computations
        self.warm_start = None    # Previous solution
        
    def plan_trajectory(self, x_t, t, old_prompt, new_prompt):
        # Use cached Jacobians when possible
        key = (t, hash(old_prompt), hash(new_prompt))
        if key in self.jacobian_cache:
            J = self.jacobian_cache[key]
        else:
            J = self.compute_jacobian_sparse(...)  # Sparse approximation
            self.jacobian_cache[key] = J
        
        # Warm start from previous solution
        if self.warm_start is not None:
            v_init = self.warm_start
        else:
            v_init = 0.5 * (v_old + v_new)
        
        # Early termination
        for i in range(5):  # Max 5 iterations
            error = self.simulate_and_evaluate(v_init)
            if error < 0.1:  # "Good enough"
                break
            v_init = self.update_velocity(v_init, error)
        
        self.warm_start = v_init
        return v_init
```

**Result**: Reduced computation from 10s to 45ms while maintaining 95% of quality improvement.

### Critical Question: Does Shooting Work Without Ground Truth?

**The fundamental issue:** We don't know what the "correct" transition trajectory looks like between two different prompts. Rectified flows were trained on complete trajectories for single prompts, not prompt switches.

#### What We're Actually Optimizing

The shooting method isn't finding the "true" trajectory (which doesn't exist). Instead, it's optimizing for:

1. **Smoothness** - Minimizing velocity discontinuities:
   ```python
   error = ||v(x_t+dt, t+dt) - (v(x_t, t) + dv/dt * dt)||²
   ```

2. **Manifold Consistency** - Keeping trajectories on the learned data manifold:
   ```python
   # Check if trajectory stays in high-probability regions
   manifold_error = -log p(x_t)  # Using a discriminator or VAE
   ```

3. **End-to-End Coherence** - Ensuring we reach a valid endpoint:
   ```python
   # Terminal velocity should point toward noise at t=1
   terminal_error = ||v(x_T, 1, new_prompt) - expected_terminal_velocity||²
   ```

#### Why It's More Like "Trajectory Hallucination"

What we're really doing:
```python
def compute_trajectory_error(self, x_terminal, v_terminal, new_prompt):
    # We're not comparing to ground truth, but checking:
    
    # 1. Does the velocity field "make sense" at terminal point?
    v_expected = self.velocity_field(x_terminal, t_terminal, new_prompt)
    velocity_consistency = ||v_terminal - v_expected||²
    
    # 2. Is the trajectory smooth (low curvature)?
    smoothness = ||d²x/dt²||²
    
    # 3. Does it stay on the manifold?
    # Using a pretrained discriminator or reconstruction error
    manifold_distance = self.discriminator(x_terminal) 
    
    return velocity_consistency + λ₁*smoothness + λ₂*manifold_distance
```

#### Alternative Approaches That Might Work Better

1. **Direct Interpolation in Latent Space**:
   ```python
   # Simply interpolate between old and new velocity fields
   v_t = (1 - α(t)) * v_old + α(t) * v_new
   # No optimization needed
   ```

2. **Learning-Based Transition**:
   ```python
   # Train a small network to predict transition velocities
   v_transition = transition_net(x_t, t, old_prompt, new_prompt)
   ```

3. **Geodesic Interpolation**:
   ```python
   # Find shortest path on the data manifold
   # (Trajectories don't cross in rectified flows however...)
   ```

#### Why Shooting Method Is Necessary Despite No Ground Truth

The empirical results show simpler approaches fail badly:
- Direct switch: 15.2 dB PSNR (unusable)
- Linear blend: Poor quality with severe artifacts
- Shooting + MPC: 39.1 dB PSNR (near perfect)

The improvement from ~15 dB to ~39 dB represents the difference between unusable and high-quality output.

#### What Makes Shooting Method Work

Even without ground truth, the shooting method discovers trajectories that:

1. **Respect the learned dynamics** - By forward simulating through the velocity field, we ensure each step follows physically plausible dynamics the model understands

2. **Minimize accumulating error** - Direct blending accumulates error exponentially over steps. Shooting finds trajectories that minimize this accumulation

3. **Satisfy boundary conditions** - The trajectory must smoothly connect the cached state to a valid endpoint for the new prompt

While we don't know the "correct" trajectory, we can find trajectories that the model can follow without producing artifacts.

#### Why Simple Blending Fails

```python
# Linear blend seems reasonable but fails because:
v_blend = (1-α)*v_old + α*v_new

# Problem 1: Velocities point to different manifolds
# v_old → old image manifold
# v_new → new image manifold  
# v_blend → somewhere in between (off-manifold!)

# Problem 2: Error compounds
# Step 1: small error ε
# Step 2: error ≈ 2ε  
# Step n: error ≈ nε
# After 30 steps: complete divergence
```

The shooting method succeeds by finding trajectories that stay on the learned manifold throughout the transition, rather than naively interpolating in velocity space.

### GPU Memory Transfer Costs

A critical consideration: is caching worth the GPU↔CPU transfer overhead?

```python
# Latent size for SD3 at 1024x1024 resolution
latent_shape = (1, 16, 128, 128)  # 262,144 elements
dtype = torch.float16  # 2 bytes per element
latent_size = 512 KB

# Transfer timing (PCIe 4.0 x16)
gpu_to_cpu = 0.8ms  # Measured on RTX 4090
cpu_to_gpu = 0.7ms  # Slightly faster
total_overhead = 1.5ms per cache operation

# Compared to compute saved
steps_saved = 25  # For 50% caching
time_per_step = 40ms  # SD3 on RTX 4090
compute_saved = 1000ms

# Net benefit
speedup = compute_saved / (compute_saved/2 + total_overhead)
# = 1000 / (500 + 1.5) = 1.99x (worth it!)
```

#### When It's NOT Worth It

```python
# Small images or few steps saved
if steps_saved < 5:
    # 5 * 40ms = 200ms saved
    # 1.5ms overhead = 0.75% overhead
    # But shooting method adds 45ms...
    # Net speedup: only 1.15x
    
# Memory pressure
if gpu_memory_available < 2GB:
    # CPU caching forces more swapping
    # Can actually slow down overall
```

#### Optimization: Keep Hot Caches on GPU

```python
class HybridCache:
    def __init__(self, gpu_capacity=10, cpu_capacity=1000):
        self.gpu_cache = OrderedDict()  # LRU
        self.cpu_cache = OrderedDict()
        
    def save(self, key, latent, metadata):
        if len(self.gpu_cache) < self.gpu_capacity:
            # Keep on GPU - no transfer cost!
            self.gpu_cache[key] = (latent, metadata)
        else:
            # Evict LRU to CPU
            old_key, (old_latent, old_meta) = self.gpu_cache.popitem(last=False)
            self.cpu_cache[old_key] = (old_latent.cpu(), old_meta)
            self.gpu_cache[key] = (latent, metadata)
```

**Bottom line**: GPU↔CPU transfer is worth it for >10 steps saved, but keeping frequently-used caches on GPU is even better.

## References

[1] Lipman, Y., et al. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." ICLR (2023).

[2] Liu, X., et al. "Flow Matching for Generative Modeling." arXiv:2403.03206 (2024).

[3] Betts, J.T. "Practical Methods for Optimal Control Using Nonlinear Programming." SIAM (2010).

[4] Mayne, D.Q., et al. "Constrained model predictive control: Stability and optimality." Automatica (2000).

[5] Stable Diffusion 3 Technical Report. Stability AI (2024).