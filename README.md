## Flow Matcing Policies for Offline RL. Skoltech Final Project Reinforcement Learning

Docker: https://hub.docker.com/r/artbash/rl_image

Then run:
```
conda activate rl
git clone https://github.com/SKholkin/Diffusion-Policies-for-Offline-RL
cd Diffusion-Policies-for-Offline-RL
```

How to run our experiments:

For Flow Matching
```
python main.py --env_name <env_name> --device <gpu_name> --ms online --lr_decay --is_flow_matching
```

For Diffusion model
```
python main.py --env_name <env_name> --device <gpu_name> --ms online --lr_decay --no-is_flow_matching
```

Computational resources:

Requires a bit of GPU resources (up to 2GB VRAM)
Training length is about 5-10 hours for different environments on NVIDIA A40 and Intel Xeon 16 cores

Notes:
gym environments are tested with our docker environments while AntMaze, Adroit, Kitchen haven't been tested and you can have problems launching them

Safe environments:
* halfcheetah-medium-v2
* hopper-medium-v2
* walker2d-medium-v2
* halfcheetah-medium-replay-v2
* hopper-medium-replay-v2
* walker2d-medium-replay-v2
* halfcheetah-medium-expert-v2
* hopper-medium-expert-v2
* walker2d-medium-expert-v2

References:

[1] Wang, Z., Hunt, J.J., & Zhou, M. (2022). Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning. ArXiv, abs/2208.06193.

[2] Janner, M., Du, Y., Tenenbaum, J.B., & Levine, S. (2022). Planning with Diffusion for Flexible Behavior Synthesis. International Conference on Machine Learning.

[3] Song, Y., Sohl-Dickstein, J.N., Kingma, D.P., Kumar, A., Ermon, S., & Poole, B. (2020). Score-Based Generative Modeling through Stochastic Differential Equations. ArXiv, abs/2011.13456.

[4] Lipman, Y., Chen, R.T., Ben-Hamu, H., Nickel, M., & Le, M. (2022). Flow Matching for Generative Modeling. ArXiv, abs/2210.02747.



## Diffusion Policies for Offline RL &mdash; Official PyTorch Implementation

**Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning**<br>
Zhendong Wang, Jonathan J Hunt and Mingyuan Zhou <br>
https://arxiv.org/abs/2208.06193 <br>

Abstract: *Offline reinforcement learning (RL), which aims to learn an optimal policy using a previously collected static dataset,
is an important paradigm of RL. Standard RL methods often perform poorly at this task due to the function approximation errors on
out-of-distribution actions. While a variety of regularization methods have been proposed to mitigate this issue, they are often
constrained by policy classes with limited expressiveness that can lead to highly suboptimal solutions. In this paper, we propose
representing the policy as a diffusion model, a recent class of highly-expressive deep generative models. We introduce Diffusion
Q-learning (Diffusion-QL) that utilizes a conditional diffusion model for behavior cloning and policy regularization. 
In our approach, we learn an action-value function and we add a term maximizing action-values into the training loss of the conditional diffusion model,
which results in a loss that seeks optimal actions that are near the behavior policy. We show the expressiveness of the diffusion model-based policy,
and the coupling of the behavior cloning and policy improvement under the diffusion model both contribute to the outstanding performance of Diffusion-QL.
We illustrate the superiority of our method compared to prior works in a simple 2D bandit example with a multimodal behavior policy.
We further show that our method can achieve state-of-the-art performance on the majority of the D4RL benchmark tasks for offline RL.*

## Experiments

### Requirements
Installations of [PyTorch](https://pytorch.org/), [MuJoCo](https://github.com/deepmind/mujoco), and [D4RL](https://github.com/Farama-Foundation/D4RL) are needed. Please see the ``requirements.txt`` for environment set up details.

### Running
Running experiments based our code could be quite easy, so below we use `walker2d-medium-expert-v2` dataset as an example. 

For reproducing the optimal results, we recommend running with 'online model selection' as follows. 
The best_score will be stored in the `best_score_online.txt` file.
```.bash
python main.py --env_name walker2d-medium-expert-v2 --device 0 --ms online --lr_decay
```

For conducting 'offline model selection', run the code below. The best_score will be stored in the `best_score_offline.txt` file.
```.bash
python main.py --env_name walker2d-medium-expert-v2 --device 0 --ms offline --lr_decay --early_stop
```

Hyperparameters for Diffusion-QL have been hard coded in `main.py` for easily reproducing our reported results. 
Definitely, there could exist better hyperparameter settings. Feel free to have your own modifications. 

## Citation

If you find this open source release useful, please cite in your paper:
```
@article{wang2022diffusion,
  title={Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning},
  author={Wang, Zhendong and Hunt, Jonathan J and Zhou, Mingyuan},
  journal={arXiv preprint arXiv:2208.06193},
  year={2022}
}
```

