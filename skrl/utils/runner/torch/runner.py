"""skrl 实验运行器：由配置字典/YAML 一键装配模型、智能体与训练器。"""

from __future__ import annotations

from typing import Any, Literal, Type

import copy
import dataclasses
import math  # noqa

from skrl import logger
from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import MultiAgentEnvWrapper, Wrapper
from skrl.models.torch import Model
# 以下带 noqa 的导入（含上面的 math）本身不被本文件直接引用，
# 而是为 _process_cfg 中 eval 字符串（如 "KLAdaptiveLR(...)"）提供名字。
from skrl.resources.noises.torch import GaussianNoise, OrnsteinUhlenbeckNoise  # noqa
from skrl.resources.preprocessors.torch import RunningStandardScaler  # noqa
from skrl.resources.schedulers.torch import KLAdaptiveLR  # noqa
from skrl.trainers.torch import Trainer
from skrl.utils import set_seed


class Runner:
    def __init__(self, env: Wrapper | MultiAgentEnvWrapper, cfg: dict[str, Any], *, verbose: bool = False) -> None:
        """实验运行器。

        用一份配置（dict 或 YAML 加载结果）装配并实例化 skrl 各组件
        （模型、智能体、训练器），几行代码即可启动训练/评估流程。

        :param env: 用于训练的环境（skrl 包装器）。
        :param cfg: 运行器配置。
        :param verbose: 是否打印装配过程的额外信息（如生成的模型源码）。
        """
        self._env = env
        self._verbose = verbose

        # 配置兼容性检查：迁移历史遗留字段名
        self._cfg = self._check_cfg_compatibility(copy.deepcopy(cfg))

        # 设置随机种子（torch/numpy/random 全局生效）
        set_seed(self._cfg.get("seed", None))

        # 每个装配阶段都传入配置的深拷贝，隔离各阶段对配置的修改
        # （_generate_models 会删除 class/separate 等键，不影响后续阶段）
        self._models = self._generate_models(self._env, copy.deepcopy(self._cfg))
        self._agent = self._generate_agent(self._env, copy.deepcopy(self._cfg), self._models)
        self._trainer = self._generate_trainer(self._env, copy.deepcopy(self._cfg), self._agent)

    @property
    def trainer(self) -> Trainer:
        """训练器实例。"""
        return self._trainer

    @property
    def agent(self) -> Agent:
        """智能体实例。"""
        return self._agent

    @staticmethod
    def load_cfg_from_yaml(path: str) -> dict:
        """从 yaml 文件加载运行器配置。

        :param path: 文件路径。

        :return: 加载的配置字典；出错时记录日志并返回空字典。
        """
        try:
            import yaml
        except Exception as e:
            logger.error(f"{e}. Install PyYAML with 'pip install pyyaml'")
            return {}

        try:
            with open(path) as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Loading yaml error: {e}")
            return {}

    def _check_cfg_compatibility(self, cfg: dict) -> dict:
        """检查配置兼容性：把历史遗留字段名迁移到现行写法。

        :param cfg: 待检查的配置字典（原地修改）。

        :return: 迁移后的字典。
        """
        # 'lambda' 改名为 'gae_lambda'（lambda 是 Python 关键字）
        if "lambda" in cfg.get("agent", {}):
            logger.warning("The 'lambda' field in the configuration is deprecated. Use 'gae_lambda' instead")
            cfg["agent"]["gae_lambda"] = cfg["agent"]["lambda"]
            del cfg["agent"]["lambda"]
        # 移除冗余的 'clip_predicted_values'，由 'value_clip' 表达
        # （原开关语义映射为：True 保持 value_clip，False 置 0 关闭裁剪）
        if "clip_predicted_values" in cfg.get("agent", {}):
            logger.warning(
                "The 'clip_predicted_values' field in the configuration is deprecated. "
                "Use a 'value_clip' value greater than 0 to clip the predicted values instead"
            )
            value_clip = cfg["agent"].get("value_clip", 0.2)
            cfg["agent"]["value_clip"] = value_clip if cfg["agent"]["clip_predicted_values"] else 0.0
            del cfg["agent"]["clip_predicted_values"]
        # 未定义 observation_preprocessor 时，用 state_preprocessor 顶替
        # （如需保留 state_preprocessor 而关闭观测预处理，
        #  应显式把 observation_preprocessor 设为 None 阻止自动顶替）
        if "state_preprocessor" in cfg.get("agent", {}):
            if "observation_preprocessor" not in cfg.get("agent", {}):
                logger.warning(
                    "The 'state_preprocessor' field in the configuration has been replaced by 'observation_preprocessor'. "
                    "If the 'state_preprocessor' definition is desired but the `observation_preprocessor` is not, "
                    "define the last one to None (null) to avoid the automatic replacement"
                )
                cfg["agent"]["observation_preprocessor"] = cfg["agent"]["state_preprocessor"]
                cfg["agent"]["observation_preprocessor_kwargs"] = cfg["agent"].get("state_preprocessor_kwargs")
                del cfg["agent"]["state_preprocessor"]
                if "state_preprocessor_kwargs" in cfg["agent"]:
                    del cfg["agent"]["state_preprocessor_kwargs"]
        # 移除 'shared_state_preprocessor'，改用 'state_preprocessor'
        if "shared_state_preprocessor" in cfg.get("agent", {}):
            logger.warning(
                "The 'shared_state_preprocessor' field in the configuration is deprecated. Use 'state_preprocessor' instead"
            )
            cfg["agent"]["state_preprocessor"] = cfg["agent"]["shared_state_preprocessor"]
            cfg["agent"]["state_preprocessor_kwargs"] = cfg["agent"].get("shared_state_preprocessor_kwargs")
            del cfg["agent"]["shared_state_preprocessor"]
            if "shared_state_preprocessor_kwargs" in cfg["agent"]:
                del cfg["agent"]["shared_state_preprocessor_kwargs"]
        return cfg

    def _component(self, name: str) -> Type:
        """按字符串标识符获取 skrl 组件（智能体、记忆、模型实例化器等）。

        :return: 名字对应的组件类/函数；不支持的组件抛 ValueError。
        """
        # 各组件在方法内延迟导入，避免模块级拉起全部算法依赖
        from skrl.agents.torch.a2c import A2C, A2C_CFG
        from skrl.agents.torch.amp import AMP, AMP_CFG
        from skrl.agents.torch.cem import CEM, CEM_CFG
        from skrl.agents.torch.ddpg import DDPG, DDPG_CFG
        from skrl.agents.torch.ddqn import DDQN, DDQN_CFG
        from skrl.agents.torch.dqn import DQN, DQN_CFG
        from skrl.agents.torch.ppo import PPO, PPO_CFG
        from skrl.agents.torch.rpo import RPO, RPO_CFG
        from skrl.agents.torch.sac import SAC, SAC_CFG
        from skrl.agents.torch.td3 import TD3, TD3_CFG
        from skrl.agents.torch.trpo import TRPO, TRPO_CFG
        from skrl.memories.torch import RandomMemory
        from skrl.multi_agents.torch.ippo import IPPO, IPPO_CFG
        from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_CFG
        from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg
        from skrl.utils.model_instantiators.torch import (
            categorical_model,
            deterministic_model,
            gaussian_model,
            multicategorical_model,
            multivariate_gaussian_model,
            shared_model,
        )

        # 名字不区分大小写的注册表：模型实例化器 / 记忆 / 算法及其配置类 / 训练器
        component = {
            # models
            "gaussianmixin": gaussian_model,
            "categoricalmixin": categorical_model,
            "multicategoricalmixin": multicategorical_model,
            "deterministicmixin": deterministic_model,
            "multivariategaussianmixin": multivariate_gaussian_model,
            "shared": shared_model,
            # memories
            "randommemory": RandomMemory,
            # agents
            "a2c": A2C,
            "a2c_cfg": A2C_CFG,
            "amp": AMP,
            "amp_cfg": AMP_CFG,
            "cem": CEM,
            "cem_cfg": CEM_CFG,
            "ddpg": DDPG,
            "ddpg_cfg": DDPG_CFG,
            "ddqn": DDQN,
            "ddqn_cfg": DDQN_CFG,
            "dqn": DQN,
            "dqn_cfg": DQN_CFG,
            "ppo": PPO,
            "ppo_cfg": PPO_CFG,
            "rpo": RPO,
            "rpo_cfg": RPO_CFG,
            "sac": SAC,
            "sac_cfg": SAC_CFG,
            "td3": TD3,
            "td3_cfg": TD3_CFG,
            "trpo": TRPO,
            "trpo_cfg": TRPO_CFG,
            # multi-agents
            "ippo": IPPO,
            "ippo_cfg": IPPO_CFG,
            "mappo": MAPPO,
            "mappo_cfg": MAPPO_CFG,
            # trainers
            "sequentialtrainer": SequentialTrainer,
            "sequentialtrainer_cfg": SequentialTrainerCfg,
        }.get(name.lower())

        if component is None:
            raise ValueError(f"Component '{name}' is not supported in the runner cfg")
        return component

    def _process_cfg(self, cfg: dict) -> dict:
        """把配置里的简单类型转换为 skrl 类/组件。

        转换规则：
        * ``_direct_eval`` 中的键若为字符串则 eval 求值——即允许在
          YAML 里写 ``"KLAdaptiveLR"`` / ``"RunningStandardScaler"`` 等
          类名字符串（依赖文件顶部那些 noqa 导入提供的名字）；
        * 以 ``_kwargs`` 结尾的键，None 统一替换为空字典；
        * 删除 "class" 键（已被 _component 解析为真实组件）；
        * ``exploration_scheduler`` 字符串编译为 lambda(timestep, timesteps)；
        * ``rewards_shaper_scale`` 生成 rewards * scale 的整形函数。

        :param cfg: 配置字典。

        :return: 更新后的字典（深拷贝上操作，不影响传入配置）。
        """
        _direct_eval = [
            "learning_rate_scheduler",
            "observation_preprocessor",
            "state_preprocessor",
            "value_preprocessor",
            "amp_observation_preprocessor",
            "exploration_noise",
            "smooth_regularization_noise",
        ]

        def update_dict(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    update_dict(value)
                else:
                    if key in _direct_eval:
                        if isinstance(value, str):
                            d[key] = eval(value)
                    elif key.endswith("_kwargs"):
                        d[key] = value if value is not None else {}
            return d

        cfg = update_dict(copy.deepcopy(cfg))
        if "class" in cfg:
            del cfg["class"]

        # 物化探索调度器：字符串表达式 -> lambda(timestep, timesteps)
        if "exploration_scheduler" in cfg:
            cfg["exploration_scheduler"] = eval(f"lambda timestep, timesteps: {cfg['exploration_scheduler']}")
        # 物化奖励整形：rewards_shaper_scale -> rewards * scale
        if "rewards_shaper_scale" in cfg:
            scale = cfg["rewards_shaper_scale"]
            if scale is not None and scale != 1.0:
                cfg["rewards_shaper"] = lambda rewards, *args, **kwargs: rewards * scale
            del cfg["rewards_shaper_scale"]

        return cfg

    def _generate_models(self, env: Wrapper | MultiAgentEnvWrapper, cfg: dict[str, Any]) -> dict[str, dict[str, Model]]:
        """按环境规格与配置生成模型实例。

        单智能体环境用占位身份 "agent" 组织，多智能体按
        possible_agents 展开；返回 {agent_id: {role: model}}。
        shared 模式下两个角色共享同一实例（第二个角色是别名）。

        :param env: 包装后的环境。
        :param cfg: 配置字典。

        :return: 模型实例字典。
        """
        multi_agent = isinstance(env, MultiAgentEnvWrapper)
        device = env.device
        # 单智能体：空间字典以占位键 "agent" 组织，与多智能体统一处理
        possible_agents = env.possible_agents if multi_agent else ["agent"]
        observation_spaces = env.observation_spaces if multi_agent else {"agent": env.observation_space}
        state_spaces = env.state_spaces if multi_agent else {"agent": env.state_space}
        action_spaces = env.action_spaces if multi_agent else {"agent": env.action_space}

        agent_class = cfg.get("agent", {}).get("class")
        if not agent_class:
            raise ValueError(f"The 'agent.class' field is not defined in the specified configuration")
        agent_class = agent_class.lower()

        # 实例化模型
        models = {}
        for agent_id in possible_agents:
            _cfg = copy.deepcopy(cfg)
            models[agent_id] = {}
            models_cfg = _cfg.get("models")
            if not models_cfg:
                raise ValueError("The 'models' field is not defined in the specified configuration")
            # 读取 separate（非共享）开关并从配置中移除该键；缺省回退 True
            try:
                separate = models_cfg["separate"]
                del models_cfg["separate"]
            except KeyError:
                separate = True
                logger.warning(
                    "The 'models.separate' field is not defined in the specified configuration. Falling back to True by default"
                )
            # 非共享模型：每个角色（policy/value 等）独立实例化
            if separate:
                for role in models_cfg:
                    # 取模型实例化器并移除 'class' 键
                    model_class = models_cfg[role].get("class")
                    if not model_class:
                        raise ValueError(
                            f"The 'models.{role}.class' field is not defined in the specified configuration"
                        )
                    del models_cfg[role]["class"]
                    model_class = self._component(model_class)
                    # 按算法/角色选择空间：AMP 判别器用专门的 amp 观测空间
                    observation_space = observation_spaces[agent_id]
                    if agent_class == "amp" and role == "discriminator":
                        try:
                            observation_space = env.amp_observation_space
                        except Exception as e:
                            logger.warning(
                                "Unable to get AMP space via 'env.amp_observation_space'. Using 'env.observation_space' instead"
                            )
                    # verbose 时打印生成的模型源码（return_source 不产生实例副作用）
                    if self._verbose:
                        source = model_class(
                            observation_space=observation_space,
                            state_space=state_spaces[agent_id],
                            action_space=action_spaces[agent_id],
                            device=device,
                            **self._process_cfg(models_cfg[role]),
                            return_source=True,
                        )
                        print("==================================================")
                        print(f"Model (role): {role}")
                        print("==================================================\n")
                        print(source)
                        print("--------------------------------------------------")
                    # 实例化模型
                    models[agent_id][role] = model_class(
                        observation_space=observation_space,
                        state_space=state_spaces[agent_id],
                        action_space=action_spaces[agent_id],
                        device=device,
                        **self._process_cfg(models_cfg[role]),
                    )
            # 共享模型：仅支持恰好两个角色（如 policy 与 value 共享主干）
            else:
                roles = list(models_cfg.keys())
                if len(roles) != 2:
                    raise ValueError(
                        "Runner currently only supports shared models, made up of exactly two models. "
                        "Set 'separate' field to True to create non-shared models for the given cfg"
                    )
                # 收集共享模型的结构名与参数
                structure = []
                parameters = []
                for role in roles:
                    # 取模型结构名并移除 'class' 键
                    model_structure = models_cfg[role].get("class")
                    if not model_structure:
                        raise ValueError(
                            f"The 'models.{role}.class' field is not defined in the specified configuration"
                        )
                    del models_cfg[role]["class"]
                    structure.append(model_structure)
                    parameters.append(self._process_cfg(models_cfg[role]))
                model_class = self._component("Shared")
                # verbose 时打印生成的共享模型源码
                if self._verbose:
                    source = model_class(
                        observation_space=observation_spaces[agent_id],
                        state_space=state_spaces[agent_id],
                        action_space=action_spaces[agent_id],
                        device=device,
                        structure=structure,
                        roles=roles,
                        parameters=parameters,
                        return_source=True,
                    )
                    print("==================================================")
                    print(f"Shared model (roles): {roles}")
                    print("==================================================\n")
                    print(source)
                    print("--------------------------------------------------")
                # 实例化共享模型：两个角色指向同一实例（第二个是别名）
                models[agent_id][roles[0]] = model_class(
                    observation_space=observation_spaces[agent_id],
                    state_space=state_spaces[agent_id],
                    action_space=action_spaces[agent_id],
                    device=device,
                    structure=structure,
                    roles=roles,
                    parameters=parameters,
                )
                models[agent_id][roles[1]] = models[agent_id][roles[0]]

        # 初始化惰性模块参数：跑一次前向触发形状推导
        for agent_id in possible_agents:
            for role, model in models[agent_id].items():
                model.init_state_dict(role=role)

        return models

    def _generate_agent(
        self,
        env: Wrapper | MultiAgentEnvWrapper,
        cfg: dict[str, Any],
        models: dict[str, dict[str, Model]],
    ) -> Agent:
        """按环境规格、配置与模型生成智能体实例。

        记忆在装配智能体前按 agent_id 逐个创建；预处理器的 size/device
        在这里注入配置。智能体配置经 asdict 转为 dict 传入（各算法的
        cfg 参数支持 dict 形式，内部再展开为对应 CFG）。

        :param env: 包装后的环境。
        :param cfg: 配置字典。
        :param models: 智能体的模型实例。

        :return: 智能体实例。
        """
        multi_agent = isinstance(env, MultiAgentEnvWrapper)
        device = env.device
        num_envs = env.num_envs
        # 单智能体：空间字典以占位键 "agent" 组织
        possible_agents = env.possible_agents if multi_agent else ["agent"]
        observation_spaces = env.observation_spaces if multi_agent else {"agent": env.observation_space}
        state_spaces = env.state_spaces if multi_agent else {"agent": env.state_space}
        action_spaces = env.action_spaces if multi_agent else {"agent": env.action_space}

        # 取智能体类名
        if "agent" not in cfg:
            raise ValueError(f"The 'agent' field is not defined in the specified configuration")
        if "class" not in cfg["agent"]:
            raise ValueError(f"The 'agent.class' field is not defined in the specified configuration")
        agent_class = cfg["agent"]["class"].lower()

        # 取记忆类并实例化：负 memory_size 表示沿用算法的 rollouts 步数
        if "memory" not in cfg:
            raise ValueError(f"The 'memory' field is not defined in the specified configuration")
        if "class" not in cfg["memory"]:
            raise ValueError(f"The 'memory.class' field is not defined in the specified configuration")
        memory_class = self._component(cfg["memory"]["class"])
        if cfg["memory"]["memory_size"] < 0:
            cfg["memory"]["memory_size"] = cfg["agent"]["rollouts"]  # memory_size 取算法的 rollouts
        memories = {
            agent_id: memory_class(num_envs=num_envs, device=device, **self._process_cfg(cfg["memory"]))
            for agent_id in possible_agents
        }

        # 单智能体 AMP：附加判别器专用空间、动作数据集与回放缓冲
        if agent_class in ["amp"]:
            agent_id = possible_agents[0]
            try:
                amp_observation_space = env.amp_observation_space
            except Exception as e:
                logger.warning(
                    "Unable to get AMP space via 'env.amp_observation_space'. Using 'env.observation_space' instead"
                )
                amp_observation_space = observation_spaces[agent_id]
            agent_cfg = dataclasses.asdict(self._component(f"{agent_class}_CFG")(**self._process_cfg(cfg["agent"])))
            # 注入各预处理器的 size/device（value 输出恒为标量宽度 1）
            agent_cfg.get("observation_preprocessor_kwargs", {}).update(
                {"size": observation_spaces[agent_id], "device": device}
            )
            agent_cfg.get("state_preprocessor_kwargs", {}).update({"size": state_spaces[agent_id], "device": device})
            agent_cfg.get("value_preprocessor_kwargs", {}).update({"size": 1, "device": device})
            agent_cfg.get("amp_observation_preprocessor_kwargs", {}).update(
                {"size": amp_observation_space, "device": device}
            )

            motion_dataset = None
            if "motion_dataset" in cfg:
                if "class" not in cfg["motion_dataset"]:
                    raise ValueError(f"The 'motion_dataset.class' field is not defined in the specified configuration")
                motion_dataset_class = self._component(cfg["motion_dataset"]["class"])
                motion_dataset = motion_dataset_class(device=device, **self._process_cfg(cfg["motion_dataset"]))
            reply_buffer = None
            if "reply_buffer" in cfg:
                if "class" not in cfg["reply_buffer"]:
                    raise ValueError(f"The 'reply_buffer.class' field is not defined in the specified configuration")
                reply_buffer_class = self._component(cfg["reply_buffer"]["class"])
                reply_buffer = reply_buffer_class(device=device, **self._process_cfg(cfg["reply_buffer"]))

            agent_kwargs = {
                "models": models[agent_id],
                "memory": memories[agent_id],
                "observation_space": observation_spaces[agent_id],
                "state_space": state_spaces[agent_id],
                "action_space": action_spaces[agent_id],
                "amp_observation_space": amp_observation_space,
                "motion_dataset": motion_dataset,
                "reply_buffer": reply_buffer,
                "collect_reference_motions": lambda num_samples: env.collect_reference_motions(num_samples),
            }
        # 标准单智能体算法族
        elif agent_class in ["a2c", "cem", "ddpg", "ddqn", "dqn", "ppo", "rpo", "sac", "td3", "trpo"]:
            agent_id = possible_agents[0]
            agent_cfg = dataclasses.asdict(self._component(f"{agent_class}_CFG")(**self._process_cfg(cfg["agent"])))
            # 注入各预处理器的 size/device（value 输出恒为标量宽度 1）
            agent_cfg.get("observation_preprocessor_kwargs", {}).update(
                {"size": observation_spaces[agent_id], "device": device}
            )
            agent_cfg.get("state_preprocessor_kwargs", {}).update({"size": state_spaces[agent_id], "device": device})
            agent_cfg.get("value_preprocessor_kwargs", {}).update({"size": 1, "device": device})
            agent_cfg.get("exploration_noise_kwargs", {}).update({"device": device})
            agent_cfg.get("smooth_regularization_noise_kwargs", {}).update({"device": device})
            agent_kwargs = {
                "models": models[agent_id],
                "memory": memories[agent_id],
                "observation_space": observation_spaces[agent_id],
                "state_space": state_spaces[agent_id],
                "action_space": action_spaces[agent_id],
            }
        # 多智能体（IPPO/MAPPO）：观测/状态预处理器按智能体逐个注入，
        # value 预处理器保持平坦字典（随后的 cfg.expand 会按智能体复制）
        elif agent_class in ["ippo", "mappo"]:
            agent_cfg = dataclasses.asdict(self._component(f"{agent_class}_CFG")(**self._process_cfg(cfg["agent"])))
            agent_cfg.get("observation_preprocessor_kwargs", {}).update(
                {agent_id: {"size": observation_spaces[agent_id], "device": device} for agent_id in possible_agents}
            )
            agent_cfg.get("state_preprocessor_kwargs", {}).update(
                {agent_id: {"size": state_spaces[agent_id], "device": device} for agent_id in possible_agents}
            )
            agent_cfg.get("value_preprocessor_kwargs", {}).update({"size": 1, "device": device})
            agent_kwargs = {
                "models": models,
                "memories": memories,
                "observation_spaces": observation_spaces,
                "state_spaces": state_spaces,
                "action_spaces": action_spaces,
                "possible_agents": possible_agents,
            }
        return self._component(agent_class)(cfg=agent_cfg, device=device, **agent_kwargs)

    def _generate_trainer(self, env: Wrapper | MultiAgentEnvWrapper, cfg: dict[str, Any], agent: Agent) -> Trainer:
        """按环境规格、配置与智能体生成训练器实例。

        :param env: 包装后的环境。
        :param cfg: 配置字典。
        :param agent: 智能体实例。

        :return: 训练器实例。
        """
        # 取训练器类名并实例化
        if "trainer" not in cfg:
            raise ValueError(f"The 'trainer' field is not defined in the specified configuration")
        if "class" not in cfg["trainer"]:
            raise ValueError(f"The 'trainer.class' field is not defined in the specified configuration")
        trainer_class = cfg["trainer"]["class"].lower()
        trainer_cfg = self._component(f"{trainer_class}_CFG")(**self._process_cfg(cfg["trainer"]))
        return self._component(trainer_class)(env=env, agents=agent, cfg=trainer_cfg)

    def run(self, mode: Literal["train", "eval"] = "train") -> None:
        """运行训练/评估。

        :param mode: 运行模式：``"train"`` 训练，``"eval"`` 评估。

        :raises ValueError: 运行模式不合法。
        """
        if mode == "train":
            self._trainer.train()
        elif mode == "eval":
            self._trainer.eval()
        else:
            raise ValueError(f"Unknown running mode: {mode}")
