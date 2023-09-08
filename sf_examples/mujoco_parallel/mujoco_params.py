def mujoco_override_defaults(env, parser):
    env_configs = dict(
        mujoco_ant=dict(
            encoder_mlp_layers=[256, 256, 256, 8],
            save_every_sec=15,
        ),
        mujoco_humanoid=dict(
            encoder_mlp_layers=[256, 256, 256, 17],
        ),
        mujoco_halfcheetah=dict(
            encoder_mlp_layers=[256, 256, 256, 6],
        ),
        mujoco_walker2d=dict(
            encoder_mlp_layers=[256, 256, 256, 6],
        ),
    )

    parser.set_defaults(
        batched_sampling=True,
        num_workers=1,
        num_envs_per_worker=1,
        worker_num_splits=1,
        train_for_env_steps=100_000_000,
        encoder_mlp_layers=env_configs[env]["encoder_mlp_layers"],
        env_frameskip=1,
        nonlinearity="tanh",
        batch_size=2048,
        kl_loss_coeff=0.1,
        use_rnn=False,
        adaptive_stddev=False,
        policy_initialization="torch_default",
        reward_scale=1,
        rollout=64,
        max_grad_norm=3.5,
        num_epochs=2,
        num_batches_per_epoch=4,
        ppo_clip_ratio=0.2,
        value_loss_coeff=1.3,
        exploration_loss_coeff=0.0,
        learning_rate=0.00295,
        lr_schedule="linear_decay",
        shuffle_minibatches=False,
        gamma=0.99,
        gae_lambda=0.95,
        with_vtrace=False,
        recurrence=1,
        normalize_input=True,
        normalize_returns=True,
        value_bootstrap=True,
        experiment_summaries_interval=3,
        save_every_sec=15,
        serial_mode=True,
        async_rl=False,
        meta_batch_size=8,
        wandb_project = "hyperppo",
        save_milestones_sec = 1800,
        eval_every_steps = 0,
        train_for_seconds = 10800,
    )


def add_mujoco_env_args(env, parser):
    # in case we need to add more args in the future
    p = parser
    p.add_argument(
        "--env_agents",
        default=128,
        type=int,
        help="Num. agents in a vectorized env",
    )
