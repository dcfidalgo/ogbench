python main.py --env_name=pointmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=0.003
python main.py --env_name=pointmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=0.003
python main.py --env_name=pointmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=0.003 --agent.discount=0.995
python main.py --env_name=pointmaze-teleport-navigate-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=0.003
python main.py --env_name=pointmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.003
python main.py --env_name=pointmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.003
python main.py --env_name=pointmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.003 --agent.discount=0.995
python main.py --env_name=pointmaze-teleport-stitch-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.003
python main.py --env_name=antmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=0.3
python main.py --env_name=antmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=0.3
python main.py --env_name=antmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=0.3 --agent.discount=0.995
python main.py --env_name=antmaze-teleport-navigate-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=0.3
python main.py --env_name=antmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.3
python main.py --env_name=antmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.3
python main.py --env_name=antmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.3 --agent.discount=0.995
python main.py --env_name=antmaze-teleport-stitch-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.3
python main.py --env_name=antmaze-medium-explore-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.01
python main.py --env_name=antmaze-large-explore-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.01
python main.py --env_name=antmaze-teleport-explore-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.actor_p_randomgoal=1.0 --agent.actor_p_trajgoal=0.0 --agent.alpha=0.01
python main.py --env_name=humanoidmaze-medium-navigate-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=0.1 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-navigate-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=0.1 --agent.discount=0.995
python main.py --env_name=humanoidmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=0.1 --agent.discount=0.995
python main.py --env_name=humanoidmaze-medium-stitch-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1 --agent.discount=0.995
python main.py --env_name=humanoidmaze-giant-stitch-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1 --agent.discount=0.995
python main.py --env_name=antsoccer-arena-navigate-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=0.1
python main.py --env_name=antsoccer-medium-navigate-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=0.1
python main.py --env_name=antsoccer-arena-stitch-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1
python main.py --env_name=antsoccer-medium-stitch-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 --agent.alpha=0.1
python main.py --env_name=cube-single-play-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=1.0
python main.py --env_name=cube-double-play-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=1.0
python main.py --env_name=cube-triple-play-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=1.0
python main.py --env_name=cube-quadruple-play-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=1.0
python main.py --env_name=cube-single-noisy-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=0.03
python main.py --env_name=cube-double-noisy-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=0.03
python main.py --env_name=cube-triple-noisy-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=0.03
python main.py --env_name=cube-quadruple-noisy-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=0.03
python main.py --env_name=scene-play-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=1.0
python main.py --env_name=scene-noisy-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=0.03
python main.py --env_name=puzzle-3x3-play-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=1.0
python main.py --env_name=puzzle-4x4-play-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=1.0
python main.py --env_name=puzzle-4x5-play-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=1.0
python main.py --env_name=puzzle-4x6-play-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=1.0
python main.py --env_name=puzzle-3x3-noisy-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=0.03
python main.py --env_name=puzzle-4x4-noisy-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=0.03
python main.py --env_name=puzzle-4x5-noisy-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=0.03
python main.py --env_name=puzzle-4x6-noisy-v0 --eval_episodes=50 --agent=agents/csiq.py --agent.alpha=0.03
