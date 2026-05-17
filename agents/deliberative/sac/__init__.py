"""SAC components — actor, twin critics, replay buffer.

The actor and critics operate on the LATENT z produced by the world model, not
on raw observations. This is what allows imagined rollouts to be cheap (we
never decode back to obs during training) and is the key architectural
difference vs vanilla SAC on raw state.
"""
