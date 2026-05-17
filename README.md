# Reinforcemnt Learning for motor simulations: Exploring Different World Models
Rowan Evers - University of Nottingham 

## Abstract 

## Introduction 
Reinforcement Learning is an exciting machine learning paradigm, using a goal oriented approach, with rewards reinforcing an agents actions when interacting with an environment. This human like approach was first theorised in the ... with ... This further progressed into ... It applications... 

Reinforcement learning agents can be either be model-free or model-based. The former meaning the agent is purely 'unaware' of the environment it interacts with. It purely makes decisions based on an initial policy which iterated on via rewards. The latter...   

World Models >> link to reactive and deliberative ... 


The main premise of my project will be comparing RL models that use different representations of the environment to learn in the context of continuous action spaces, specifically physics based robotics simulations from the mujoco library.

I will be comparing three agents that use different world models. A reactive agent with no internal world model which will use a direct mapping of state to action via a learned policy, a deliberative agent that will use a continuous (gaussian) world model, and another that uses discrete (categorical) world model. Then I can compare how reactive and deliberative agents perform and how the type of world model the latter uses affects performance.

 It has been shown that discrete representations can have an advantage over continuous ones in certain situations like with the Dreamer models (Hafner et al). My project will further investigate this by comparing how the different agents learn the environment and policy, as well as how they adapt to changes within the environment such as gravity and friction. I will perform a literature review to look at the current solutions for physics based problems along with affects of representation spaces in other environments to give a foundation to work on.

## Related Work 

1. Continuous Control Benchmarks Introduce the benchmarks you're using as the evaluation foundation. Bring in Duan et al. (2016) here — their benchmarking paper is the justification for why MuJoCo tasks are the standard, and which tasks you're using (all but hierarchical).
2. Model-Free RL for Continuous Control One paragraph on the reactive/model-free baseline. Mention TRPO, TNPG, DDPG as strong baselines (from the Duan benchmarking paper) — this positions your reactive agent within the existing landscape.
3. Model-Based RL Cover the general case using the survey (Moerland et al., 2021) and Deisenroth et al. as a concrete example of model-based methods for continuous state/action spaces. The key point is that world models improve sample efficiency — that's the motivation for your deliberative agents.
4. Representation Learning in World Models This is the heart of your contribution. Structure it as: (a) continuous/Gaussian representations — Zhao et al. on Gaussian processes and representation learning for continuous action spaces; (b) discrete representations — DreamerV3 (Hafner et al., 2022) as the key reference, then Scannell's discrete codebook paper as a more recent fringe result worth noting. End with the gap: there is limited systematic comparison of discrete vs. continuous world models specifically on physics-based continuous control benchmarks — that's exactly what your work addresses.
5. Environment Adaptation (brief, if your paper covers it) If you're writing up the gravity/friction adaptation experiments, a short paragraph here on hybrid/adaptive approaches (the Pinosky et al. hybrid model-based + model-free paper) frames why adaptation matters and sets up your adaptation results section.

The Related Work then naturally funnels into your contribution: "In this work we directly compare all three agent types under matched conditions on standard MuJoCo benchmarks, and further evaluate their ability to adapt to physical perturbations..."
The notes in your lit review already tell you roughly what to say about each paper — you're essentially just converting your own annotations into prose. Should take an hour or two at most on writing day.

## Methods 

## Results 

## Discussion 

## Limitations and Future Work 

## Conclusion 