---
layout: default
title: Proposal
---

## Summary of the Project

The goal of an agent is to kill as many other agents as possible while staying alive. The input is the map, the 
agent's current position, time elapsed, equipped items, and available items in sight. The agent should be able to 
decide where to move and which action to take according to the map and its current position. The output will be 
the continuous actions taken by agents till the end of the game (only one agent is alive).

***

## AI/ML Algorithms

Reinforcement Learning with Deep Deterministic Policy Gradient

***

## Evaluation Plan

* Quantative Metrics:
Time elapsed -- As time elapse, all agents will receive different punishments inversely proportional to the number 
of kills they have (e.g. the agent with the larger number of kills will receive the fewer punishments)  
Number of kills -- Each kill will bring the agent with a fixed reward  
Ranking -- Agents with higher rank will receive more rewards after each game. The reward of an agent is proportional 
to its rank. Also the ranking will be used to measure the performance of the algorithm when encountering randomly 
moving agents which cooperate and shoot the trained agent  

* Qualitative Metrics:
It will be impressive if this algorithm can defeat five randomly moving agents which cooperate and shoot the trained 
agent in their sights at most of the times.

***

## Appointment with the Instructor

10:30am - 10:45am, Monday, April 29, 2019 @ DBH 4204