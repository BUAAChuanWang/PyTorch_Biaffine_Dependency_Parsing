# PyTorch Biaffine Dependency Parsing
A re-implementation of [Deep Biaffine Attention for Neural Dependency Parsing ](https://arxiv.org/abs/1611.01734)based on PyTorch.

# Requirement
	Python  == 3.6  
	PyTorch == 1.0.1
	Cuda == 9.0

# Usage  
	modify the config file, detail see the Config directory
	Train:
	(1) sh run_train_p.sh
	(2) python -u main.py --config ./Config/config.cfg --device cuda:0--train -p 
	    [device: "cpu", "cuda:0", "cuda:1", ......]


# Performance
| Data/score | UAS | LAS |  
| ------------ | ------------ | ------------ |  
| CTB51 | 90.20 | 88.83 |  
| PTB | --- | --- |  


# Reference
- [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734)  
- [https://github.com/tdozat/Parser-v3](https://github.com/tdozat/Parser-v3)  
- [https://github.com/zhangmeishan/BiaffineDParser](https://github.com/zhangmeishan/BiaffineDParser)

# Question #
- if you have any question, you can open a issue or email **mason.zms@gmail.com**、**yunan.hlju@gmail.com**、**bamtercelboo@{gmail.com, 163.com}**.

- if you have any good suggestions, you can PR or email me.