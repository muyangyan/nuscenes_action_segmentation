# Sample-Efficient Learning of Novel Actions in Neuro-symbolic Action Anticipation

## Abstract:
The ability to anticipate the actions of agents is crucial for agent understanding, especially for multi-agent settings such as human-robot interaction and autonomous driving. However, state-of the-art action anticipation models are unable to predict novel actions in a few-shot manner. We propose a novel neuro-symbolic approach that leverages knowledge graphs to segment and anticipate novel actions in a sample-efficient manner. In our approach, a representation of the knowledge graph is used to rectify the base transformer model’s attention mechanism to improve the prediction. To enable continual learning, the knowledge graph expands as new actions are learned, with new entities forming connections with existing ones to explicitly reflect the model’s knowledge. This explicit model allows for greater interpretability through analysis of the activated entities and connections. Using the nuScenes dataset, we empirically show that our approach can predict novel actions such as lane changes learned in a few-shot manner with state-of-the-art accuracy.

## Introduction
The ability to anticipate the actions of agents is crucial for agent understanding, especially for multi-agent settings such as human-robot interaction and autonomous driving.
For example, an autonomous driver may need to anticipate other cars merging to decide one’s speed.
However, many real-world settings are too extensive to model perfectly upon initial training, even when trained on a large corpus of data.
In addition, while current neural-network based models are great at handling high-dimensional data, they suffer from a lack of interpretability.
Therefore, the ability to continually learn novel actions as they arise in an interpretable, sample-efficient manner is paramount.
To this end, we propose a novel neuro-symbolic approach that leverages explicit domain knowledge in the form of knowledge graphs to segment and anticipate novel actions.
In our approach, a representation of the knowledge graph is used to rectify the base transformer model’s attention mechanism by boosting attention to relevant objects in the scene.
The knowledge graph also enhances interpretability by clearly illustrating the significance of various concepts and their interconnections in arriving at the final prediction.
To enable few-shot learning, new actions are first connected with existing nodes in the knowledge graph, providing context that would otherwise be learned through additional data, thereby reducing the number of examples needed to learn an action.
Using the nuScenes dataset, we empirically show that our approach can predict novel actions such as lane changes learned in a few-shot manner with state-of-the-art accuracy.