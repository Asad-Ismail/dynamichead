
## Dynamic Head for Object Detection

Dynamic Head: Unifying Object Detection Heads with Attentions  [Paper](https://arxiv.org/pdf/2106.08322)
 
 ### Summary 

It coherently combining multiple self-attention mechanisms between feature levels for scaleawareness, among spatial locations for spatial-awareness, and within output channels for task-awareness, the proposed approach significantly improves the representation ability of object detection heads without any computational
overhead. 
Consider output of a backbone network,ca 3-dimensional tensor with dimensions: level, space, and channel. One solution proposed is to implement a full self-attention mechanism over this tensor which will be very computational expensive O(LXCxS)^2 for full attention main idea is to have sperate self-attention for each dimension thus significantly reducing the computation cost compard to full attention.


### Architecture

<p align="center">
    <img src="imgs/dyanamichead.png" alt="Dynamic head Architecture" width="800" height="200">
</p>

Details of dynamic head pipeline

<p align="center">
    <img src="imgs/dynamichead_pipeline.png" alt="Dynamic head Architecture" width="800" height="200">
</p>

### Equations to Implement

We are implementing this main equation for the paper
<p align="center">
    <img src="imgs/complete_eq.png" alt="Dynamic head Architecture" width="400" height="100">
</p>

where 

**Scale Attention (πL)** :

<p align="center">
    <img src="imgs/scale_eq.png" alt="Dynamic head Architecture" width="300" height="150">
</p>


**Spatial Attention (πS)** :

<p align="center">
    <img src="imgs/scale_eq.png" alt="Dynamic head Architecture" width="300" height="150">
</p>


**Task Attention (πS)** :

<p align="center">
    <img src="imgs/task_eq.png" alt="Dynamic head Architecture" width="300" height="150">
</p>
