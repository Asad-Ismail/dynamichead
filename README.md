
## Dynamic Head for Object Detection

Dynamic Head: Unifying Object Detection Heads with Attentions  [Paper](https://arxiv.org/pdf/2106.08322)

This is not official implmenetaito but prpbably easier to read and easier to follow workflow more closely related to workflow described in paper. See our summary of paper here 

[Summary](https://github.com/Asad-Ismail/Paper-Summaries/tree/main/ImageOD)

[Youtube Explaination](https://www.youtube.com/watch?v=LLbJIzAMmCM)

## Installation

```bash
git clone https://github.com/yourusername/dynamichead.git
cd dynamichead
pip install -e .
```

## Usage

```python
## Set up input input is dict of feature level key:tensor
batch_size = 2
features = {
    'p3': torch.randn(batch_size, 256, 64, 64),
    'p4': torch.randn(batch_size, 512, 32, 32),
    'p5': torch.randn(batch_size, 1024, 16, 16)
}
# Channel configuration
in_channels_dict = {k:v.shape[1] for k,v in features.items()}

## We need to have same output channels for all levels. OD mostly have that but to make sure 
out_channels = 256

## Create Dynamic head class
dynamic_head = DynamicHead(in_channels_dict, out_channels)

# Process features outputs is a dict with tensor of same shape as input dict
outputs = dynamic_head(features)
```
