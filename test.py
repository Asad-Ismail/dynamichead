#import timm
import torch
from dynamichead import DynamicHead



# Example usage
if __name__ == "__main__":
    # Example Input
    batch_size = 2
    features = {
        'p3': torch.randn(batch_size, 256, 64, 64),
        'p4': torch.randn(batch_size, 512, 32, 32),
        'p5': torch.randn(batch_size, 1024, 16, 16)
    }
    # Channel configuration
    in_channels_dict = {k:v.shape[1] for k,v in features.items()}
    out_channels = 256
    dynamic_head = DynamicHead(in_channels_dict, out_channels)
    
    # Process features
    outputs = dynamic_head(features)
    
    # Print shapes
    print("Input shapes:")
    for k, v in features.items():
        print(f"{k}: {v.shape}")
    
    print("\nOutput shapes:")
    for k, v in outputs.items():
        print(f"{k}: {v.shape}")


#backbone = timm.create_model("resnet50", pretrained=False, features_only=True)
#print(backbone.feature_info.channels())
#sa = ScaleAwareAttention(in_channels=backbone.feature_info.channels()[0:3],
#                         out_channels=backbone.feature_info.channels()[1],
#                         num_levels=3) 
#x = torch.randn(2, 3, 224, 224)
#features = backbone(x)
#output = sa(features[0:3])  # Process last 3 levels
#print(output.shape)  # [2, 256, H, W]