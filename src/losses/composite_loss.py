import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.optical_flow import raft_small

class VGGLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features.to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
            
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg[x])
            
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def forward(self, input, target):
        # input, target: (B, C, H, W) in [-1, 1]
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)

        input = (input + 1) / 2
        target = (target + 1) / 2
        
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        h = self.slice1(input)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        
        h_target = self.slice1(target)
        h_target_relu1_2 = h_target
        h_target = self.slice2(h_target)
        h_target_relu2_2 = h_target
        h_target = self.slice3(h_target)
        h_target_relu3_3 = h_target
        h_target = self.slice4(h_target)
        h_target_relu4_3 = h_target
        
        loss = F.mse_loss(h_relu1_2, h_target_relu1_2) + \
               F.mse_loss(h_relu2_2, h_target_relu2_2) + \
               F.mse_loss(h_relu3_3, h_target_relu3_3) + \
               F.mse_loss(h_relu4_3, h_target_relu4_3)
        return loss

class CompositeLoss(nn.Module):
    def __init__(self, lambda_rec=1.0, lambda_flow=1.0, lambda_perceptual=0.1, device='cuda'):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_flow = lambda_flow
        self.lambda_perceptual = lambda_perceptual
        self.device = device
        
        # Optical Flow Model (Frozen)
        # We use RAFT for high quality flow estimation
        self.flow_model = raft_small(pretrained=True, progress=False).to(device)
        self.flow_model.eval()
        for param in self.flow_model.parameters():
            param.requires_grad = False
            
        # Perceptual Loss (LPIPS)
        # Assuming lpips library is installed, otherwise use a simple VGG feature loss
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net='vgg').to(device)
            self.lpips_fn.eval()
        except ImportError:
            print("Warning: LPIPS not found, using simple VGG loss placeholder.")
            self.lpips_fn = VGGLoss(device=device)

    def calculate_flow(self, video):
        # video: (B, C, T, H, W)
        # RAFT expects (B, C, H, W) pairs
        # We calculate flow between consecutive frames
        b, c, t, h, w = video.shape
        
        # Reshape to process all pairs in batch
        img1 = video[:, :, :-1].reshape(-1, c, h, w)
        img2 = video[:, :, 1:].reshape(-1, c, h, w)
        
        if c == 1:
            img1 = img1.repeat(1, 3, 1, 1)
            img2 = img2.repeat(1, 3, 1, 1)
        
        # Normalize to [-1, 1] if not already
        # RAFT expects input in [-1, 1] range roughly, usually 0-255 mapped.
        # Assuming video is normalized.
        
        # Upsample if too small for RAFT (needs at least 128x128)
        if h < 128 or w < 128:
            img1 = F.interpolate(img1, size=(max(h, 128), max(w, 128)), mode='bilinear', align_corners=False)
            img2 = F.interpolate(img2, size=(max(h, 128), max(w, 128)), mode='bilinear', align_corners=False)

        output = self.flow_model(img1, img2)
        if isinstance(output, list):
            predicted_flow = output[-1] # Use the final refinement
        else:
            print("Unexpected RAFT output format.")
            predicted_flow = output
        
        # Reshape back to (B, T-1, 2, H, W)
        h_flow, w_flow = predicted_flow.shape[-2:]
        predicted_flow = predicted_flow.view(b, t-1, 2, h_flow, w_flow)
        return predicted_flow

    def forward(self, target_video, pred_video):
        """
        target_video: (B, C, T, H, W) - Original Clean Video
        pred_video: (B, C, T, H, W) - Reconstructed Video (x0 prediction from Diffusion)
        """
        losses = {}
        
        # 1. Reconstruction Loss (MSE)
        rec_loss = F.mse_loss(pred_video, target_video)
        losses['rec_loss'] = rec_loss
        
        # 2. Optical Flow Consistency Loss
        # Calculate flow for both target and pred
        # Note: This is expensive. In practice, might compute on a subset of frames or fewer iterations.
        if self.lambda_flow > 0:
            with torch.no_grad():
                target_flow = self.calculate_flow(target_video)
            
            # We want gradients to flow through pred_video's flow calculation?
            # Yes, we want pred_video to have similar flow dynamics.
            # However, differentiating through RAFT is heavy.
            # Alternative: Use a simpler differentiable flow or just pixel difference of frames.
            # But user requested "Optical Flow Consistency".
            # Let's try differentiating through RAFT (it is differentiable).
            pred_flow = self.calculate_flow(pred_video)
            
            flow_loss = F.l1_loss(pred_flow, target_flow)
            losses['flow_loss'] = flow_loss
        else:
            losses['flow_loss'] = torch.tensor(0.0, device=self.device)

        # 3. Perceptual Loss
        if self.lambda_perceptual > 0 and self.lpips_fn is not None:
            # LPIPS expects (B, C, H, W)
            # Flatten time dimension
            b, c, t, h, w = target_video.shape
            t_flat = target_video.view(-1, c, h, w)
            p_flat = pred_video.view(-1, c, h, w)
            
            if c == 1:
                t_flat = t_flat.repeat(1, 3, 1, 1)
                p_flat = p_flat.repeat(1, 3, 1, 1)
            
            # Downsample if too large to save memory
            if h > 128:
                t_flat = F.interpolate(t_flat, size=(128, 128), mode='bilinear')
                p_flat = F.interpolate(p_flat, size=(128, 128), mode='bilinear')
                
            perceptual_loss = self.lpips_fn(p_flat, t_flat).mean()
            losses['perceptual_loss'] = perceptual_loss
        else:
            losses['perceptual_loss'] = torch.tensor(0.0, device=self.device)
            
        # Total Loss
        total_loss = (self.lambda_rec * losses['rec_loss'] + 
                      self.lambda_flow * losses['flow_loss'] + 
                      self.lambda_perceptual * losses['perceptual_loss'])
                      
        losses['total_loss'] = total_loss
        return total_loss, losses
