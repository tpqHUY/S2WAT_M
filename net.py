import torch.nn as nn
from model.configuration import TransModule_Config
from model.transformer_components import TransformerDecoderLayer


########################################## VGG & components ##########################################

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


# compute channel-wise means and variances of features
def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert len(size) == 4, 'The shape of feature needs to be a tuple with length 4.'
    B, C = size[:2]
    feat_mean = feat.reshape(B, C, -1).mean(dim=2).reshape(B, C, 1, 1)
    feat_std = (feat.reshape(B, C, -1).var(dim=2) + eps).sqrt().reshape(B, C, 1, 1)
    return feat_mean, feat_std


# normalize features
def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


########################################## Transfer Module ##########################################

class TransModule(nn.Module):
  """The Transfer Module of Style Transfer via Transformer

  Taking Transformer Decoder as the transfer module.

  Args:
    config: The configuration of the transfer module
  """
  def __init__(self, config: TransModule_Config=None):
    super(TransModule, self).__init__()
    self.layers = nn.ModuleList([
      TransformerDecoderLayer(
          d_model=config.d_model,
          nhead=config.nhead,
          mlp_ratio=config.mlp_ratio,
          qkv_bias=config.qkv_bias,
          attn_drop=config.attn_drop,
          drop=config.drop,
          drop_path=config.drop_path,
          act_layer=config.act_layer,
          norm_layer=config.norm_layer,
          norm_first=config.norm_first
          ) \
      for i in range(config.nlayer)
    ])

  def forward(self, content_feature, style_feature):
    """
    Args:
      content_feature: Content features，for producing Q sequences. Similar to tgt sequences in pytorch. (Tensor,[Batch,sequence,dim])
      style_feature : Style features，for producing K,V sequences.Similar to memory sequences in pytorch.(Tensor,[Batch,sequence,dim])

    Returns:
      Tensor with shape (Batch,sequence,dim)
    """
    for layer in self.layers:
      content_feature = layer(content_feature, style_feature)
    
    return content_feature


# Example
# import torch
# transModule_config = TransModule_Config(
#             nlayer=3,
#             d_model=768,
#             nhead=8,
#             mlp_ratio=4,
#             qkv_bias=False,
#             attn_drop=0.,
#             drop=0.,
#             drop_path=0.,
#             act_layer=nn.GELU,
#             norm_layer=nn.LayerNorm,
#             norm_first=True
#             )
# transModule = TransModule(transModule_config)
# tgt = torch.randn(1, 20, 768)
# memory = torch.randn(1, 10, 768)
# print(transModule(tgt, memory).shape)


########################################## Decoder ##########################################

decoder_stem = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)


class Decoder_MVGG(nn.Module):
  def __init__(self, d_model=768, seq_input=False):
      super(Decoder_MVGG, self).__init__()
      self.d_model = d_model
      self.seq_input = seq_input
      self.decoder = nn.Sequential(
        # Proccess Layer 1        

        # Upsample Layer 2
        nn.ReflectionPad2d(1),
        nn.Conv2d(int(self.d_model), 256, 3, 1, 0),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1, 0),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1, 0),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1, 0),
        nn.ReLU(),

        # Upsample Layer 3
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 128, 3, 1, 0),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 128, 3, 1, 0),
        nn.ReLU(),

        # Upsample Layer 4
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 64, 3, 1, 0),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 64, 3, 1, 0),
        nn.ReLU(),

        # Channel to 3
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 3, 3, 1, 0),
      )
        
        
  def forward(self, x, input_resolution):
    if self.seq_input == True:
      B, N, C = x.size()
#       H, W = math.ceil(self.img_H//self.patch_size), math.ceil(self.img_W//self.patch_size)
      (H, W) = input_resolution
      x = x.permute(0, 2, 1).reshape(B, C, H, W)
    x = self.decoder(x)  
    return x


# Example 1
# import torch
# decoder = Decoder_MVGG(d_model=768, seq_input=True)
# x = torch.randn(1, 3087, 768)
# y = decoder(x, input_resolution=(63, 49))
# print(y.shape)


########################################## Net ##########################################

class Net(nn.Module):
  def __init__(self, encoder, decoder, transModule, lossNet):
    super(Net, self).__init__()
    self.mse_loss = nn.MSELoss()
    self.grad_hist_loss = GradientHistogramLoss()  # add silk loss 1
    self.lbp_loss = LBPLoss()  # add silk loss 2
    self.wavelet_loss = WaveletLoss()  # add silk loss 3
    self.filtered_gram_loss = FilteredGramLoss(layer='relu3_1') # add silk lóss 4

# Thêm hook để lấy features từ VGG
    self.vgg = lossNet
    self.target_layer = None
    self.gen_feature = None
    self.style_feature = None
        
# Đăng ký hook cho layer relu3_1
    def get_features(module, input, output, name):
      if name == 'relu3_1':
        self.gen_feature = output
      elif name == 'style_relu3_1':
        self.style_feature = output
        
    for name, module in self.vgg.named_children():
      if name == 'feat_3':  # Layer relu3_1
        module.register_forward_hook(
          lambda m, i, o: get_features(m, i, o, 'relu3_1'))
      if name == 'feat_4':  # Layer relu4_1 (dùng cho style)
        module.register_forward_hook(
          lambda m, i, o: get_features(m, i, o, 'style_relu3_1'))
                  
    self.encoder = encoder
    self.decoder = decoder
    self.transModule = transModule

    # features of intermediate layers
    lossNet_layers = list(lossNet.children())
    self.feat_1 = nn.Sequential(*lossNet_layers[:4])  # input -> relu1_1
    self.feat_2 = nn.Sequential(*lossNet_layers[4:11]) # relu1_1 -> relu2_1
    self.feat_3 = nn.Sequential(*lossNet_layers[11:18]) # relu2_1 -> relu3_1
    self.feat_4 = nn.Sequential(*lossNet_layers[18:31]) # relu3_1 -> relu4_1
    self.feat_5 = nn.Sequential(*lossNet_layers[31:44]) # relu3_1 -> relu4_1

    # fix parameters
    for name in ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5']:
      for param in getattr(self, name).parameters():
        param.requires_grad = False


  # get intermediate features
  def get_interal_feature(self, input):
    result = []
    for i in range(5):
      input = getattr(self, 'feat_{:d}'.format(i+1))(input)
      result.append(input)
    return result
  

  def calc_content_loss(self, input, target, norm = False):
    assert input.size() == target.size(), 'To calculate loss needs the same shape between input and taget.'
    assert target.requires_grad == False, 'To calculate loss target shoud not require grad.'
    if norm == False:
        return self.mse_loss(input, target) 
    else:
        return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))


  def calc_style_loss(self, input, target):
    assert input.size() == target.size(), 'To calculate loss needs the same shape between input and taget.'
    assert target.requires_grad == False, 'To calculate loss target shoud not require grad.'
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return self.mse_loss(input_mean, target_mean) + \
        self.mse_loss(input_std, target_std)


  # calculate losses 
  ## i_c : content image
  ## i_s : style image (same size as content image)
  ## i_cs : generated image
  ## f_c : content feature
  ## f_s : style feature
  ## f_i_cs : generated image feature
  ## f_i_cc : content image feature
  ## f_i_ss : style image feature
  ## loss_c : content loss
  ## loss_s : style loss
  ## loss_id_1 : identity loss in pixel space
  ## loss_id_2 : identity loss in feature space
  ## loss_all : total loss
  def forward(self, i_c, i_s):
    # Encode Content and Style
    # Pass content and style images into encoder
    # f_c, f_s : the deep features representation 
    # f_c_reso: the original height x width resolution for decoder reshape
    f_c = self.encoder(i_c)
    f_s = self.encoder(i_s)
    f_c, f_c_reso = f_c[0], f_c[2]
    f_s, f_s_reso = f_s[0], f_s[2]
    
    # Feature Transformation (Transformer Decoder)
    f_cs = self.transModule(f_c, f_s) # blend content features with style features
    f_cc = self.transModule(f_c, f_c) # should reconstruct the original content features
    f_ss = self.transModule(f_s, f_s) # should reconstruct the original style features
    
    # Decode into Images
    i_cs = self.decoder(f_cs, f_c_reso) # stylized image
    i_cc = self.decoder(f_cc, f_c_reso) # indentity of content
    i_ss = self.decoder(f_ss, f_c_reso) # Identity of style

    # Add extra loss for silk texture
    loss_grad_hist = self.grad_hist_loss(i_cs, i_s)  # So sánh ảnh stylized và style
    loss_lbp = self.lbp_loss(i_cs, i_s)
    loss_wavelet = self.wavelet_loss(i_cs, i_s)  # Thêm dòng này

    # Forward qua VGG để lấy features (Loss4)
    _ = self.get_interal_feature(i_cs)  # Kích hoạt hook
    _ = self.get_interal_feature(i_s)
    loss_gram = self.filtered_gram_loss(self.gen_feature, self.style_feature)


    # Extract features from Output
    # These VGG-style features are used to compute c, s, id losses
    # get_interal_feature() run images through VGG layers to get relu1_1 to relu5_1
    f_c_loss = self.get_interal_feature(i_c)
    f_s_loss = self.get_interal_feature(i_s)
    f_i_cs_loss = self.get_interal_feature(i_cs)
    f_i_cc_loss = self.get_interal_feature(i_cc)
    f_i_ss_loss = self.get_interal_feature(i_ss)

    # Identity loss 1
    loss_id_1 = self.mse_loss(i_cc, i_c) + self.mse_loss(i_ss, i_s)

    loss_c, loss_s, loss_id_2 = 0, 0, 0
    
    loss_c = self.calc_content_loss(f_i_cs_loss[-2], f_c_loss[-2], norm=True) + \
             self.calc_content_loss(f_i_cs_loss[-1], f_c_loss[-1], norm=True)
    for i in range(1, 5):
      # Compare style statistics (mean and std) of stylized image vs style image from relu2_1 to relu5_1
      loss_s += self.calc_style_loss(f_i_cs_loss[i], f_s_loss[i])
      # Identity loss in feature space (how well i_cc =(xap xi) i_c in VGG features)
      loss_id_2 += self.mse_loss(f_i_cc_loss[i], f_c_loss[i]) + self.mse_loss(f_i_ss_loss[i], f_s_loss[i])
    
    return loss_c, loss_s, loss_id_1, loss_id_2, loss_grad_hist, loss_lbp, loss_wavelet,loss_gram, i_cs


# Example 1
# import torch
# from model.s2wat import S2WAT
# transModule_config = TransModule_Config(
#             nlayer=3,
#             d_model=384,
#             nhead=8,
#             mlp_ratio=4,
#             qkv_bias=False,
#             attn_drop=0.,
#             drop=0.,
#             drop_path=0.,
#             act_layer=nn.GELU,
#             norm_layer=nn.LayerNorm,
#             norm_first=True
#             )
# encoder = S2WAT(
#   img_size=224,
#   patch_size=2,
#   in_chans=3,
#   embed_dim=96,
#   depths=[2, 2, 2],
#   nhead=[3, 6, 12],
#   strip_width=[2, 4, 7],
#   drop_path_rate=0.,
#   patch_norm=True
# )
# transModule = TransModule(transModule_config)
# decoder = Decoder_MVGG(d_model=384, seq_input=True)
# vgg.load_state_dict(torch.load('../input/vggpretrainedmodel/vgg_normalised.pth'))
# net = Net(encoder, decoder, transModule, vgg)
# i_c = torch.randn(1, 3, 224, 224)
# i_s = torch.randn(1, 3, 224, 224)
# loss_c, loss_s, loss_id_1, loss_id_2, i_cs = net(i_c, i_s)
# print(loss_c.item(), loss_s.item(), loss_id_1.item(), loss_id_2.item())
# print(i_cs.shape)

###############################Gradient Histogram Loss####################################

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import wasserstein_distance
class GradientHistogramLoss(nn.Module):
  def __init__(self, bins=32, scales=[1.0, 0.5]):
    super().__init__()
    self.bins = bins
    self.scales = scales
    self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

  def forward(self, gen_img, target_img):
    total_loss = 0.0
    device = gen_img.device
    self.sobel_x = self.sobel_x.to(device)
    self.sobel_y = self.sobel_y.to(device)
        
    for scale in self.scales:
      h, w = int(gen_img.shape[2] * scale), int(gen_img.shape[3] * scale)
      gen_resized = F.interpolate(gen_img, size=(h, w), mode='bilinear')
      target_resized = F.interpolate(target_img, size=(h, w), mode='bilinear')
            
      grad_x_gen = F.conv2d(gen_resized, self.sobel_x, padding=1)
      grad_y_gen = F.conv2d(gen_resized, self.sobel_y, padding=1)
      grad_x_target = F.conv2d(target_resized, self.sobel_x, padding=1)
      grad_y_target = F.conv2d(target_resized, self.sobel_y, padding=1)
            
      theta_gen = torch.atan2(grad_y_gen, grad_x_gen) * (180 / np.pi)
      theta_target = torch.atan2(grad_y_target, grad_x_target) * (180 / np.pi)
            
      hist_gen = torch.histc(theta_gen, bins=self.bins, min=-180, max=180).cpu().numpy()
      hist_target = torch.histc(theta_target, bins=self.bins, min=-180, max=180).cpu().numpy()
            
      hist_gen = hist_gen / (h * w + 1e-6)
      hist_target = hist_target / (h * w + 1e-6)
            
      loss = wasserstein_distance(hist_gen, hist_target)
      total_loss += loss
        
      return total_loss / len(self.scales)
###############################################Second Loss############################################
class LBPLoss(nn.Module):
  def __init__(self, radius=1, neighbors=8):
    super().__init__()
    self.radius = radius
    self.neighbors = neighbors
    # Tạo kernel để lấy điểm lân cận
    self.unfold = nn.Unfold(kernel_size=3, padding=1)
  
  def forward(self, gen_img, target_img):
    # Chuyển ảnh sang grayscale
    gen_gray = 0.2989 * gen_img[:, 0] + 0.5870 * gen_img[:, 1] + 0.1140 * gen_img[:, 2]
    target_gray = 0.2989 * target_img[:, 0] + 0.5870 * target_img[:, 1] + 0.1140 * target_img[:, 2]
        
    # Lấy giá trị lân cận
    gen_patches = self.unfold(gen_gray.unsqueeze(1)).reshape(gen_img.shape[0], 9, -1)
    target_patches = self.unfold(target_gray.unsqueeze(1)).reshape(target_img.shape[0], 9, -1)
        
    # Tính LBP (so sánh với điểm trung tâm)
    center_gen = gen_patches[:, 4:5, :]  # Điểm trung tâm
    center_target = target_patches[:, 4:5, :]
        
    lbp_gen = (gen_patches > center_gen).float()
    lbp_target = (target_patches > center_target).float()
        
    # Tính L1 loss giữa các LBP patterns
    return F.l1_loss(lbp_gen, lbp_target)
##############################################Third Loss##################################################
class HaarWaveletTransform(nn.Module):
  def __init__(self):
    super().__init__()
    # Khởi tạo kernel Haar Wavelet thuần PyTorch
    self.register_buffer('ll_weight', torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 4.0)
    self.register_buffer('lh_weight', torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32) / 4.0)
    self.register_buffer('hl_weight', torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32) / 4.0)
    self.register_buffer('hh_weight', torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 4.0)

  def forward(self, x):
    b, c, h, w = x.size()
    x = x.view(b * c, 1, h, w)  # Gộp batch và channel
        
    # Tạo kernel 4D [out_ch, in_ch, H, W]
    kernel_size = 2
    ll_kernel = self.ll_weight.view(1, 1, kernel_size, kernel_size)
    lh_kernel = self.lh_weight.view(1, 1, kernel_size, kernel_size)
    hl_kernel = self.hl_weight.view(1, 1, kernel_size, kernel_size)
    hh_kernel = self.hh_weight.view(1, 1, kernel_size, kernel_size)
        
    # Áp dụng convolution với stride=2 để downsample
    ll = F.conv2d(x, ll_kernel, stride=2)
    lh = F.conv2d(x, lh_kernel, stride=2)
    hl = F.conv2d(x, hl_kernel, stride=2)
    hh = F.conv2d(x, hh_kernel, stride=2)
        
    return ll, lh, hl, hh
class WaveletLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.wavelet = HaarWaveletTransform()
    self.mse_loss = nn.MSELoss()

  def forward(self, gen_img, target_img):
    # Chuyển ảnh -> grayscale [B, 1, H, W]
    gen_gray = 0.2989 * gen_img[:, :1] + 0.5870 * gen_img[:, 1:2] + 0.1140 * gen_img[:, 2:3]
    target_gray = 0.2989 * target_img[:, :1] + 0.5870 * target_img[:, 1:2] + 0.1140 * target_img[:, 2:3]
        
    # Wavelet decomposition
    ll_gen, lh_gen, hl_gen, hh_gen = self.wavelet(gen_gray)
    ll_tar, lh_tar, hl_tar, hh_tar = self.wavelet(target_gray)
        
    # Focus on horizontal (LH) and vertical (HL) details
    loss = self.mse_loss(lh_gen, lh_tar) + \
    self.mse_loss(hl_gen, hl_tar) + \
    0.5 * self.mse_loss(hh_gen, hh_tar)
               
    return loss
  
  ##########################################Fourth Loss###############################################
class LaplacianFilter(nn.Module):
  def __init__(self):
    super().__init__()
    # Kernel Laplacian để bắt cạnh
    kernel = torch.tensor([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
    self.register_buffer('kernel', kernel)
    
  def forward(self, x):
    # Áp dụng filter cho từng channel
    b, c, h, w = x.size()
    x = x.view(b * c, 1, h, w)  # [B*C, 1, H, W]
    filtered = F.conv2d(x, self.kernel, padding=1)
    return filtered.view(b, c, h, w)

class FilteredGramLoss(nn.Module):
  def __init__(self, layer='relu3_1'):
    super().__init__()
    self.layer = layer
    self.laplacian = LaplacianFilter()
    
  def gram_matrix(self, x):
    b, c, h, w = x.size()
    x = x.view(b, c, -1)  # [B, C, H*W]
    return torch.bmm(x, x.transpose(1, 2)) / (c * h * w)  # [B, C, C]
    
  def forward(self, gen_features, target_features):
    # Lọc Laplacian
    gen_filtered = self.laplacian(gen_features)
    target_filtered = self.laplacian(target_features)
        
    # Tính Gram matrix
    gram_gen = self.gram_matrix(gen_filtered)
    gram_target = self.gram_matrix(target_filtered)
        
    return F.mse_loss(gram_gen, gram_target)