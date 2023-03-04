from WMRNet import generator
from PIL import Image
import torch
import os.path as osp
import os
from torchvision import datasets, transforms


G=generator(3,3)
G.eval()
G.load_state_dict(torch.load(os.path.join('WDNet_G.pkl'),map_location=torch.device('cpu')))
#G.cuda()
root = 'PS2_sample_images'
imageJ_path=osp.join(root,'%s.jpg')
img_save_path=osp.join('results','result_img','%s.jpg')
img_vision_path=osp.join('results','result_vision','%s.jpg')
ids = list()
for file in os.listdir(root):
			ids.append(file.strip('.jpg'))

for img_id in ids:
  transform_norm=transforms.Compose([transforms.ToTensor()])
  img_J=Image.open(imageJ_path%img_id)
  img_source = transform_norm(img_J)
  img_source=torch.unsqueeze(img_source,0)
  pred_target,mask,alpha,w,I_watermark=G(img_source)
  p0=torch.squeeze(img_source)
  p1=torch.squeeze(pred_target)
  p2=mask
  p3=torch.squeeze(w*mask)
  p2=torch.squeeze(torch.cat([p2,p2,p2],1))
  p0=torch.cat([p0,p1],1)
  p2=torch.cat([p2,p3],1)
  p0=torch.cat([p0,p2],2)
  p0=transforms.ToPILImage()(p0.detach().cpu()).convert('RGB')
  pred_target=transforms.ToPILImage()(p1.detach().cpu()).convert('RGB')
  pred_target.show()
  
  
  
  
