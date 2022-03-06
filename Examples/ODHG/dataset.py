class ODHG_Classifier(Dataset):

  def __init__(self, path, train=True,transform=None):
    
    self.train = train
    self.transform = transform
    if train:
      self.data_idx = np.loadtxt(path)
    else:
      self.data_idx = np.loadtxt(path + "test_gestures.txt")
    self.full_path = [path + "gesture_" + str(int(self.data_idx[i,0])) + "/finger_"+str(int(self.data_idx[i,1])) +
                      "/subject_" + str(int(self.data_idx[i,2])) + "/essai_" + str(int(self.data_idx[i,3])) + "/skeletons_world.txt" for i in range(np.shape(self.data_idx)[0])]    
    

  def __len__(self):
    return len(self.full_path)

  def __getitem__(self, idx):
    path = self.full_path[idx]
    data = np.loadtxt(path)[:,6:]
    row,col = np.shape(data)
    data = torch.from_numpy(data)
    data = data.T.reshape((1,col,row))
    data = nn.functional.interpolate(data,size = 500).squeeze().T
    data = vg.normalize(data)
    data = data.reshape((data.size(0),20,3,1))
    label = torch.tensor([int(self.data_idx[idx][0]-1)])
    sample = {"data":data,"label":label}
    return sample
