# summary function
def image_summary(image, labels):
    
    print('--------------')
    print('Image Details:')
    print('--------------')
    print(f'Image dimensions: {image.shape}')
    print('Channels:')
    
    if len(labels) == 1:
        image = image[..., np.newaxis]
        
    for i, lab in enumerate(labels):
        min_val = np.min(image[:,:,i])
        max_val = np.max(image[:,:,i])
        mean_val = np.mean(image[:,:,i])
        print(f'{lab} : min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}')


# combine L channel with ab and convert to rgb
def l_ab2rgb(l, ab):
  lab = l_ab2lab(l, ab)
  img = m_lab2rgb(lab)
  return img


# combine l and ab channels
def l_ab2lab(l, ab):
  if len(l.shape) == 2:
    l = reshape_l_up(l)
  if ab.shape[0] == 2:
      ab = ab.transpose((1,2,0))
  if l.shape[0] == 1:
    l = l.transpose((1,2,0))
  return np.dstack((l, ab))


# seperat ab channels
def ab2a_b(ab):
  if ab.shape[0] == 2:
    a = ab[0, :, :]
    b = ab[1, :, :]
  else:
    a = ab[:, :, 0]
    b = ab[:, :, 1]
  return a, b


def m_lab2rgb(lab):
  if lab.shape[0] == 3:
    rgb = cv2.cvtColor(lab.transpose(), cv2.COLOR_LAB2RGB)
  else:
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
  return rgb